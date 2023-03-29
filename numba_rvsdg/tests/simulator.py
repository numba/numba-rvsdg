from collections import ChainMap
from dis import Instruction
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    PythonBytecodeBlock,
    ControlVariableBlock,
    BranchBlock,
)
from numba_rvsdg.core.datastructures.labels import (
    Label,
    PythonBytecodeLabel,
    ControlLabel,
    SynthenticAssignment,
    SyntheticExitingLatch,
    SyntheticExit,
    SyntheticHead,
    SyntheticTail,
    SyntheticReturn,
    BlockName,
)

import builtins


class Simulator:
    """SCFG simulator.

    This is a simulator utility to be used for testing.

    This simulator will take a given structured control flow graph (SCFG) and
    simulate it. It contains a tiny barebones Python interpreter with stack and
    registers to simulate any Python bytecode Instructions present in any
    PythonBytecodeBlock. For any SyntheticBlock it will determine where and how
    to jump based on the values of the control variables.

    Parameters
    ----------
    flow: ByteFlow
        The ByteFlow to be simulated.
    globals: dict of any
        The globals to become available during simulation

    Attributes
    ----------
    bcmap: Dict[int, Instruction]
        Mapping of bytecode offset to instruction.
    varmap: Dict[Str, Any]
        Python variable map more or less a register
    ctrl_varmap: Dict[Str, int]
        Control variable map
    stack: List[Instruction]
        Instruction stack
    trace: List[Tuple(name, block)]
        List of names, block combinations visisted
    branch: Boolean
        Flag to be set during execution.
    return_value: Any
        The return value of the function.

    """

    def __init__(self, flow: ByteFlow, globals: dict):

        self.flow = flow
        self.scfg = flow.scfg
        self.globals = ChainMap(globals, builtins.__dict__)

        self.bcmap = {inst.offset: inst for inst in flow.bc}
        self.varmap = dict()
        self.ctrl_varmap = dict()
        self.stack = []
        self.trace = []
        self.branch = None
        self.return_value = None

    def get_block(self, name: BlockName):
        """Return the BasicBlock object for a give name.

        This method is aware of the recusion level of the `Simulator` into the
        `region_stack`. That is to say, if we have recursed into regions, the
        BasicBlock is  returned from the current region (the top region of the
        region_stack). Otherwise the BasicBlock is returned from the initial
        ByteFlow supplied to the simulator. The method `run_RegionBlock` is
        responsible for maintaining the `region_stack`.

        Parameters
        ----------
        name: BlockName
            The name for which to fetch the BasicBlock

        Return
        ------
        block: BasicBlock
            The requested block

        """
        return self.flow.scfg[name]

    def run(self, args):
        """Run the given simulator with given args.

        Parameters
        ----------
        args: Dict[Any, Any]
            Arguments for function execution

        Returns
        -------
        result: Any
            The result of the simulation.

        """
        self.varmap.update(args)
        name = self.flow.scfg.find_head()
        while True:
            action = self.run_BasicBlock(name)
            if "return" in action:
                return action["return"]
            name = action["jumpto"]

    def run_BasicBlock(self, name: BlockName):
        """Run a BasicBlock.

        Paramters
        ---------
        name: BlockName
            The BlockName of the BasicBlock

        Returns
        -------
        action: Dict[Str: Int or Boolean or Any]
            The action to be taken as a result of having executed the
            BasicBlock.

        """
        print("AT", name)
        block = self.get_block(name)
        self.trace.append((name, block))

        if isinstance(block.label, ControlLabel):
            self.run_synth_block(name)
        elif isinstance(block.label, PythonBytecodeLabel):
            self.run_PythonBytecodeBlock(name)
        if len(self.scfg.out_edges[name]) == 1:
            [name] = self.scfg.out_edges[name]
            return {"jumpto": name}
        elif len(self.scfg.out_edges[name]) == 2:
            [br_false, br_true] = self.scfg.out_edges[name]
            return {"jumpto": br_true if self.branch else br_false}
        else:
            return {"return": self.return_value}

    def run_PythonBytecodeBlock(self, name: BlockName):
        """Run PythonBytecodeBlock

        Parameters
        ----------
        name: BlockName
            The BlockName for the block.

        """
        block: PythonBytecodeBlock = self.get_block(name)
        assert type(block) is PythonBytecodeBlock
        for inst in block.get_instructions(self.bcmap):
            self.run_inst(inst)

    def run_synth_block(self, name: BlockName):
        """Run a SyntheticBlock

        Paramaters
        ----------
        name: BlockName
            The BlockName for the block.

        """
        print("----", name)
        print(f"control variable map: {self.ctrl_varmap}")
        block = self.get_block(name)
        handler = getattr(self, f"synth_{type(name).__name__}")
        handler(name, block)

    def run_inst(self, inst: Instruction):
        """Run a bytecode Instruction

        Paramaters
        ----------
        inst: Instruction
            The Python bytecode instruction to execute.

        """
        print("----", inst)
        print(f"variable map before: {self.varmap}")
        print(f"stack before: {self.stack}")
        handler = getattr(self, f"op_{inst.opname}")
        handler(inst)
        print(f"variable map after: {self.varmap}")
        print(f"stack after: {self.stack}")

    ### Synthetic Instructions ###
    def synth_SynthenticAssignment(
        self, control_label: BlockName, block: ControlVariableBlock
    ):
        self.ctrl_varmap.update(block.variable_assignment)

    def _synth_branch(self, control_label: BlockName, block: BranchBlock):
        jump_target = block.branch_value_table[self.ctrl_varmap[block.variable]]
        self.branch = bool(block._jump_targets.index(jump_target))

    def synth_SyntheticExitingLatch(
        self, control_label: BlockName, block: ControlVariableBlock
    ):
        self._synth_branch(control_label, block)

    def synth_SyntheticHead(
        self, control_label: BlockName, block: ControlVariableBlock
    ):
        self._synth_branch(control_label, block)

    def synth_SyntheticExit(
        self, control_label: BlockName, block: ControlVariableBlock
    ):
        self._synth_branch(control_label, block)

    def synth_SyntheticReturn(
        self, control_label: BlockName, block: ControlVariableBlock
    ):
        pass

    def synth_SyntheticTail(
        self, control_label: BlockName, block: ControlVariableBlock
    ):
        pass

    def synth_SyntheticBranch(
        self, control_label: BlockName, block: ControlVariableBlock
    ):
        pass

    ### Bytecode Instructions ###
    def op_LOAD_CONST(self, inst: Instruction):
        self.stack.append(inst.argval)

    def op_COMPARE_OP(self, inst: Instruction):
        arg1 = self.stack.pop()
        arg2 = self.stack.pop()
        self.stack.append(eval(f"{arg2} {inst.argval} {arg1}"))

    def op_LOAD_FAST(self, inst: Instruction):
        self.stack.append(self.varmap[inst.argval])

    def op_LOAD_GLOBAL(self, inst: Instruction):
        v = self.globals[inst.argval]
        if inst.argrepr.startswith("NULL"):
            append_null = True
            self.stack.append(v)
            self.stack.append(None)
        else:
            raise NotImplementedError

    def op_STORE_FAST(self, inst: Instruction):
        val = self.stack.pop()
        self.varmap[inst.argval] = val

    def op_CALL_FUNCTION(self, inst: Instruction):
        args = [self.stack.pop() for _ in range(inst.argval)][::-1]
        fn = self.stack.pop()
        res = fn(*args)
        self.stack.append(res)

    def op_GET_ITER(self, inst: Instruction):
        val = self.stack.pop()
        res = iter(val)
        self.stack.append(res)

    def op_FOR_ITER(self, inst: Instruction):
        tos = self.stack[-1]
        try:
            ind = next(tos)
        except StopIteration:
            self.stack.pop()
            self.branch = True
        else:
            self.branch = False
            self.stack.append(ind)

    def op_INPLACE_ADD(self, inst: Instruction):
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        lhs += rhs
        self.stack.append(lhs)

    def op_RETURN_VALUE(self, inst: Instruction):
        v = self.stack.pop()
        self.return_value = v

    def op_JUMP_ABSOLUTE(self, inst: Instruction):
        pass

    def op_JUMP_FORWARD(self, inst: Instruction):
        pass

    def op_POP_JUMP_IF_FALSE(self, inst: Instruction):
        self.branch = not self.stack.pop()

    def op_POP_JUMP_IF_TRUE(self, inst: Instruction):
        self.branch = bool(self.stack.pop())

    def op_JUMP_IF_TRUE_OR_POP(self, inst: Instruction):
        if self.stack[-1]:
            self.branch = True
        else:
            self.stack.pop()
            self.branch = False

    def op_JUMP_IF_FALSE_OR_POP(self, inst: Instruction):
        if not self.stack[-1]:
            self.branch = True
        else:
            self.stack.pop()
            self.branch = False

    def op_POP_TOP(self, inst: Instruction):
        self.stack.pop()

    def op_RESUME(self, inst: Instruction):
        pass

    def op_PRECALL(self, inst: Instruction):
        pass

    def op_CALL_FUNCTION(self, inst: Instruction):
        args = [self.stack.pop() for _ in range(inst.argval)][::-1]
        fn = self.stack.pop()
        res = fn(*args)
        self.stack.append(res)

    def op_CALL(self, inst: Instruction):
        args = [self.stack.pop() for _ in range(inst.argval)][::-1]
        first, second = self.stack.pop(), self.stack.pop()
        if first == None:
            func = second
        else:
            raise NotImplementedError
        res = func(*args)
        self.stack.append(res)

    def op_BINARY_OP(self, inst: Instruction):
        rhs, lhs, op = self.stack.pop(), self.stack.pop(), inst.argrepr
        op = op if len(op) == 1 else op[0]
        self.stack.append(eval(f"{lhs} {op} {rhs}"))

    def op_JUMP_BACKWARD(self, inst: Instruction):
        pass

    def op_POP_JUMP_FORWARD_IF_TRUE(self, inst: Instruction):
        self.branch = self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_TRUE(self, inst: Instruction):
        self.branch = self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_FORWARD_IF_FALSE(self, inst: Instruction):
        self.branch = not self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_FALSE(self, inst: Instruction):
        self.branch = not self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, inst: Instruction):
        self.branch = self.stack[-1] is not None
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_NOT_NONE(self, inst: Instruction):
        self.branch = self.stack[-1] is not None
        self.stack.pop()

    def op_POP_JUMP_FORWARD_IF_NONE(self, inst: Instruction):
        self.branch = self.stack[-1] is None
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_NONE(self, inst: Instruction):
        self.branch = self.stack[-1] is None
        self.stack.pop()
