from collections import ChainMap
from dis import Instruction
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.block_map import BlockMap
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    PythonBytecodeBlock,
    RegionBlock,
)
from numba_rvsdg.core.datastructures.labels import (
    PythonBytecodeLabel,
    ControlLabel,
    SyntheticForIter,
    SynthenticAssignment,
    SyntheticExitingLatch,
    SyntheticExit,
    SyntheticHead,
    SyntheticTail,
    SyntheticReturn,
)

import builtins


class Simulator:
    """BlockMap simulator.

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
    self.stack: List[Instruction]
        Instruction stack
    self.branch: Boolean
        Flag to be set during execution.
    self.return_value: Any
        The return value of the function.

    """

    def __init__(self, flow: ByteFlow, globals: dict):

        self.flow = flow
        self.globals = ChainMap(globals, builtins.__dict__)

        self.bcmap = {inst.offset: inst for inst in flow.bc}
        self.varmap = dict()
        self.ctrl_varmap = dict()
        self.stack = []
        self.branch = None
        self.return_value = None

    def run(self, args):
        """Run the given simulator with gievn args.

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
        target = PythonBytecodeLabel(index=0)
        while True:
            bb = self.flow.bbmap.graph[target]
            action = self.run_bb(bb, target)
            if "return" in action:
                return action["return"]
            target = action["jumpto"]

    def run_bb(self, bb: BasicBlock, target: Label):
        """Run a BasicBlock.

        Paramters
        ---------
        bb: BasicBlock
            The BasicBlock to execute.
        target: Label
            The Label of the BasicBlock

        Returns
        -------
        action: Dict[Str: Int or Boolean or Any]
            The action to be taken as a result of having executed the
            BasicBlock.

        """
        print("AT", target)
        if isinstance(bb, RegionBlock):
            return self._run_region(bb, target)

        if isinstance(target, ControlLabel):
            self.run_synth_block(target, bb)
        elif isinstance(target, PythonBytecodeLabel):
            assert type(bb) is PythonBytecodeBlock
            for inst in bb.get_instructions(self.bcmap):
                self.run_inst(inst)
        if bb.fallthrough:
            [target] = bb.jump_targets
            return {"jumpto": target}
        elif len(bb._jump_targets) == 2:
            [br_false, br_true] = bb._jump_targets
            return {"jumpto": br_true if self.branch else br_false}
        else:
            return {"return": self.return_value}

    def _run_region(self, region: RegionBlock, target):
        """Run region.

        Execute all BasicBlocks in this region. Stay within the region, only
        return the action when we jump out of the region or when we return from
        within the region.

        Parameters
        ----------
        region: RegionBlock
            The region to execute
        target: Label
            The Label for the RegionBlock

        Returns
        -------
        action: Dict[Str: Int or Boolean or Any]
            The action to be taken as a result of having executed the
            BasicBlock.

        """
        while True:
            bb = region.subregion[target]
            action = self.run_bb(bb, target)
            if "return" in action:
                return action
            elif "jumpto" in action:
                target = action["jumpto"]
                if target in region.subregion.graph:
                    continue
                else:
                    return action
            else:
                assert False, "unreachable"

    def run_synth_block(self, control_label, block):
        """Run a SyntheticBlock

        Paramaters
        ----------
        control_label: Label
            The Label for the block.
        block: SyntheticBlock
            The block itself.

        """
        print("----", control_label)
        print(f"control variable map: {self.ctrl_varmap}")
        handler = getattr(self, f"synth_{type(control_label).__name__}")
        handler(control_label, block)

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
    def synth_SyntheticForIter(self, control_label, block):
        self.op_FOR_ITER(None)

    def synth_SynthenticAssignment(self, control_label, block):
        self.ctrl_varmap.update(block.variable_assignment)

    def _synth_branch(self, control_label, block):
        jump_target = block.branch_value_table[self.ctrl_varmap[block.variable]]
        self.branch = bool(block._jump_targets.index(jump_target))

    def synth_SyntheticExitingLatch(self, control_label, block):
        self._synth_branch(control_label, block)

    def synth_SyntheticHead(self, control_label, block):
        self._synth_branch(control_label, block)

    def synth_SyntheticExit(self, control_label, block):
        self._synth_branch(control_label, block)

    def synth_SyntheticReturn(self, control_label, block):
        pass

    def synth_SyntheticTail(self, control_label, block):
        pass

    def synth_SyntheticBranch(self, control_label, block):
        pass

    ### Bytecode Instructions ###
    def op_LOAD_CONST(self, inst):
        self.stack.append(inst.argval)

    def op_COMPARE_OP(self, inst):
        arg1 = self.stack.pop()
        arg2 = self.stack.pop()
        self.stack.append(eval(f"{arg2} {inst.argval} {arg1}"))

    def op_LOAD_FAST(self, inst):
        self.stack.append(self.varmap[inst.argval])

    def op_LOAD_GLOBAL(self, inst):
        v = self.globals[inst.argval]
        if inst.argrepr.startswith("NULL"):
            append_null = True
            self.stack.append(v)
            self.stack.append(None)
        else:
            raise NotImplementedError

    def op_STORE_FAST(self, inst):
        val = self.stack.pop()
        self.varmap[inst.argval] = val

    def op_CALL_FUNCTION(self, inst):
        args = [self.stack.pop() for _ in range(inst.argval)][::-1]
        fn = self.stack.pop()
        res = fn(*args)
        self.stack.append(res)

    def op_GET_ITER(self, inst):
        val = self.stack.pop()
        res = iter(val)
        self.stack.append(res)

    def op_FOR_ITER(self, inst):
        tos = self.stack[-1]
        try:
            ind = next(tos)
        except StopIteration:
            self.stack.pop()
            self.branch = True
        else:
            self.branch = False
            self.stack.append(ind)

    def op_INPLACE_ADD(self, inst):
        rhs = self.stack.pop()
        lhs = self.stack.pop()
        lhs += rhs
        self.stack.append(lhs)

    def op_RETURN_VALUE(self, inst):
        v = self.stack.pop()
        self.return_value = v

    def op_JUMP_ABSOLUTE(self, inst):
        pass

    def op_JUMP_FORWARD(self, inst):
        pass

    def op_POP_JUMP_IF_FALSE(self, inst):
        self.branch = not self.stack.pop()

    def op_POP_JUMP_IF_TRUE(self, inst):
        self.branch = bool(self.stack.pop())

    def op_JUMP_IF_TRUE_OR_POP(self, inst):
        if self.stack[-1]:
            self.branch = True
        else:
            self.stack.pop()
            self.branch = False

    def op_JUMP_IF_FALSE_OR_POP(self, inst):
        if not self.stack[-1]:
            self.branch = True
        else:
            self.stack.pop()
            self.branch = False

    def op_POP_TOP(self, inst):
        self.stack.pop()

    def op_RESUME(self, inst):
        pass

    def op_PRECALL(self, inst):
        pass

    def op_CALL_FUNCTION(self, inst):
        args = [self.stack.pop() for _ in range(inst.argval)][::-1]
        fn = self.stack.pop()
        res = fn(*args)
        self.stack.append(res)

    def op_CALL(self, inst):
        args = [self.stack.pop() for _ in range(inst.argval)][::-1]
        first, second = self.stack.pop(), self.stack.pop()
        if first == None:
            func = second
        else:
            raise NotImplementedError
        res = func(*args)
        self.stack.append(res)

    def op_BINARY_OP(self, inst):
        rhs, lhs, op = self.stack.pop(), self.stack.pop(), inst.argrepr
        op = op if len(op) == 1 else op[0]
        self.stack.append(eval(f"{lhs} {op} {rhs}"))

    def op_JUMP_BACKWARD(self, inst):
        pass

    def op_POP_JUMP_FORWARD_IF_TRUE(self, inst):
        self.branch = self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_TRUE(self, inst):
        self.branch = self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_FORWARD_IF_FALSE(self, inst):
        self.branch = not self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_FALSE(self, inst):
        self.branch = not self.stack[-1]
        self.stack.pop()

    def op_POP_JUMP_FORWARD_IF_NOT_NONE(self, inst):
        self.branch = self.stack[-1] is not None
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_NOT_NONE(self, inst):
        self.branch = self.stack[-1] is not None
        self.stack.pop()

    def op_POP_JUMP_FORWARD_IF_NONE(self, inst):
        self.branch = self.stack[-1] is None
        self.stack.pop()

    def op_POP_JUMP_BACKWARD_IF_NONE(self, inst):
        self.branch = self.stack[-1] is None
        self.stack.pop()
