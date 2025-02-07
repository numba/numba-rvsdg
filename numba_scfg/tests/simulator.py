# mypy: ignore-errors

from collections import ChainMap
from dis import Instruction
from numba_scfg.core.datastructures.byte_flow import ByteFlow
from numba_scfg.core.datastructures.basic_block import (
    PythonBytecodeBlock,
    RegionBlock,
    SyntheticBlock,
)
from numba_scfg.core.utils import PYVERSION

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
    region_stack: List[RegionBlocks]
        Stack to hold the recusion level for regions
    trace: List[Tuple(name, block)]
        List of name, block combinations visisted
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
        self.region_stack = []
        self.trace = []
        self.branch = None
        self.return_value = None

    def get_block(self, name: str):
        """Return the BasicBlock object for a give name.

        This method is aware of the recusion level of the `Simulator` into the
        `region_stack`. That is to say, if we have recursed into regions, the
        BasicBlock is  returned from the current region (the top region of the
        region_stack). Otherwise the BasicBlock is returned from the initial
        ByteFlow supplied to the simulator. The method `run_RegionBlock` is
        responsible for maintaining the `region_stack`.

        Parameters
        ----------
        name: str
            The name for which to fetch the BasicBlock

        Return
        ------
        block: BasicBlock
            The requested block

        """
        # Recursed into regions, return block from region
        if self.region_stack:
            return self.region_stack[-1].subregion[name]
        # Not recursed into regions, return block from ByteFlow
        else:
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
        name = self.scfg.find_head()
        while True:
            action = self.run_BasicBlock(name)
            if "return" in action:
                return action["return"]
            name = action["jumpto"]

    def run_BasicBlock(self, name: str):
        """Run a BasicBlock.

        Paramters
        ---------
        name: str
            The str of the BasicBlock

        Returns
        -------
        action: Dict[Str: Int or Boolean or Any]
            The action to be taken as a result of having executed the
            BasicBlock.

        """
        print("AT", name)
        block = self.get_block(name)
        self.trace.append((name, block))
        if isinstance(block, RegionBlock):
            return self.run_RegionBlock(name)
        elif isinstance(block, PythonBytecodeBlock):
            self.run_PythonBytecodeBlock(name)
        elif isinstance(block, SyntheticBlock):
            self.run_synth_block(name)
        if block.fallthrough:
            [name] = block.jump_targets
            return {"jumpto": name}
        elif len(block._jump_targets) == 2:
            [br_false, br_true] = block._jump_targets
            return {"jumpto": br_true if self.branch else br_false}
        else:
            return {"return": self.return_value}

    def run_RegionBlock(self, name: str):
        """Run region.

        Execute all BasicBlocks in this region. Stay within the region, only
        return the action when we jump out of the region or when we return
        from within the region.

        Special attention is directed at the use of the `region_stack` here.
        Since the blocks for the subregion are stored in the
        `region.subregion` graph, we need to use a region aware
        `get_blocks` in methods such as `run_BasicBlock` so that we get
        the correct `BasicBlock`. The net effect of placing the `region`
        onto the `region_stack` is that `run_BasicBlock` will be able to
        fetch the correct name from the `region.subregion` graph, and thus
        be able to run the correct sequence of blocks.

        Parameters
        ----------
        name: str
            The str for the RegionBlock

        Returns
        -------
        action: Dict[Str: Int or Boolean or Any]
            The action to be taken as a result of having executed the
            BasicBlock.

        """
        # Get the RegionBlock and place it onto the region_stack
        region: RegionBlock = self.get_block(name)
        self.region_stack.append(region)
        while True:
            # If the given name is of the same region,
            # we execute it's first block, i.e. head
            if name == region.name:
                name = region.subregion.find_head()
            # Execute the first block of the region.
            action = self.run_BasicBlock(name)
            # If we need to return, break and do so
            if "return" in action:
                break  # break and return action
            elif "jumpto" in action:
                name = action["jumpto"]
                # Otherwise check if we stay in the region and break otherwise
                if name in region.subregion.graph:
                    continue  # stay in the region
                else:
                    break  # break and return action
            else:
                assert False, "unreachable"  # in case of coding errors
        # Pop the region from the region stack again and return the final
        # action for this region
        popped = self.region_stack.pop()
        assert popped == region
        return action

    def run_PythonBytecodeBlock(self, name: str):
        """Run PythonBytecodeBlock

        Parameters
        ----------
        name: PythonBytecodename
            The str for the block.

        """
        block: PythonBytecodeBlock = self.get_block(name)
        assert type(block) is PythonBytecodeBlock
        for inst in block.get_instructions(self.bcmap):
            self.run_inst(inst)

    def run_synth_block(self, name: str):
        """Run a SyntheticBlock

        Paramaters
        ----------
        name: Controlname
            The str for the block.

        """
        print("----", name)
        print(f"control variable map: {self.ctrl_varmap}")
        block = self.get_block(name)
        handler = getattr(self, "synth_" + block.__class__.__name__)
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

    ### Synthetic Instructions ### # noqa
    def synth_SyntheticAssignment(self, control_name, block):
        self.ctrl_varmap.update(block.variable_assignment)

    def _synth_branch(self, control_name, block):
        jump_target = block.branch_value_table[
            self.ctrl_varmap[block.variable]
        ]
        self.branch = bool(block._jump_targets.index(jump_target))

    def synth_SyntheticExitingLatch(self, control_name, block):
        self._synth_branch(control_name, block)

    def synth_SyntheticExitBranch(self, control_name, block):
        self._synth_branch(control_name, block)

    def synth_SyntheticHead(self, control_name, block):
        self._synth_branch(control_name, block)

    def synth_SyntheticExit(self, control_name, block):
        pass

    def synth_SyntheticTail(self, control_name, block):
        pass

    def synth_SyntheticReturn(self, control_name, block):
        pass

    def synth_SyntheticFill(self, control_name, block):
        pass

    def synth_SyntheticBlock(self, control_name, block):
        raise NotImplementedError("SyntheticBlock should not be instantiated")

    def synth_SyntheticBranch(self, control_name, block):
        raise NotImplementedError("SyntheticBranch should not be instantiated")

    ### Bytecode Instructions ### # noqa
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
            self.branch = True
            if PYVERSION in ((3, 11),):
                self.stack.pop()
            elif PYVERSION in ((3, 12),):
                self.stack.append(None)
            else:
                raise NotImplementedError(PYVERSION)
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

    def op_CALL(self, inst):
        args = [self.stack.pop() for _ in range(inst.argval)][::-1]
        first, second = self.stack.pop(), self.stack.pop()
        if first is None:
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

    if PYVERSION in ((3, 12),):

        def op_END_FOR(self, inst):
            self.stack.pop()
            self.stack.pop()

    else:

        def op_END_FOR(self, inst):
            raise NotImplementedError(PYVERSION)

    def op_COPY(self, inst):
        assert inst.argval > 0
        self.stack.append(self.stack[-inst.argval])
