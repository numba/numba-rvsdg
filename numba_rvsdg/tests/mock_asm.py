"""
Defines a mock assembly with a minimal operation semantic for testing
control flow transformation.
"""


from dataclasses import dataclass, field
from enum import IntEnum
from io import StringIO
from typing import IO
import random
import os

from numba_rvsdg.rendering.rendering import SCFGRenderer
from numba_rvsdg.core.datastructures.scfg import SCFG, NameGenerator
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    SyntheticBlock,
)
from numba_rvsdg.core.transformations import (
    restructure_loop,
    restructure_branch,
)


DEBUGGRAPH = int(os.environ.get("MOCKASM_DEBUGGRAPH", 0))


class Opcode(IntEnum):
    PRINT = 1
    """Instruction format: print <TEXT>

    Semantic:
    - Print the *TEXT*.
    """

    GOTO = 2
    """Instruction format: goto <LABEL>

    Semantic:
    - Set PC to *LABEL*.
    """

    CTR = 3
    """Instruction format: ctr <COUNT>

    Semantic:
    1. Set a counter associated for this PC location to *COUNT*
       if it is not already set.
    2. Compute Counter[PC] = max(0, CTR - 1)
    3. Store Counter[PC] to LAST_CTR register

    """

    BRCTR = 4
    """Instruction format: brctr <TRUE_LABEL> <FALSE_LABEL>

    Semantic:
    - If LAST_CTR value is not zero, set PC to *TRUE_LABEL*
    - else, set PC to *FALSE_LABEL*
    """


@dataclass(frozen=True)
class Operands:
    pass


@dataclass(frozen=True)
class Inst:
    opcode: Opcode
    operands: Operands


@dataclass(frozen=True)
class PrintOperands:
    text: str


@dataclass(frozen=True)
class GotoOperands:
    jump_target: int


@dataclass(frozen=True)
class CtrOperands:
    counter: int


@dataclass(frozen=True)
class BrCtrOperands:
    true_target: int
    false_target: int


def parse(asm: str) -> list[Inst]:
    """Parse an assembly text into a list of instructions.

    See `test_mock_asm()` for an example.
    """
    # pass 1: scan for labels
    labelmap: dict[str, int] = {}
    todos = []
    for line in asm.splitlines():
        line = line.strip()
        if not line:
            continue
        head, *tail = line.split()
        if head == "label":
            [label_name] = tail
            labelmap[label_name] = len(todos)
        else:
            todos.append((head, tail))

    # pass 2: parse the instructions
    instlist: list[Inst] = []
    for head, tail in todos:
        if head == "print":
            [text] = tail
            inst = Inst(Opcode.PRINT, PrintOperands(text))
        elif head == "goto":
            [label] = tail
            inst = Inst(Opcode.GOTO, GotoOperands(labelmap[label]))
        elif head == "ctr":
            [counter] = tail
            inst = Inst(Opcode.CTR, CtrOperands(int(counter)))
        elif head == "brctr":
            [true_label, false_label] = tail
            inst = Inst(
                Opcode.BRCTR,
                BrCtrOperands(labelmap[true_label], labelmap[false_label]),
            )
        else:
            assert False, f"invalid instruction {head!r}"
        instlist.append(inst)
    return instlist


class VM:
    """A virtual machine for the assembly language.

    This assembly is defined for a virtual machine that execute a simple
    instruction stream. The VM has the following states:
    - program counter (PC)
    - an output buffer for printing;
    - a LAST_CTR register to store the last accessed counter.
    - a table that maps PC location of `ctr` instruction to the counter
      value

    The VM do not have another other input source beside the instruction
    stream. Therefore, a program behavior is known statically.

    """

    def __init__(self, outbuf: IO[str]):
        self._outbuf = outbuf
        self._last_ctr = None
        self._ctrmap: dict[int, int] = {}

    def run(self, instlist: list[Inst], max_step: int | None = None) -> bool:
        """Execute a list of instructions.

        Each VM instance should only call `.run()` once.

        Program starts at PC=0 and runs until the end of the instruction list
        (PC >= length of instruction list) or `max_step` is reached

        Returns whether program terminates by reaching the end.
        """
        step = 0
        pc = 0
        while pc < len(instlist):
            if max_step is not None and step >= max_step:
                return False
            pc = self.eval_inst(pc, instlist[pc])
            step += 1

        return True

    def eval_inst(self, pc: int, inst: Inst) -> int:
        """Evaluate a single instruction.

        Returns the next PC.
        """
        outbuf = self._outbuf
        ctrmap = self._ctrmap
        if inst.opcode == Opcode.PRINT:
            print(inst.operands.text, file=outbuf)
            return pc + 1
        elif inst.opcode == Opcode.GOTO:
            pc = inst.operands.jump_target
            return pc
        elif inst.opcode == Opcode.CTR:
            operands: CtrOperands = inst.operands
            # Set default state to the counter if not already defined
            ctr = ctrmap.setdefault(pc, operands.counter)
            # Decrement counter
            ctrmap[pc] = ctr = max(0, ctr - 1)
            # Store to LAST_CTR
            self._last_ctr = ctr
            return pc + 1
        elif inst.opcode == Opcode.BRCTR:
            operands: BrCtrOperands = inst.operands
            # Read the LAST_CTR
            ctr = self._last_ctr
            assert ctr >= 0
            # Do the jump
            if ctr != 0:
                return inst.operands.true_target
            else:
                return inst.operands.false_target
        else:
            raise AssertionError(f"invalid instruction {inst}")


class ProgramGen:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def generate_program(self, min_base_length=5, max_base_length=30) -> str:
        size = self.rng.randrange(min_base_length, max_base_length)
        source = []

        # generate BB
        indent = " " * 4
        for i in range(size):
            bb: list[str] = [f"label BB{i}", f"{indent}print P{i}"]
            [kind] = self.rng.choices(["goto", "brctr", ""], [1, 10, 20])
            if kind == "goto":
                target = self.rng.randrange(1, size) # avoid jump back to entry
                bb.append(f"{indent}goto BB{target}")
            elif kind == "brctr":
                target0 = self.rng.randrange(1, size) # avoid jump back to entry
                target1 = self.rng.randrange(1, size) # avoid jump back to entry
                while target1 == target0:
                    # avoid same target on both side
                    target1 = self.rng.randrange(1, size)
                ctr = self.rng.randrange(1, 10)
                bb.append(f"{indent}ctr {ctr}")
                bb.append(f"{indent}brctr BB{target0} BB{target1}")
            else:
                # fallthrough
                assert kind == ""
            source.extend(bb)
        return "\n".join(source)


# The code below are for simulating SCFG transformed MockAsm


@dataclass(frozen=True)
class MockAsmBasicBlock(BasicBlock):
    bbinstlist: list[Inst] = field(default_factory=list)
    bboffset: int = 0
    bbtargets: tuple[int, ...] = ()


def _iter_subregions(scfg: "SCFG"):
    for node in scfg.graph.values():
        if isinstance(node, RegionBlock):
            yield node
            yield from _iter_subregions(node.subregion)


def recursive_restructure_loop(scfg: "SCFG"):
    restructure_loop(scfg.region)
    for region in _iter_subregions(scfg):
        restructure_loop(region)


def recursive_restructure_branch(scfg: "SCFG"):
    restructure_branch(scfg.region)
    for region in _iter_subregions(scfg):
        restructure_branch(region)


def to_scfg(instlist: list[Inst]) -> SCFG:
    labels = set([0, len(instlist)])
    for pc, inst in enumerate(instlist):
        if isinstance(inst.operands, GotoOperands):
            labels.add(inst.operands.jump_target)
            if pc + 1 < len(instlist):
                labels.add(pc + 1)
        elif isinstance(inst.operands, BrCtrOperands):
            labels.add(inst.operands.true_target)
            labels.add(inst.operands.false_target)
            if pc + 1 < len(instlist):
                labels.add(pc + 1)
    block_map_graph = {}
    scfg = SCFG(block_map_graph, NameGenerator())
    bb_offsets = sorted(labels)
    labelmap = {}

    for begin, end in zip(bb_offsets, bb_offsets[1:]):
        labelmap[begin] = label = scfg.name_gen.new_block_name("mock")

    for begin, end in zip(bb_offsets, bb_offsets[1:]):
        bb = instlist[begin:end]
        inst = bb[-1]  # terminator
        if isinstance(inst.operands, GotoOperands):
            targets = [inst.operands.jump_target]
        elif isinstance(inst.operands, BrCtrOperands):
            targets = [inst.operands.true_target, inst.operands.false_target]
        elif end < len(instlist):
            targets = [end]
        else:
            targets = []

        label = labelmap[begin]
        block = MockAsmBasicBlock(
            name=label,
            bbinstlist=bb,
            bboffset=begin,
            bbtargets=tuple(targets),
            _jump_targets=tuple(labelmap[tgt] for tgt in targets),
        )
        scfg.add_block(block)

    # remove dead code from reachabiliy of entry block
    reachable = set([labelmap[0]])
    stack = [labelmap[0]]
    while stack:
        blk: BasicBlock = scfg.graph[stack.pop()]
        for k in blk._jump_targets:
            if k not in reachable:
                stack.append(k)
                reachable.add(k)
    scfg.remove_blocks(set(scfg.graph.keys()) - reachable)

    print("YAML\n", scfg.to_yaml())
    scfg.join_returns()
    if DEBUGGRAPH:
        MockAsmRenderer(scfg).view("jointed")
    recursive_restructure_loop(scfg)
    if DEBUGGRAPH:
        MockAsmRenderer(scfg).view("loop")
    recursive_restructure_branch(scfg)
    if DEBUGGRAPH:
        MockAsmRenderer(scfg).view("branch")
    return scfg


class MockAsmRenderer(SCFGRenderer):
    def render_block(self, digraph, name: str, block: BasicBlock):
        if isinstance(block, MockAsmBasicBlock):
            # Extend base renderer

            # format bbinstlist
            instbody = []
            for inst in block.bbinstlist:
                instbody.append(f"\l    {inst}")

            body = (
                name
                + "\l"
                + "\n"
                + "".join(instbody)
                + "\n"
                + "\njump targets: "
                + str(block.jump_targets)
                + "\nback edges: "
                + str(block.backedges)
            )

            digraph.node(str(name), shape="rect", label=body)
        else:
            super().render_block(digraph, name, block)


class MaxStepError(Exception):
    pass


class Simulator:
    DEBUG = False

    def __init__(self, scfg: SCFG, buf: StringIO, max_step):
        self.vm = VM(buf)
        self.scfg = scfg
        self.region_stack = []
        self.ctrl_varmap = dict()
        self.max_step = max_step
        self.step = 0

    def _debug_print(self, *args, **kwargs):
        if self.DEBUG:
            print(*args, **kwargs)

    def run(self):
        scfg = self.scfg
        label = scfg.find_head()
        while True:
            action = self.run_block(self.scfg.graph[label])
            # If we need to return, break and do so
            if "return" in action:
                break  # break and return action
            elif "jumpto" in action:
                label = action["jumpto"]
                # Otherwise check if we stay in the region and break otherwise
                if label in self.scfg.graph:
                    continue  # stay in the region
                else:
                    break  # break and return action
            else:
                assert False, "unreachable"  # in case of coding errors

    def run_block(self, block):
        self._debug_print("run block", block.name)
        if isinstance(block, RegionBlock):
            return self.run_RegionBlock(block)
        elif isinstance(block, MockAsmBasicBlock):
            return self.run_MockAsmBasicBlock(block)
        elif isinstance(block, SyntheticBlock):
            self._debug_print("    ", block)
            label = block.name
            handler = getattr(self, f"synth_{type(block).__name__}")
            out = handler(label, block)
            self._debug_print("    ctrl_varmap dump:", self.ctrl_varmap)
            return out
        else:
            assert False, type(block)

    def run_RegionBlock(self, block: RegionBlock):
        self.region_stack.append(block)

        label = block.subregion.find_head()
        while True:
            action = self.run_block(block.subregion.graph[label])
            # If we need to return, break and do so
            if "return" in action:
                break  # break and return action
            elif "jumpto" in action:
                label = action["jumpto"]
                # Otherwise check if we stay in the region and break otherwise
                if label in block.subregion.graph:
                    continue  # stay in the region
                else:
                    break  # break and return action
            else:
                assert False, "unreachable"  # in case of coding errors

        self.region_stack.pop()
        return action

    def run_MockAsmBasicBlock(self, block: MockAsmBasicBlock):
        vm = self.vm
        pc = block.bboffset

        if self.step > self.max_step:
            raise MaxStepError("step > max_step")

        for inst in block.bbinstlist:
            self._debug_print("inst", pc, inst)
            pc = vm.eval_inst(pc, inst)
            self.step += 1
        if block.bbtargets:
            pos = block.bbtargets.index(pc)
            label = block._jump_targets[pos]
            return {"jumpto": label}
        else:
            return {"return": None}

    ### Synthetic Instructions ###
    def synth_SyntheticAssignment(self, control_label, block):
        self.ctrl_varmap.update(block.variable_assignment)
        [label] = block.jump_targets
        return {"jumpto": label}

    def _synth_branch(self, control_label, block):
        jump_target = block.branch_value_table[
            self.ctrl_varmap[block.variable]
        ]
        return {"jumpto": jump_target}

    def synth_SyntheticExitingLatch(self, control_label, block):
        return self._synth_branch(control_label, block)

    def synth_SyntheticHead(self, control_label, block):
        return self._synth_branch(control_label, block)

    def synth_SyntheticExitBranch(self, control_label, block):
        return self._synth_branch(control_label, block)

    def synth_SyntheticFill(self, control_label, block):
        [label] = block.jump_targets
        return {"jumpto": label}

    def synth_SyntheticReturn(self, control_label, block):
        [label] = block.jump_targets
        return {"jumpto": label}

    def synth_SyntheticTail(self, control_label, block):
        [label] = block.jump_targets
        return {"jumpto": label}

    def synth_SyntheticBranch(self, control_label, block):
        [label] = block.jump_targets
        return {"jumpto": label}


def simulate_scfg(scfg: SCFG):
    with StringIO() as buf:
        Simulator(scfg, buf, max_step=1000).run()
        return buf.getvalue()
