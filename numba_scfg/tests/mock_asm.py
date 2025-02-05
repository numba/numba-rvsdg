# mypy: ignore-errors

"""
Defines a mock assembly with a minimal operation semantic for testing
control flow transformation.
"""


from dataclasses import dataclass
from enum import IntEnum
from typing import IO, List, Tuple
import random


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
class PrintOperands(Operands):
    text: str


@dataclass(frozen=True)
class GotoOperands(Operands):
    jump_target: int


@dataclass(frozen=True)
class CtrOperands(Operands):
    counter: int


@dataclass(frozen=True)
class BrCtrOperands(Operands):
    true_target: int
    false_target: int


def parse(asm: str) -> list[Inst]:
    """Parse an assembly text into a list of instructions.

    See `test_mock_asm()` for an example.
    """
    # pass 1: scan for labels
    labelmap: dict[str, int] = {}
    todos: List[Tuple[str, List[str]]] = []
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
                target = self.rng.randrange(size)
                bb.append(f"{indent}goto BB{target}")
            elif kind == "brctr":
                target0 = self.rng.randrange(size)
                target1 = self.rng.randrange(size)
                ctr = self.rng.randrange(1, 10)
                bb.append(f"{indent}ctr {ctr}")
                bb.append(f"{indent}brctr BB{target0} BB{target1}")
            else:
                # fallthrough
                assert kind == ""
            source.extend(bb)
        return "\n".join(source)
