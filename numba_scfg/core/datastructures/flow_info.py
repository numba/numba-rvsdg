import dis

from typing import Set, Tuple, Dict, Sequence, Optional
from dataclasses import dataclass, field

from numba_scfg.core.datastructures.basic_block import PythonBytecodeBlock
from numba_scfg.core.datastructures import block_names
from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.core.utils import (
    is_conditional_jump,
    _next_inst_offset,
    is_unconditional_jump,
    is_exiting,
    _prev_inst_offset,
)


@dataclass()
class FlowInfo:
    """The FlowInfo class is responsible for converting bytecode into a
    ByteFlow object.

    Attributes
    ----------
    block_offsets: Set[int]
        A set that marks the starting offsets of basic blocks in the bytecode.
    jump_insts: Dict[int, Tuple[int, ...]]
        A dictionary that contains jump instructions and their target offsets.
    last_offset: int
        The offset of the last bytecode instruction.
    """

    block_offsets: Set[int] = field(default_factory=set)

    jump_insts: Dict[int, Tuple[int, ...]] = field(default_factory=dict)

    last_offset: int = field(default=0)

    def _add_jump_inst(self, offset: int, targets: Sequence[int]) -> None:
        """Internal method to add a jump instruction to the FlowInfo.

        This method adds the target offsets of the jump instruction
        to the block_offsets set and updates the jump_insts dictionary.

        Parameters
        ----------
        offset: int
            The given target offset.
        targets: Sequence[int]
            target jump instrcutions.
        """
        for off in targets:
            assert isinstance(off, int)
            self.block_offsets.add(off)
        self.jump_insts[offset] = tuple(targets)

    @staticmethod
    def from_bytecode(bc: dis.Bytecode) -> "FlowInfo":
        """Static method that builds the structured control flow graph (SCFG)
        from the given `dis.Bytecode` object bc.

        This method analyzes the bytecode instructions, marks the start of
        basic blocks, and records jump instructions and their target offsets.
        It builds the structured control flow graph (SCFG) from the given
        `dis.Bytecode` object and returns a FlowInfo object.

        Parameters
        ----------
        bc: dis.Bytecode
            Bytecode from which flowinfo is to be constructed.

        Returns
        -------
        flowinfo: FlowInfo
            FlowInfo object representing the given bytecode.
        """
        flowinfo = FlowInfo()

        for inst in bc:
            # Handle jump-target instruction
            if inst.offset == 0 or inst.is_jump_target:
                flowinfo.block_offsets.add(inst.offset)
            # Handle by op
            if is_conditional_jump(inst.opname):
                flowinfo._add_jump_inst(
                    inst.offset, (_next_inst_offset(inst.offset), inst.argval)
                )
            elif is_unconditional_jump(inst.opname):
                flowinfo._add_jump_inst(inst.offset, (inst.argval,))
            elif is_exiting(inst.opname):
                flowinfo._add_jump_inst(inst.offset, ())

        flowinfo.last_offset = inst.offset
        return flowinfo

    def build_basicblocks(
        self: "FlowInfo", end_offset: Optional[int] = None
    ) -> "SCFG":
        """Builds a graph of basic blocks based on the flow information.

        It creates a structured control flow graph (SCFG) object, assigns
        names to the blocks, and defines the block boundaries, jump targets,
        and backedges. It returns an SCFG object representing the control
        flow graph.

        Parameters
        ----------
        end_offset: int
            The byte offset of the last instruction.

        Returns
        -------
        scfg: SCFG
            SCFG object corresponding to the bytecode contained within the
            current FlowInfo object.
        """
        scfg = SCFG()
        offsets = sorted(self.block_offsets)
        # enumerate names
        names = {
            offset: scfg.name_gen.new_block_name(block_names.PYTHON_BYTECODE)
            for offset in offsets
        }
        if end_offset is None:
            end_offset = _next_inst_offset(self.last_offset)

        for begin, end in zip(offsets, [*offsets[1:], end_offset]):
            name = names[begin]
            targets: Tuple[str, ...]
            term_offset = _prev_inst_offset(end)
            if term_offset not in self.jump_insts:
                # implicit jump
                targets = (names[end],)
            else:
                targets = tuple(names[o] for o in self.jump_insts[term_offset])
            block = PythonBytecodeBlock(
                name=name,
                begin=begin,
                end=end,
                _jump_targets=targets,
                backedges=(),
            )
            scfg.add_block(block)
        return scfg
