import dis
from collections import ChainMap
from typing import Tuple, Dict, List
from dataclasses import dataclass, field, replace

from numba_rvsdg.core.utils import _next_inst_offset


@dataclass(frozen=True)
class BasicBlock:
    name: str
    """The corresponding name for this block.  """

    _jump_targets: Tuple[str] = tuple()
    """Jump targets (branch destinations) for this block"""

    backedges: Tuple[str] = tuple()
    """Backedges for this block."""

    @property
    def is_exiting(self) -> bool:
        return not self.jump_targets

    @property
    def fallthrough(self) -> bool:
        return len(self._jump_targets) == 1

    @property
    def jump_targets(self) -> Tuple[str]:
        acc = []
        for j in self._jump_targets:
            if j not in self.backedges:
                acc.append(j)
        return tuple(acc)

    def declare_backedge(self, target: str) -> "BasicBlock":
        """Declare one of the jump targets as a backedge of this block.
        """
        if target in self.jump_targets:
            assert not self.backedges
            return replace(self, backedges=(target,))
        return self

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        """Replaces jump targets of this block by the given tuple.
        The provided jump targets must be in the same order as their
        intended original replacements.
        
        Note that replacing jump targets will not replace the backedge
        tuple, so replacement for any jump targets that is declared as
        a backedge needs to be updated separately using replace_backedges"""
        return replace(self, _jump_targets=jump_targets)

    def replace_backedges(self, backedges: Tuple) -> "BasicBlock":
        """Replace back edges of this block by the given tuple.
        The provided back edges must be in the same order as their
        intended original replacements."""
        return replace(self, backedges=backedges)


@dataclass(frozen=True)
class PythonBytecodeBlock(BasicBlock):
    begin: int = None
    """The starting bytecode offset.
    """

    end: int = None
    """The bytecode offset immediate after the last bytecode of the block.
    """

    def get_instructions(
        self, bcmap: Dict[int, dis.Instruction]
    ) -> List[dis.Instruction]:
        begin = self.begin
        end = self.end
        it = begin
        out = []
        while it < end:
            # Python 3.11 hack: account for gaps in the bytecode sequence
            try:
                out.append(bcmap[it])
            except KeyError:
                pass
            finally:
                it = _next_inst_offset(it)

        return out


@dataclass(frozen=True)
class SyntheticBlock(BasicBlock):
    pass

@dataclass(frozen=True)
class SyntheticExit(SyntheticBlock):
    pass

@dataclass(frozen=True)
class SyntheticReturn(SyntheticBlock):
    pass

@dataclass(frozen=True)
class SyntheticTail(SyntheticBlock):
    pass

@dataclass(frozen=True)
class SyntheticFill(SyntheticBlock):
    pass

@dataclass(frozen=True)
class SyntheticAssignment(SyntheticBlock):
    variable_assignment: dict = None


@dataclass(frozen=True)
class SyntheticBranch(SyntheticBlock):
    variable: str = None
    branch_value_table: dict = None

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        """Replaces jump targets of this block by the given tuple.
        The provided jump targets must be in the same order as their
        intended original replacements. Additionally also updates the
        branch value table of the branch block.
        
        Note that replacing jump targets will not replace the backedge
        tuple, so replacement for any jump targets that is declared as
        a backedge needs to be updated separately using replace_backedges"""

        old_branch_value_table = self.branch_value_table
        new_branch_value_table = {}
        for target in self._jump_targets:
            if target not in jump_targets:
                # ASSUMPTION: only one jump_target is being updated
                diff = set(jump_targets).difference(self._jump_targets)
                assert len(diff) == 1
                new_target = next(iter(diff))
                for k, v in old_branch_value_table.items():
                    if v == target:
                        new_branch_value_table[k] = new_target
            else:
                # copy all old values
                for k, v in old_branch_value_table.items():
                    if v == target:
                        new_branch_value_table[k] = v

        return replace(
            self,
            _jump_targets=jump_targets,
            branch_value_table=new_branch_value_table,
        )


@dataclass(frozen=True)
class SyntheticHead(SyntheticBranch):
    pass


@dataclass(frozen=True)
class SyntheticExitingLatch(SyntheticBranch):
    pass


@dataclass(frozen=True)
class SyntheticExitBranch(SyntheticBranch):
    pass


@dataclass(frozen=True)
class RegionBlock(BasicBlock):
    kind: str = None
    """The kind of region. Can be 'head', 'tail', 'branch', 
    'loop' or 'meta' strings"""
    parent_region: "RegionBlock" = None
    """The parent region of this region as per the SCFG."""
    header: str = None
    """The header node of the region"""
    subregion: "SCFG" = None
    """The subgraph as an independent SCFG. Note that in case
    of subregions the exiting node may point to blocks outside
    of the current SCFG object."""
    exiting: str = None
    """The exiting node of the region."""

    def replace_header(self, new_header):
        """Does inplace replacement of the header block."""
        object.__setattr__(self, "header", new_header)

    def replace_exiting(self, new_exiting):
        """Does, inplace replacement of the exiting block."""
        object.__setattr__(self, "exiting", new_exiting)
