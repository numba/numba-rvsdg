import dis
from collections import ChainMap
from typing import Tuple, Dict, List
from dataclasses import dataclass, field, replace

from numba_rvsdg.core.datastructures.labels import Label
from numba_rvsdg.core.utils import _next_inst_offset


@dataclass(frozen=True)
class BasicBlock:
    label: Label
    """The corresponding Label for this block.  """

    _jump_targets: Tuple[Label]
    """Jump targets (branch destinations) for this block"""

    backedges: Tuple[Label]
    """Backedges for this block."""

    @property
    def is_exiting(self) -> bool:
        return not self.jump_targets

    @property
    def fallthrough(self) -> bool:
        return len(self._jump_targets) == 1

    @property
    def jump_targets(self) -> Tuple[Label]:
        acc = []
        for j in self._jump_targets:
            if j not in self.backedges:
                acc.append(j)
        return tuple(acc)

    def replace_backedge(self, target: Label) -> "BasicBlock":
        if target in self.jump_targets:
            assert not self.backedges
            return replace(self, backedges=(target,))
        return self

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        return replace(self, _jump_targets=jump_targets)


@dataclass(frozen=True)
class PythonBytecodeBlock(BasicBlock):
    begin: int
    """The starting bytecode offset.
    """

    end: int
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
class ControlVariableBlock(BasicBlock):
    variable_assignment: dict


@dataclass(frozen=True)
class BranchBlock(BasicBlock):
    variable: str
    branch_value_table: dict

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        fallthrough = len(jump_targets) == 1
        old_branch_value_table = self.branch_value_table
        new_branch_value_table = {}
        for target in self.jump_targets:
            if target not in jump_targets:
                # ASSUMPTION: only one jump_target is being updated
                diff = set(jump_targets).difference(self.jump_targets)
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
            self, _jump_targets=jump_targets, branch_value_table=new_branch_value_table,
        )


@dataclass(frozen=True)
class RegionBlock(BasicBlock):
    kind: str
    headers: Dict[Label, BasicBlock]
    """The header of the region"""
    subregion: "BlockMap"
    """The subgraph excluding the headers
    """
    exit: Label
    """The exit node.
    """

    def get_full_graph(self):
        graph = ChainMap(self.subregion.graph, self.headers)
        return graph
