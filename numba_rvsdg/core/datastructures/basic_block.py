import dis
from collections import ChainMap
from typing import Tuple, Dict, List, Set
from dataclasses import dataclass, field, replace

from numba_rvsdg.core.datastructures.labels import Label
from numba_rvsdg.core.utils import _next_inst_offset


@dataclass(frozen=True)
class BasicBlock:
    label: Label
    """The corresponding Label for this block.  """

    _jump_targets: Tuple[Label] = tuple()
    """Jump targets (branch destinations) for this block"""

    backedges: Tuple[Label] = tuple()
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
class ControlVariableBlock(BasicBlock):
    variable_assignment: dict = None


@dataclass(frozen=True)
class BranchBlock(BasicBlock):
    variable: str = None
    branch_value_table: dict = None

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
            self,
            _jump_targets=jump_targets,
            branch_value_table=new_branch_value_table,
        )


@dataclass(frozen=True)
class RegionBlock(BasicBlock):
    kind: str = None
    headers: Set[Label] = None
    """The header of the region"""
    subregion: "BlockMap" = None
    """The subgraph excluding the headers
    """
    exit: Label = None
    """The exit node.
    """
    def __post_init__(self):
        assert isinstance(self.subregion.graph, dict)
        assert isinstance(self.headers, set)

    def get_full_graph(self):
        # assert not (self.headers - set(self.subregion.graph))
        graph = self.subregion.graph.copy()
        return graph
