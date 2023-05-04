import dis
from collections import ChainMap
from typing import Tuple, Dict, List
from dataclasses import dataclass, field, replace

from numba_rvsdg.core.utils import _next_inst_offset


@dataclass(frozen=True)
class BasicBlock:
    name: str
    """The corresponding name for this block.  """


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
    headers: Dict[str, BasicBlock] = None
    """The header of the region"""
    subregion: "SCFG" = None
    """The subgraph excluding the headers
    """
    exiting: str = None
    """The exiting node.
    """

    def get_full_graph(self):
        graph = ChainMap(self.subregion.blocks, self.headers)
        return graph
