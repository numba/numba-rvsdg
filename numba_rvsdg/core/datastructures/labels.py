from dataclasses import dataclass
from typing import List


@dataclass(frozen=True, order=True)
class Label:
    info: List[str] = None
    """Any Block specific information we want to add can go here"""
    ...


@dataclass(frozen=True, order=True)
class PythonBytecodeLabel(Label):
    pass


@dataclass(frozen=True, order=True)
class ControlLabel(Label):
    pass


@dataclass(frozen=True, order=True)
class SyntheticBranch(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticTail(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticExit(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticHead(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticReturn(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticLatch(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticExitingLatch(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SynthenticAssignment(ControlLabel):
    pass


@dataclass
class NameGenerator:
    """Name generator for various element names.

    Attributes
    ----------

    block_index : int
        The starting index for blocks
    variable_index: int
        The starting index for control variables
    region_index : int
        The starting index for regions
    """
    block_index: int = 0
    variable_index: int = 97  # Variables start at lowercase 'a'
    region_index: int = 0

    def new_block_name(self, label: str) -> str:
        ret = self.block_index
        self.block_index += 1
        return str(label).lower().split("(")[0] + "_" + str(ret)

    def new_region_name(self, label: str) -> str:
        ret = self.region_index
        self.region_index += 1
        return str(label).lower().split("(")[0] + "_" + str(ret)

    def new_var_name(self) -> str:
        variable_name = chr(self.variable_index)
        self.variable_index += 1
        return str(variable_name)
