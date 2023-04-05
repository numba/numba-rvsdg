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
class RegionLabel(Label):
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


@dataclass(frozen=True, order=True)
class LoopRegionLabel(RegionLabel):
    pass


@dataclass(frozen=True, order=True)
class MetaRegionLabel(RegionLabel):
    pass


# Maybe we can register new labels over here instead of static lists
label_types = {
    "label": Label,
    "python_bytecode": PythonBytecodeLabel,
    "control": ControlLabel,
    "synth_branch": SyntheticBranch,
    "synth_tail": SyntheticTail,
    "synth_exit": SyntheticExit,
    "synth_head": SyntheticHead,
    "synth_return": SyntheticReturn,
    "synth_latch": SyntheticLatch,
    "synth_exit_latch": SyntheticExitingLatch,
    "synth_assign": SynthenticAssignment,
}


def get_label_class(label_type_string):
    if label_type_string in label_types:
        return label_types[label_type_string]
    else:
        raise TypeError(f"Block Type {label_type_string} not recognized.")


@dataclass(frozen=True, order=True)
class Name:
    name: str
    
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass(frozen=True, order=True)
class BlockName(Name):
    pass


@dataclass(frozen=True, order=True)
class RegionName(Name):
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

    def new_block_name(self, label: str) -> BlockName:
        ret = self.block_index
        self.block_index += 1
        return BlockName(str(label).lower().split("(")[0] + "_" + str(ret))

    def new_region_name(self, label: str) -> RegionName:
        ret = self.region_index
        self.region_index += 1
        return RegionName(str(label).lower().split("(")[0] + "_" + str(ret))

    def new_var_name(self) -> str:
        variable_name = chr(self.variable_index)
        self.variable_index += 1
        return str(variable_name)
