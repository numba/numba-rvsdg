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
class BlockName:
    name: str
    ...


@dataclass(frozen=True, order=True)
class RegionName:
    name: str
    ...


@dataclass
class NameGenerator:
    index: int = 0
    var_index: int = 0
    region_index: int = 0

    def new_block_name(self, label):
        ret = self.index
        self.index += 1
        return BlockName(str(label).lower().split("(")[0] + "_" + str(ret))

    def new_region_name(self, kind):
        ret = self.region_index
        self.region_index += 1
        return RegionName(str(kind).lower().split("(")[0] + "_" + str(ret))

    def new_var_name(self):
        var_name = chr(self.var_index)
        self.var_index = self.var_index + 1
        return str(var_name)
