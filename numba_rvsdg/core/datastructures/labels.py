from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Label:
    index: int
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


@dataclass(frozen=True, order=True)
class SyntheticForIter(ControlLabel):
    pass


class ControlLabelGenerator:
    def __init__(self, index=0, variable=97):
        self.index = index
        self.variable = variable

    def new_index(self):
        ret = self.index
        self.index += 1
        return ret

    def new_variable(self):
        ret = chr(self.variable)
        self.variable += 1
        return ret
