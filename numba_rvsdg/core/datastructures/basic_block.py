import dis
from typing import Dict, List
from dataclasses import dataclass, field, InitVar

from numba_rvsdg.core.datastructures.labels import Label, NameGenerator, BlockName
from numba_rvsdg.core.utils import _next_inst_offset


@dataclass(frozen=True)
class BasicBlock:
    name_gen: InitVar[NameGenerator]
    """Block Name Generator associated with this BasicBlock.
       Note: This is an initialization only argument and not
       a class attribute."""

    block_name: BlockName = field(init=False)
    """Unique name identifier for this block"""

    label: Label
    """The corresponding Label for this block."""

    def __post_init__(self, name_gen):
        block_name = name_gen.new_block_name(label=self.label)
        object.__setattr__(self, "block_name", block_name)


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


# Maybe we can register new blocks over here instead of static lists
block_types = {
    "basic": BasicBlock,
    "python_bytecode": PythonBytecodeBlock,
    "control_variable": ControlVariableBlock,
    "branch": BranchBlock,
}


def get_block_class(block_type_string: str):
    if block_type_string in block_types:
        return block_types[block_type_string]
    else:
        raise TypeError(f"Block Type {block_type_string} not recognized.")

def get_block_class_str(basic_block: BasicBlock):
    for key, value in block_types.items():
        if isinstance(basic_block, value):
            return key
    else:
        raise TypeError(f"Block Type of {basic_block} not recognized.")
