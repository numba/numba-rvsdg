# mypy: ignore-errors
from dis import Bytecode, Instruction, Positions
import unittest
from numba_scfg.core.datastructures.basic_block import PythonBytecodeBlock
from numba_scfg.core.datastructures.byte_flow import ByteFlow
from numba_scfg.core.datastructures.scfg import SCFG, NameGenerator
from numba_scfg.core.datastructures.flow_info import FlowInfo
from numba_scfg.core.datastructures import block_names


def fun():
    x = 1
    return x


bytecode = Bytecode(fun)
# If the function definition line changes, just change the variable below,
# rest of it will adjust as long as function remains the same
func_def_line = 11


class TestBCMapFromBytecode(unittest.TestCase):
    def test(self):
        expected = {
            0: Instruction(
                opname="RESUME",
                opcode=151,
                arg=0,
                argval=0,
                argrepr="",
                offset=0,
                starts_line=func_def_line,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line,
                    end_lineno=func_def_line,
                    col_offset=0,
                    end_col_offset=0,
                ),
            ),
            2: Instruction(
                opname="LOAD_CONST",
                opcode=100,
                arg=1,
                argval=1,
                argrepr="1",
                offset=2,
                starts_line=func_def_line + 1,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line + 1,
                    end_lineno=func_def_line + 1,
                    col_offset=8,
                    end_col_offset=9,
                ),
            ),
            4: Instruction(
                opname="STORE_FAST",
                opcode=125,
                arg=0,
                argval="x",
                argrepr="x",
                offset=4,
                starts_line=None,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line + 1,
                    end_lineno=func_def_line + 1,
                    col_offset=4,
                    end_col_offset=5,
                ),
            ),
            6: Instruction(
                opname="LOAD_FAST",
                opcode=124,
                arg=0,
                argval="x",
                argrepr="x",
                offset=6,
                starts_line=func_def_line + 2,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line + 2,
                    end_lineno=func_def_line + 2,
                    col_offset=11,
                    end_col_offset=12,
                ),
            ),
            8: Instruction(
                opname="RETURN_VALUE",
                opcode=83,
                arg=None,
                argval=None,
                argrepr="",
                offset=8,
                starts_line=None,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line + 2,
                    end_lineno=func_def_line + 2,
                    col_offset=4,
                    end_col_offset=12,
                ),
            ),
        }
        received = SCFG.bcmap_from_bytecode(bytecode)
        self.assertEqual(expected, received)


class TestPythonBytecodeBlock(unittest.TestCase):
    def test_constructor(self):
        name_gen = NameGenerator()
        block = PythonBytecodeBlock(
            name=name_gen.new_block_name(block_names.PYTHON_BYTECODE),
            begin=0,
            end=8,
            _jump_targets=(),
            backedges=(),
        )
        self.assertEqual(block.name, "python_bytecode_block_0")
        self.assertEqual(block.begin, 0)
        self.assertEqual(block.end, 8)
        self.assertFalse(block.fallthrough)
        self.assertTrue(block.is_exiting)
        self.assertEqual(block.jump_targets, ())
        self.assertEqual(block.backedges, ())

    def test_is_jump_target(self):
        name_gen = NameGenerator()
        block = PythonBytecodeBlock(
            name=name_gen.new_block_name(block_names.PYTHON_BYTECODE),
            begin=0,
            end=8,
            _jump_targets=(
                name_gen.new_block_name(block_names.PYTHON_BYTECODE),
            ),
            backedges=(),
        )
        self.assertEqual(block.jump_targets, ("python_bytecode_block_1",))
        self.assertFalse(block.is_exiting)

    def test_get_instructions(self):
        name_gen = NameGenerator()
        block = PythonBytecodeBlock(
            name=name_gen.new_block_name(block_names.PYTHON_BYTECODE),
            begin=0,
            end=8,
            _jump_targets=(),
            backedges=(),
        )
        expected = [
            Instruction(
                opname="RESUME",
                opcode=151,
                arg=0,
                argval=0,
                argrepr="",
                offset=0,
                starts_line=func_def_line,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line,
                    end_lineno=func_def_line,
                    col_offset=0,
                    end_col_offset=0,
                ),
            ),
            Instruction(
                opname="LOAD_CONST",
                opcode=100,
                arg=1,
                argval=1,
                argrepr="1",
                offset=2,
                starts_line=func_def_line + 1,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line + 1,
                    end_lineno=func_def_line + 1,
                    col_offset=8,
                    end_col_offset=9,
                ),
            ),
            Instruction(
                opname="STORE_FAST",
                opcode=125,
                arg=0,
                argval="x",
                argrepr="x",
                offset=4,
                starts_line=None,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line + 1,
                    end_lineno=func_def_line + 1,
                    col_offset=4,
                    end_col_offset=5,
                ),
            ),
            Instruction(
                opname="LOAD_FAST",
                opcode=124,
                arg=0,
                argval="x",
                argrepr="x",
                offset=6,
                starts_line=func_def_line + 2,
                is_jump_target=False,
                positions=Positions(
                    lineno=func_def_line + 2,
                    end_lineno=func_def_line + 2,
                    col_offset=11,
                    end_col_offset=12,
                ),
            ),
        ]

        received = block.get_instructions(SCFG.bcmap_from_bytecode(bytecode))
        self.assertEqual(expected, received)


class TestFlowInfo(unittest.TestCase):
    def test_constructor(self):
        flowinfo = FlowInfo()
        self.assertEqual(len(flowinfo.block_offsets), 0)
        self.assertEqual(len(flowinfo.jump_insts), 0)

    def test_from_bytecode(self):
        expected = FlowInfo(
            block_offsets={0}, jump_insts={8: ()}, last_offset=8
        )

        received = FlowInfo.from_bytecode(bytecode)
        self.assertEqual(expected, received)

    def test_build_basic_blocks(self):
        name_gen = NameGenerator()
        new_name = name_gen.new_block_name(block_names.PYTHON_BYTECODE)
        expected = SCFG(
            graph={
                new_name: PythonBytecodeBlock(
                    name=new_name,
                    begin=0,
                    end=10,
                    _jump_targets=(),
                    backedges=(),
                )
            }
        )
        received = FlowInfo.from_bytecode(bytecode).build_basicblocks()
        self.assertEqual(expected, received)


class TestByteFlow(unittest.TestCase):
    def test_constructor(self):
        byteflow = ByteFlow([], [])
        self.assertEqual(len(byteflow.bc), 0)
        self.assertEqual(len(byteflow.scfg), 0)

    def test_from_bytecode(self):
        name_gen = NameGenerator()
        new_name = name_gen.new_block_name(block_names.PYTHON_BYTECODE)
        scfg = SCFG(
            graph={
                new_name: PythonBytecodeBlock(
                    name=new_name,
                    begin=0,
                    end=10,
                    _jump_targets=(),
                    backedges=(),
                )
            }
        )
        expected = ByteFlow(bc=bytecode, scfg=scfg)
        received = ByteFlow.from_bytecode(fun)
        self.assertEqual(expected.scfg, received.scfg)


if __name__ == "__main__":
    unittest.main()
