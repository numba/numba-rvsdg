from unittest import TestCase
from textwrap import dedent
from numba_rvsdg.core.datastructures.scfg import SCFG, NameGenerator

from numba_rvsdg.tests.test_utils import SCFGComparator
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    PythonBytecodeBlock,
)
from numba_rvsdg.core.datastructures import block_names
from dis import Bytecode, Instruction, Positions

import unittest


class TestSCFGConversion(SCFGComparator):
    def test_yaml_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [
            """
            blocks:
                '0':
                    type: basic
                '1':
                    type: basic
                '2':
                    type: basic
                '3':
                    type: basic
                '4':
                    type: basic
            edges:
                '0': ['1', '2']
                '1': ['3']
                '2': ['4']
                '3': ['4']
                '4': []
            backedges:
            """,
            # Case # 2: Cyclic graph, no back edges
            """
            blocks:
                '0':
                    type: basic
                '1':
                    type: basic
                '2':
                    type: basic
                '3':
                    type: basic
                '4':
                    type: basic
                '5':
                    type: basic
            edges:
                '0': ['1', '2']
                '1': ['5']
                '2': ['1', '5']
                '3': ['1']
                '4': []
                '5': ['3', '4']
            backedges:
            """,
            # Case # 3: Graph with backedges
            """
            blocks:
                '0':
                    type: basic
                '1':
                    type: basic
                '2':
                    type: basic
                '3':
                    type: basic
                '4':
                    type: basic
            edges:
                '0': ['1']
                '1': ['2', '3']
                '2': ['4']
                '3': []
                '4': ['2', '3']
            backedges:
                '4': ['2']
            """,
        ]

        for case in cases:
            case = dedent(case)
            scfg, block_dict = SCFG.from_yaml(case)
            self.assertYAMLEqual(case, scfg.to_yaml(), {"0": block_dict["0"]})

    def test_dict_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [
            {
                "blocks": {
                    "0": {"type": "basic"},
                    "1": {"type": "basic"},
                    "2": {"type": "basic"},
                    "3": {"type": "basic"},
                    "4": {"type": "basic"},
                },
                "edges": {
                    "0": ["1", "2"],
                    "1": ["3"],
                    "2": ["4"],
                    "3": ["4"],
                    "4": [],
                },
                "backedges": {},
            },
            # Case # 2: Cyclic graph, no back edges
            {
                "blocks": {
                    "0": {"type": "basic"},
                    "1": {"type": "basic"},
                    "2": {"type": "basic"},
                    "3": {"type": "basic"},
                    "4": {"type": "basic"},
                    "5": {"type": "basic"},
                },
                "edges": {
                    "0": ["1", "2"],
                    "1": ["5"],
                    "2": ["1", "5"],
                    "3": ["1"],
                    "4": [],
                    "5": ["3", "4"],
                },
                "backedges": {},
            },
            # Case # 3: Graph with backedges
            {
                "blocks": {
                    "0": {"type": "basic"},
                    "1": {"type": "basic"},
                    "2": {"type": "basic"},
                    "3": {"type": "basic"},
                    "4": {"type": "basic"},
                },
                "edges": {
                    "0": ["1"],
                    "1": ["2", "3"],
                    "2": ["4"],
                    "3": [],
                    "4": ["2", "3"],
                },
                "backedges": {"4": ["2"]},
            },
        ]

        for case in cases:
            scfg, block_dict = SCFG.from_dict(case)
            self.assertDictEqual(case, scfg.to_dict(), {"0": block_dict["0"]})


class TestSCFGIterator(SCFGComparator):
    def test_scfg_iter(self):
        name_gen = NameGenerator()
        block_0 = name_gen.new_block_name(block_names.BASIC)
        block_1 = name_gen.new_block_name(block_names.BASIC)
        expected = [
            (block_0, BasicBlock(name=block_0, _jump_targets=(block_1,))),
            (block_1, BasicBlock(name=block_1)),
        ]
        scfg, _ = SCFG.from_yaml(
            """
        blocks:
            'basic_block_0':
                type: basic
            'basic_block_1':
                type: basic
        edges:
            'basic_block_0': ['basic_block_1']
            'basic_block_1': []
        backedges:
        """
        )
        received = list(scfg)
        self.assertEqual(expected, received)


class TestConcealedRegionView(TestCase):
    def setUp(self):
        def foo(n):
            c = 0
            for i in range(n):
                c += i
            return c

        self.foo = foo

    def test_concealed_region_view_iter(self):
        scfg = SCFG.from_bytecode(self.foo)
        scfg._restructure_loop()
        expected = [
            ("python_bytecode_block_0", PythonBytecodeBlock),
            ("loop_region_0", RegionBlock),
            ("python_bytecode_block_3", PythonBytecodeBlock),
        ]
        received = list(
            ((k, type(v)) for k, v in scfg.concealed_region_view.items())
        )
        self.assertEqual(expected, received)


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


if __name__ == "__main__":
    unittest.main()
