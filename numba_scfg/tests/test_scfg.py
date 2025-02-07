# mypy: ignore-errors

from unittest import main, TestCase
from textwrap import dedent
from numba_scfg.core.datastructures.scfg import SCFG, NameGenerator

from numba_scfg.tests.test_utils import SCFGComparator
from numba_scfg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    PythonBytecodeBlock,
)
from numba_scfg.core.datastructures.byte_flow import ByteFlow
from numba_scfg.core.datastructures import block_names


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
                    "3": ["0"],
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
        flow = ByteFlow.from_bytecode(self.foo)
        flow.scfg.restructure_loop()
        expected = [
            ("python_bytecode_block_0", PythonBytecodeBlock),
            ("loop_region_0", RegionBlock),
            ("python_bytecode_block_3", PythonBytecodeBlock),
        ]
        received = list(
            ((k, type(v)) for k, v in flow.scfg.concealed_region_view.items())
        )
        self.assertEqual(expected, received)


if __name__ == "__main__":
    main()
