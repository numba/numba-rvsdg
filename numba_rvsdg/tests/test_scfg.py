from unittest import main, TestCase
from textwrap import dedent
from numba_rvsdg.core.datastructures.scfg import SCFG, NameGenerator

from numba_rvsdg.tests.test_utils import SCFGComparator
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    PythonBytecodeBlock,
)
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures import block_names


class TestSCFGConversion(SCFGComparator):
    def test_yaml_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [
            """
            "0":
                jt: ["1", "2"]
            "1":
                jt: ["3"]
            "2":
                jt: ["4"]
            "3":
                jt: ["4"]
            "4":
                jt: []""",
            # Case # 2: Cyclic graph, no back edges
            """
            "0":
                jt: ["1", "2"]
            "1":
                jt: ["5"]
            "2":
                jt: ["1", "5"]
            "3":
                jt: ["0"]
            "4":
                jt: []
            "5":
                jt: ["3", "4"]""",
            # Case # 3: Graph with backedges
            """
            "0":
                jt: ["1"]
            "1":
                jt: ["2", "3"]
            "2":
                jt: ["4"]
            "3":
                jt: []
            "4":
                jt: ["2", "3"]
                be: ["2"]""",
        ]

        for case in cases:
            case = dedent(case)
            scfg, block_dict = SCFG.from_yaml(case)
            self.assertYAMLEqual(case, scfg.to_yaml(), {"0": block_dict["0"]})

    def test_dict_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [
            {
                "0": {"jt": ["1", "2"]},
                "1": {"jt": ["3"]},
                "2": {"jt": ["4"]},
                "3": {"jt": ["4"]},
                "4": {"jt": []},
            },
            # Case # 2: Cyclic graph, no back edges
            {
                "0": {"jt": ["1", "2"]},
                "1": {"jt": ["5"]},
                "2": {"jt": ["1", "5"]},
                "3": {"jt": ["0"]},
                "4": {"jt": []},
                "5": {"jt": ["3", "4"]},
            },
            # Case # 3: Graph with backedges
            {
                "0": {"jt": ["1"]},
                "1": {"jt": ["2", "3"]},
                "2": {"jt": ["4"]},
                "3": {"jt": []},
                "4": {"jt": ["2", "3"], "be": ["2"]},
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
        "0":
            jt: ["1"]
        "1":
            jt: []
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
        restructured = flow._restructure_loop()
        expected = [
            ("python_bytecode_block_0", PythonBytecodeBlock),
            ("loop_region_0", RegionBlock),
            ("python_bytecode_block_3", PythonBytecodeBlock),
        ]
        received = list(
            (
                (k, type(v))
                for k, v in restructured.scfg.concealed_region_view.items()
            )
        )
        self.assertEqual(expected, received)


if __name__ == "__main__":
    main()
