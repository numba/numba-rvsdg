
from unittest import main, TestCase
from textwrap import dedent
from numba_rvsdg.core.datastructures.scfg import SCFG

from numba_rvsdg.tests.test_utils import SCFGComparator
from numba_rvsdg.core.datastructures.basic_block import (BasicBlock,
                                                         RegionBlock,
                                                         PythonBytecodeBlock,
                                                         )
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.labels import (ControlLabel,
                                                    PythonBytecodeLabel,
                                                    )


class TestSCFGConversion(SCFGComparator):

    def test_yaml_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = ["""
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
                be: ["2"]"""]

        for case in cases:
            case = dedent(case)
            scfg = SCFG.from_yaml(case)
            self.assertEqual(case, scfg.to_yaml())
    
    def test_dict_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [{
    "0":
        {"jt": ["1", "2"]},
    "1":
        {"jt": ["3"]},
    "2":
        {"jt": ["4"]},
    "3":
        {"jt": ["4"]},
    "4":
        {"jt": []}},
        # Case # 2: Cyclic graph, no back edges
        {
    "0":
        {"jt": ["1", "2"]},
    "1":
        {"jt": ["5"]},
    "2":
        {"jt": ["1", "5"]},
    "3":
        {"jt": ["0"]},
    "4":
        {"jt": []},
    "5":
        {"jt": ["3", "4"]}},
        # Case # 3: Graph with backedges
        {
    "0":
        {"jt": ["1"]},
    "1":
        {"jt": ["2", "3"]},
    "2":
        {"jt": ["4"]},
    "3":
        {"jt": []},
    "4":
        {"jt": ["2", "3"],
        "be": ["2"]}}]

        for case in cases:
            scfg = SCFG.from_dict(case)
            self.assertEqual(case, scfg.to_dict())


class TestSCFGIterator(SCFGComparator):

    def test_scfg_iter(self):
        expected = [
            (ControlLabel("0"), BasicBlock(label=ControlLabel("0"),
                                           _jump_targets=(ControlLabel("1"),))),
            (ControlLabel("1"), BasicBlock(label=ControlLabel("1"))),
        ]
        scfg = SCFG.from_yaml("""
        "0":
            jt: ["1"]
        "1":
            jt: []
        """)
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
        expected = [(PythonBytecodeLabel(index='0'), PythonBytecodeBlock),
                    (PythonBytecodeLabel(index='1'), RegionBlock),
                    (PythonBytecodeLabel(index='3'), PythonBytecodeBlock)]
        received = list(((k, type(v)) for k,v in restructured.scfg.concealed_region_view.items()))
        self.assertEqual(expected, received)


if __name__ == "__main__":
    main()
