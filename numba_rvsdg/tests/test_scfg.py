
from unittest import main
from textwrap import dedent
from numba_rvsdg.core.datastructures.scfg import SCFG

from numba_rvsdg.tests.test_utils import MapComparator
from numba_rvsdg.core.datastructures.basic_block import BasicBlock
from numba_rvsdg.core.datastructures.labels import Label, NameGenerator


class TestBlockMapConversion(MapComparator):

    def test_yaml_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = ["""
            "0":
                type: "basic"
                out: ["1", "2"]
            "1":
                type: "basic"
                out: ["3"]
            "2":
                type: "basic"
                out: ["4"]
            "3":
                type: "basic"
                out: ["4"]
            "4":
                type: "basic"
                out: []""",
        # Case # 2: Cyclic graph, no back edges
            """
            "0":
                type: "basic"
                out: ["1", "2"]
            "1":
                type: "basic"
                out: ["5"]
            "2":
                type: "basic"
                out: ["1", "5"]
            "3":
                type: "basic"
                out: ["0"]
            "4":
                type: "basic"
                out: []
            "5":
                type: "basic"
                out: ["3", "4"]""",
        # Case # 3: Graph with backedges
            """
            "0":
                type: "basic"
                out: ["1"]
            "1":
                type: "basic"
                out: ["2", "3"]
            "2":
                type: "basic"
                out: ["4"]
            "3":
                type: "basic"
                out: []
            "4":
                type: "basic"
                out: ["2", "3"]
                back: ["2"]"""]

        for case in cases:
            case = dedent(case)
            block_map, ref_dict = SCFG.from_yaml(case)
            # TODO: use ref_dict for comparision
            self.assertEqual(case, block_map.to_yaml())

    def test_dict_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [{
            "0":
                {"type": "basic",
                "out": ["1", "2"]},
            "1":
                {"type": "basic",
                "out": ["3"]},
            "2":
                {"type": "basic",
                "out": ["4"]},
            "3":
                {"type": "basic",
                "out": ["4"]},
            "4":
                {"type": "basic",
                "out": []}},
                # Case # 2: Cyclic graph, no back edges
                {
            "0":
                {"type": "basic",
                "out": ["1", "2"]},
            "1":
                {"type": "basic",
                "out": ["5"]},
            "2":
                {"type": "basic",
                "out": ["1", "5"]},
            "3":
                {"type": "basic",
                "out": ["0"]},
            "4":
                {"type": "basic",
                "out": []},
            "5":
                {"type": "basic",
                "out": ["3", "4"]}},
                # Case # 3: Graph with backedges
                {
            "0":
                {"type": "basic",
                "out": ["1"]},
            "1":
                {"type": "basic",
                "out": ["2", "3"]},
            "2":
                {"type": "basic",
                "out": ["4"]},
            "3":
                {"type": "basic",
                "out": []},
            "4":
                {"type": "basic",
                "out": ["2", "3"],
                "back": ["2"]}
        }]

        for case in cases:
            block_map = SCFG.from_dict(case)
            self.assertEqual(case, block_map.to_dict())


class TestSCFGIterator(MapComparator):

    def test_scfg_iter(self):
        name_generator = NameGenerator()
        block_0 = BasicBlock(name_generator, Label())
        block_1 = BasicBlock(name_generator, Label())
        expected = [
            (block_0.block_name, block_0),
            (block_1.block_name, block_1),
        ]
        scfg = SCFG.from_yaml("""
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
        """)
        received = list(scfg)
        self.assertEqual(expected, received)

if __name__ == "__main__":
    main()
           
