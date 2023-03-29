from unittest import main
from textwrap import dedent
from numba_rvsdg.core.datastructures.scfg import SCFG

from numba_rvsdg.tests.test_utils import SCFGComparator
from numba_rvsdg.core.datastructures.basic_block import BasicBlock
from numba_rvsdg.core.datastructures.labels import Label, NameGenerator


class TestSCFGConversion(SCFGComparator):
    def test_yaml_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [
            """
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
                back: ["2"]""",
        ]

        for case in cases:
            case = dedent(case)
            scfg, ref_dict = SCFG.from_yaml(case)
            yaml = scfg.to_yaml()
            self.assertYAMLEquals(case, yaml, ref_dict)

    def test_dict_conversion(self):
        # Case # 1: Acyclic graph, no back-edges
        cases = [
            {
                "0": {"type": "basic", "out": ["1", "2"]},
                "1": {"type": "basic", "out": ["3"]},
                "2": {"type": "basic", "out": ["4"]},
                "3": {"type": "basic", "out": ["4"]},
                "4": {"type": "basic", "out": []},
            },
            # Case # 2: Cyclic graph, no back edges
            {
                "0": {"type": "basic", "out": ["1", "2"]},
                "1": {"type": "basic", "out": ["5"]},
                "2": {"type": "basic", "out": ["1", "5"]},
                "3": {"type": "basic", "out": ["0"]},
                "4": {"type": "basic", "out": []},
                "5": {"type": "basic", "out": ["3", "4"]},
            },
            # Case # 3: Graph with backedges
            {
                "0": {"type": "basic", "out": ["1"]},
                "1": {"type": "basic", "out": ["2", "3"]},
                "2": {"type": "basic", "out": ["4"]},
                "3": {"type": "basic", "out": []},
                "4": {"type": "basic", "out": ["2", "3"], "back": ["2"]},
            },
        ]

        for case in cases:
            scfg = SCFG.from_dict(case)
            scfg, ref_dict = SCFG.from_dict(case)
            generated_dict = scfg.to_dict()
            self.assertDictEquals(case, generated_dict, ref_dict)


class TestSCFGIterator(SCFGComparator):
    def test_scfg_iter(self):
        name_generator = NameGenerator()
        block_0 = BasicBlock(name_generator, Label())
        block_1 = BasicBlock(name_generator, Label())
        expected = [
            (block_0.block_name, block_0),
            (block_1.block_name, block_1),
        ]
        scfg, ref_dict = SCFG.from_yaml(
            """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
        """
        )
        received = list(scfg)
        self.assertEqual(expected, received)


class TestInsertBlock(SCFGComparator):
    def test_linear(self):
        original = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected = """
        "0":
            type: "basic"
            out: ["2"]
        "1":
            type: "basic"
            out: []
        "2":
            type: "basic"
            out: ["1"]
        """
        expected_scfg, _ = SCFG.from_yaml(expected)

        preds = list((block_ref_orig["0"],))
        succs = list((block_ref_orig["1"],))
        new_block = original_scfg.add_block()
        original_scfg.insert_block_between(new_block, preds, succs)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor(self):
        original = """
        "0":
            type: "basic"
            out: ["2"]
        "1":
            type: "basic"
            out: ["2"]
        "2":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected = """
        "0":
            type: "basic"
            out: ["3"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: []
        "3":
            type: "basic"
            out: ["2"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        preds = list((block_ref_orig["0"], block_ref_orig["1"]))
        succs = list((block_ref_orig["2"],))
        new_block = original_scfg.add_block()
        original_scfg.insert_block_between(new_block, preds, succs)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_successor(self):
        original = """
        "0":
            type: "basic"
            out: ["1", "2"]
        "1":
            type: "basic"
            out: []
        "2":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected = """
        "0":
            type: "basic"
            out: ["3"]
        "1":
            type: "basic"
            out: []
        "2":
            type: "basic"
            out: []
        "3":
            type: "basic"
            out: ["1", "2"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        preds = list((block_ref_orig["0"],))
        succs = list((block_ref_orig["1"], block_ref_orig["2"]))
        new_block = original_scfg.add_block()
        original_scfg.insert_block_between(new_block, preds, succs)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor_and_dual_successor(self):
        original = """
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
            out: []
        "4":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected = """
        "0":
            type: "basic"
            out: ["1", "2"]
        "1":
            type: "basic"
            out: ["5"]
        "2":
            type: "basic"
            out: ["5"]
        "3":
            type: "basic"
            out: []
        "4":
            type: "basic"
            out: []
        "5":
            type: "basic"
            out: ["3", "4"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        preds = list((block_ref_orig["1"], block_ref_orig["2"]))
        succs = list((block_ref_orig["3"], block_ref_orig["4"]))
        new_block = original_scfg.add_block()
        original_scfg.insert_block_between(new_block, preds, succs)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor_and_dual_successor_with_additional_arcs(self):
        original = """
        "0":
            type: "basic"
            out: ["1", "2"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["1", "4"]
        "3":
            type: "basic"
            out: ["0"]
        "4":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected = """
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
            out: ["3", "4"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        preds = list((block_ref_orig["1"], block_ref_orig["2"]))
        succs = list((block_ref_orig["3"], block_ref_orig["4"]))
        new_block = original_scfg.add_block()
        original_scfg.insert_block_between(new_block, preds, succs)

        self.assertSCFGEqual(expected_scfg, original_scfg)


if __name__ == "__main__":
    main()
