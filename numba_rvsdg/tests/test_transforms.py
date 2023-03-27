from unittest import main

from numba_rvsdg.core.datastructures.labels import (
    ControlLabel,
    SyntheticTail,
    SyntheticExit,
)
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.transformations import (
    loop_restructure_helper,
    join_returns,
    join_tails_and_exits,
)
from numba_rvsdg.tests.test_utils import SCFGComparator


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

        preds = set((block_ref_orig["0"],))
        succs = set((block_ref_orig["1"],))
        original_scfg.insert_block(preds, succs)

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

        preds = set((block_ref_orig["0"], block_ref_orig["1"]))
        succs = set((block_ref_orig["2"],))
        original_scfg.insert_block(preds, succs)

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

        preds = set((block_ref_orig["0"],))
        succs = set((block_ref_orig["1"], block_ref_orig["2"]))
        original_scfg.insert_block(preds, succs)

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

        preds = set((block_ref_orig["1"], block_ref_orig["2"]))
        succs = set((block_ref_orig["3"], block_ref_orig["4"]))
        original_scfg.insert_block(preds, succs)

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

        preds = set((block_ref_orig["1"], block_ref_orig["2"]))
        succs = set((block_ref_orig["3"], block_ref_orig["4"]))
        original_scfg.insert_block(preds, succs)

        self.assertSCFGEqual(expected_scfg, original_scfg)


class TestJoinReturns(SCFGComparator):
    def test_two_returns(self):
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
            out: ["1", "2"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["3"]
        "3":
            type: "basic"
            label_type: "synth_return" 
            out: []
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        join_returns(original_scfg)

        self.assertSCFGEqual(expected_scfg, original_scfg)


class TestJoinTailsAndExits(SCFGComparator):
    def test_join_tails_and_exits_case_00(self):
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
            out: ["1"]
        "1":
            type: "basic"
            out: []
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        tails = set((block_ref_orig["0"],))
        exits = set((block_ref_orig["1"],))
        join_tails_and_exits(original_scfg, tails, exits)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_join_tails_and_exits_case_01(self):
        original = """
        "0":
            type: "basic"
            out: ["1", "2"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["3"]
        "3":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected = """
        "0":
            type: "basic"
            out: ["4"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["3"]
        "3":
            type: "basic"
            out: []
        "4":
            type: "basic"
            label_type: "synth_exit"
            out: ["1", "2"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        tails = set((block_ref_orig["0"],))
        exits = set((block_ref_orig["1"], block_ref_orig["2"]))
        join_tails_and_exits(original_scfg, tails, exits)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_join_tails_and_exits_case_02_01(self):
        original = """
        "0":
            type: "basic"
            out: ["1", "2"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["3"]
        "3":
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
            out: ["4"]
        "2":
            type: "basic"
            out: ["4"]
        "3":
            type: "basic"
            out: []
        "4":
            type: "basic"
            label_type: "synth_tail"
            out: ["3"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        tails = set((block_ref_orig["1"], block_ref_orig["2"]))
        exits = set((block_ref_orig["3"],))
        join_tails_and_exits(original_scfg, tails, exits)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_join_tails_and_exits_case_02_02(self):
        original = """
        "0":
            type: "basic"
            out: ["1", "2"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["1", "3"]
        "3":
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
            out: ["4"]
        "2":
            type: "basic"
            out: ["1", "4"]
        "3":
            type: "basic"
            out: []
        "4":
            type: "basic"
            label_type: "synth_tail"
            out: ["3"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        tails = set((block_ref_orig["1"], block_ref_orig["2"]))
        exits = set((block_ref_orig["3"],))

        join_tails_and_exits(original_scfg, tails, exits)
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_join_tails_and_exits_case_03_01(self):

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
            out: ["5"]
        "4":
            type: "basic"
            out: ["5"]
        "5":
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
            out: ["6"]
        "2":
            type: "basic"
            out: ["6"]
        "3":
            type: "basic"
            out: ["5"]
        "4":
            type: "basic"
            out: ["5"]
        "5":
            type: "basic"
            out: []
        "6":
            type: "basic"
            label_type: "synth_tail"
            out: ["7"]
        "7":
            type: "basic"
            label_type: "synth_exit"
            out: ["3", "4"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        tails = set((block_ref_orig["1"], block_ref_orig["2"]))
        exits = set((block_ref_orig["3"], block_ref_orig["4"]))
        join_tails_and_exits(original_scfg, tails, exits)

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_join_tails_and_exits_case_03_02(self):

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
            out: ["5"]
        "4":
            type: "basic"
            out: ["5"]
        "5":
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
            out: ["6"]
        "2":
            type: "basic"
            out: ["1", "6"]
        "3":
            type: "basic"
            out: ["5"]
        "4":
            type: "basic"
            out: ["5"]
        "5":
            type: "basic"
            out: []
        "6":
            type: "basic"
            label_type: "synth_tail"
            out: ["7"]
        "7":
            type: "basic"
            label_type: "synth_exit"
            out: ["3", "4"]
        """
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        tails = set((block_ref_orig["1"], block_ref_orig["2"]))
        exits = set((block_ref_orig["3"], block_ref_orig["4"]))
        join_tails_and_exits(original_scfg, tails, exits)

        self.assertSCFGEqual(expected_scfg, original_scfg)


class TestLoopRestructure(SCFGComparator):
    def test_no_op_mono(self):
        """Loop consists of a single Block."""
        original = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["1", "2"]
        "2":
            type: "basic"
            out: []
        """
        expected = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["1", "2"]
            be: ["1"]
        "2":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)

        loop_restructure_helper(original_scfg, set(wrap_id({"1"})))

        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_no_op(self):
        """Loop consists of two blocks, but it's in form."""
        original = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["2"]
        "2":
            type: "basic"
            out: ["1", "3"]
        "3":
            type: "basic"
            out: []
        """
        expected = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["2"]
        "2":
            type: "basic"
            out: ["1", "3"]
            be: ["1"]
        "3":
            type: "basic"
            out: []
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_backedge_not_exiting(self):
        """Loop has a backedge not coming from the exiting block.

        This is the situation with the standard Python for loop.
        """
        original = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["2", "3"]
        "2":
            type: "basic"
            out: ["1"]
        "3":
            type: "basic"
            out: []
        """
        expected = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["2", "5"]
        "2":
            type: "basic"
            out: ["6"]
        "3":
            type: "basic"
            out: []
        "4":
            type: "basic"
            out: ["1", "3"]
            be: ["1"]
        "5":
            type: "basic"
            out: ["4"]
        "6":
            type: "basic"
            out: ["4"]
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_exit(self):
        """Loop has two exiting blocks.

        For example a loop with a break.

        """
        original = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["2"]
        "2":
            type: "basic"
            out: ["3", "4"]
        "3":
            type: "basic"
            out: ["1", "4"]
        "4":
            type: "basic"
            out: []
        """
        expected = """
        "0":
            type: "basic"
            out: ["1"]
        "1":
            type: "basic"
            out: ["2"]
        "2":
            type: "basic"
            out: ["3", "6"]
        "3":
            type: "basic"
            out: ["7", "8"]
        "4":
            type: "basic"
            out: []
        "5":
            type: "basic"
            out: ["1", "4"]
            be: ["1"]
        "6":
            type: "basic"
            out: ["5"]
        "7":
            type: "basic"
            out: ["5"]
        "8":
            type: "basic"
            out: ["5"]
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2", "3"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_header(self):
        """This is like the example from Bahman2015 fig. 3 --
        but with one exiting block removed."""
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
            out: ["2", "5"]
        "4":
            type: "basic"
            out: ["1"]
        "5":
            type: "basic"
            out: []
        """
        expected = """
        "0":
            type: "basic"
            out: ["7", "8"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["4"]
        "3":
            type: "basic"
            out: ["10", "11"]
        "4":
            type: "basic"
            out: ["12"]
        "5":
            type: "basic"
            out: []
        "6":
            type: "basic"
            out: ["1", "2"]
        "7":
            type: "basic"
            out: ["6"]
        "8":
            type: "basic"
            out: ["6"]
        "9":
            type: "basic"
            out: ["5", "6"]
            be: ["6"]
        "10":
            type: "basic"
            out: ["9"]
        "11":
            type: "basic"
            out: ["9"]
        "12":
            type: "basic"
            out: ["9"]
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2", "3", "4"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_header_double_exiting(self):
        """This is like the example from Bahman2015 fig. 3.

        Two headers that need to be multiplexed to, on additional branch that
        becomes the exiting latch and one branch that becomes the exit.

        """
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
            out: ["2", "5"]
        "4":
            type: "basic"
            out: ["1", "6"]
        "5":
            type: "basic"
            out: ["7"]
        "6":
            type: "basic"
            out: ["7"]
        "7":
            type: "basic"
            out: []
        """
        expected = """
        "0":
            type: "basic"
            out: ["10", "9"]
        "1":
            type: "basic"
            out: ["3"]
        "2":
            type: "basic"
            out: ["4"]
        "3":
            type: "basic"
            out: ["13", "14"]
        "4":
            type: "basic"
            out: ["15", "16"]
        "5":
            type: "basic"
            out: ["7"]
        "6":
            type: "basic"
            out: ["7"]
        "7":
            type: "basic"
            out: []
        "8":
            type: "basic"
            out: ["1", "2"]
        "9":
            type: "basic"
            out: ["8"]
        "10":
            type: "basic"
            out: ["8"]
        "11":
            type: "basic"
            out: ["12", "8"]
            be: ["8"]
        "12":
            type: "basic"
            out: ["5", "6"]
        "13":
            type: "basic"
            out: ["11"]
        "14":
            type: "basic"
            out: ["11"]
        "15":
            type: "basic"
            out: ["11"]
        "16":
            type: "basic"
            out: ["11"]
        """
        original_scfg, block_ref_orig = SCFG.from_yaml(original)
        expected_scfg, block_ref_exp = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2", "3", "4"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)


if __name__ == "__main__":
    main()
