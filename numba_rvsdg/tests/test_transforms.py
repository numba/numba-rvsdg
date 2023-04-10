
from unittest import main

from numba_rvsdg.core.datastructures.labels import (
    ControlLabel,
    SyntheticTail,
    SyntheticExit,
)
from numba_rvsdg.core.datastructures.scfg import SCFG, wrap_id
from numba_rvsdg.core.transformations import loop_restructure_helper
from numba_rvsdg.tests.test_utils import SCFGComparator


class TestInsertBlock(SCFGComparator):
    def test_linear(self):
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["2"]
        "1":
            jt: []
        "2":
            jt: ["1"]
        """
        expected_scfg = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            ControlLabel("2"), wrap_id(("0",)), wrap_id(("1",))
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor(self):
        original = """
        "0":
            jt: ["2"]
        "1":
            jt: ["2"]
        "2":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["3"]
        "1":
            jt: ["3"]
        "2":
            jt: []
        "3":
            jt: ["2"]
        """
        expected_scfg = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            ControlLabel("3"), wrap_id(("0", "1")), wrap_id(("2",))
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_successor(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: []
        "2":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["3"]
        "1":
            jt: []
        "2":
            jt: []
        "3":
            jt: ["1", "2"]
        """
        expected_scfg = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            ControlLabel("3"),
            wrap_id(("0",)),
            wrap_id(("1", "2")),
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor_and_dual_successor(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["4"]
        "3":
            jt: []
        "4":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["5"]
        "2":
            jt: ["5"]
        "3":
            jt: []
        "4":
            jt: []
        "5":
            jt: ["3", "4"]
        """
        expected_scfg = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            ControlLabel("5"),
            wrap_id(("1", "2")),
            wrap_id(("3", "4")),
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor_and_dual_successor_with_additional_arcs(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["1", "4"]
        "3":
            jt: ["0"]
        "4":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
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
            jt: ["3", "4"]
        """
        expected_scfg = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            ControlLabel("5"),
            wrap_id(("1", "2")),
            wrap_id(("3", "4")),
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)


class TestJoinReturns(SCFGComparator):
    def test_two_returns(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: []
        "2":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["3"]
        "3":
            jt: []
        """
        expected_scfg = SCFG.from_yaml(expected)
        original_scfg.join_returns()
        self.assertSCFGEqual(expected_scfg, original_scfg)


class TestJoinTailsAndExits(SCFGComparator):
    def test_join_tails_and_exits_case_00(self):
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        expected_scfg = SCFG.from_yaml(expected)

        tails = wrap_id(("0",))
        exits = wrap_id(("1",))
        solo_tail_label, solo_exit_label = original_scfg.join_tails_and_exits(
            tails, exits
        )

        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(ControlLabel("0"), solo_tail_label)
        self.assertEqual(ControlLabel("1"), solo_exit_label)

    def test_join_tails_and_exits_case_01(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["3"]
        "3":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["4"]
        "1":
            jt: ["3"]
        "2":
            jt: ["3"]
        "3":
            jt: []
        "4":
            jt: ["1", "2"]
        """
        expected_scfg = SCFG.from_yaml(expected)

        tails = wrap_id(("0",))
        exits = wrap_id(("1", "2"))
        solo_tail_label, solo_exit_label = original_scfg.join_tails_and_exits(
            tails, exits
        )

        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(ControlLabel("0"), solo_tail_label)
        self.assertEqual(SyntheticExit("4"), solo_exit_label)

    def test_join_tails_and_exits_case_02_01(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["3"]
        "3":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["4"]
        "2":
            jt: ["4"]
        "3":
            jt: []
        "4":
            jt: ["3"]
        """
        expected_scfg = SCFG.from_yaml(expected)

        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3",))
        solo_tail_label, solo_exit_label = original_scfg.join_tails_and_exits(
            tails, exits
        )

        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(SyntheticTail("4"), solo_tail_label)
        self.assertEqual(ControlLabel("3"), solo_exit_label)

    def test_join_tails_and_exits_case_02_02(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["1", "3"]
        "3":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["4"]
        "2":
            jt: ["1", "4"]
        "3":
            jt: []
        "4":
            jt: ["3"]
        """
        expected_scfg = SCFG.from_yaml(expected)

        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3",))

        solo_tail_label, solo_exit_label = original_scfg.join_tails_and_exits(
            tails, exits
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(SyntheticTail("4"), solo_tail_label)
        self.assertEqual(ControlLabel("3"), solo_exit_label)

    def test_join_tails_and_exits_case_03_01(self):

        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["4"]
        "3":
            jt: ["5"]
        "4":
            jt: ["5"]
        "5":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["6"]
        "2":
            jt: ["6"]
        "3":
            jt: ["5"]
        "4":
            jt: ["5"]
        "5":
            jt: []
        "6":
            jt: ["7"]
        "7":
            jt: ["3", "4"]
        """
        expected_scfg = SCFG.from_yaml(expected)

        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3", "4"))
        solo_tail_label, solo_exit_label = original_scfg.join_tails_and_exits(
            tails, exits
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(SyntheticTail("6"), solo_tail_label)
        self.assertEqual(SyntheticExit("7"), solo_exit_label)

    def test_join_tails_and_exits_case_03_02(self):

        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["1", "4"]
        "3":
            jt: ["5"]
        "4":
            jt: ["5"]
        "5":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["6"]
        "2":
            jt: ["1", "6"]
        "3":
            jt: ["5"]
        "4":
            jt: ["5"]
        "5":
            jt: []
        "6":
            jt: ["7"]
        "7":
            jt: ["3", "4"]
        """
        expected_scfg = SCFG.from_yaml(expected)
        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3", "4"))
        solo_tail_label, solo_exit_label = original_scfg.join_tails_and_exits(
            tails, exits
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(SyntheticTail("6"), solo_tail_label)
        self.assertEqual(SyntheticExit("7"), solo_exit_label)


class TestLoopRestructure(SCFGComparator):

    def test_no_op_mono(self):
        """Loop consists of a single Block."""
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: ["1", "2"]
        "2":
            jt: []
        """
        expected = """
        "0":
            jt: ["1"]
        "1":
            jt: ["1", "2"]
            be: ["1"]
        "2":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected_scfg = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_no_op(self):
        """Loop consists of two blocks, but it's in form."""
        original ="""
        "0":
            jt: ["1"]
        "1":
            jt: ["2"]
        "2":
            jt: ["1", "3"]
        "3":
            jt: []
        """
        expected ="""
        "0":
            jt: ["1"]
        "1":
            jt: ["2"]
        "2":
            jt: ["1", "3"]
            be: ["1"]
        "3":
            jt: []
        """
        original_scfg = SCFG.from_yaml(original)
        expected_scfg = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_backedge_not_exiting(self):
        """Loop has a backedge not coming from the exiting block.

        This is the situation with the standard Python for loop.
        """
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: ["2", "3"]
        "2":
            jt: ["1"]
        "3":
            jt: []
        """
        expected = """
        "0":
            jt: ["1"]
        "1":
            jt: ["2", "5"]
        "2":
            jt: ["6"]
        "3":
            jt: []
        "4":
            jt: ["1", "3"]
            be: ["1"]
        "5":
            jt: ["4"]
        "6":
            jt: ["4"]
        """
        original_scfg = SCFG.from_yaml(original)
        expected_scfg = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_exit(self):
        """Loop has two exiting blocks.

        For example a loop with a break.

        """
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: ["2"]
        "2":
            jt: ["3", "4"]
        "3":
            jt: ["1", "4"]
        "4":
            jt: []
        """
        expected = """
        "0":
            jt: ["1"]
        "1":
            jt: ["2"]
        "2":
            jt: ["3", "6"]
        "3":
            jt: ["7", "8"]
        "4":
            jt: []
        "5":
            jt: ["1", "4"]
            be: ["1"]
        "6":
            jt: ["5"]
        "7":
            jt: ["5"]
        "8":
            jt: ["5"]
        """
        original_scfg = SCFG.from_yaml(original)
        expected_scfg = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2", "3"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_header(self):
        """ This is like the example from Bahman2015 fig. 3 --
        but with one exiting block removed."""
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["4"]
        "3":
            jt: ["2", "5"]
        "4":
            jt: ["1"]
        "5":
            jt: []
        """
        expected = """
        "0":
            jt: ["7", "8"]
        "1":
            jt: ["3"]
        "2":
            jt: ["4"]
        "3":
            jt: ["10", "11"]
        "4":
            jt: ["12"]
        "5":
            jt: []
        "6":
            jt: ["1", "2"]
        "7":
            jt: ["6"]
        "8":
            jt: ["6"]
        "9":
            jt: ["5", "6"]
            be: ["6"]
        "10":
            jt: ["9"]
        "11":
            jt: ["9"]
        "12":
            jt: ["9"]
        """
        original_scfg = SCFG.from_yaml(original)
        expected_scfg = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2", "3", "4"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_header_double_exiting(self):
        """ This is like the example from Bahman2015 fig. 3.

        Two headers that need to be multiplexed to, on additional branch that
        becomes the exiting latch and one branch that becomes the exit.
        
        """
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["3"]
        "2":
            jt: ["4"]
        "3":
            jt: ["2", "5"]
        "4":
            jt: ["1", "6"]
        "5":
            jt: ["7"]
        "6":
            jt: ["7"]
        "7":
            jt: []
        """
        expected = """
        "0":
            jt: ["10", "9"]
        "1":
            jt: ["3"]
        "2":
            jt: ["4"]
        "3":
            jt: ["13", "14"]
        "4":
            jt: ["15", "16"]
        "5":
            jt: ["7"]
        "6":
            jt: ["7"]
        "7":
            jt: []
        "8":
            jt: ["1", "2"]
        "9":
            jt: ["8"]
        "10":
            jt: ["8"]
        "11":
            jt: ["12", "8"]
            be: ["8"]
        "12":
            jt: ["5", "6"]
        "13":
            jt: ["11"]
        "14":
            jt: ["11"]
        "15":
            jt: ["11"]
        "16":
            jt: ["11"]
        """
        original_scfg = SCFG.from_yaml(original)
        expected_scfg = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set(wrap_id({"1", "2", "3", "4"})))
        self.assertSCFGEqual(expected_scfg, original_scfg)

if __name__ == "__main__":
    main()
