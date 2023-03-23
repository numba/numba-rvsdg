
from unittest import main

from numba_rvsdg.core.datastructures.labels import (
    ControlLabel,
    SyntheticTail,
    SyntheticExit,
)
from numba_rvsdg.core.datastructures.block_map import BlockMap, wrap_id
from numba_rvsdg.core.transformations import loop_restructure_helper
from numba_rvsdg.tests.test_utils import MapComparator


class TestInsertBlock(MapComparator):
    def test_linear(self):
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        original_block_map = BlockMap.from_yaml(original)
        expected = """
        "0":
            jt: ["2"]
        "1":
            jt: []
        "2":
            jt: ["1"]
        """
        expected_block_map = BlockMap.from_yaml(expected)
        original_block_map.insert_block(
            ControlLabel("2"), wrap_id(("0",)), wrap_id(("1",))
        )
        self.assertMapEqual(expected_block_map, original_block_map)

    def test_dual_predecessor(self):
        original = """
        "0":
            jt: ["2"]
        "1":
            jt: ["2"]
        "2":
            jt: []
        """
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)
        original_block_map.insert_block(
            ControlLabel("3"), wrap_id(("0", "1")), wrap_id(("2",))
        )
        self.assertMapEqual(expected_block_map, original_block_map)

    def test_dual_successor(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: []
        "2":
            jt: []
        """
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)
        original_block_map.insert_block(
            ControlLabel("3"),
            wrap_id(("0",)),
            wrap_id(("1", "2")),
        )
        self.assertMapEqual(expected_block_map, original_block_map)

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
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)
        original_block_map.insert_block(
            ControlLabel("5"),
            wrap_id(("1", "2")),
            wrap_id(("3", "4")),
        )
        self.assertMapEqual(expected_block_map, original_block_map)

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
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)
        original_block_map.insert_block(
            ControlLabel("5"),
            wrap_id(("1", "2")),
            wrap_id(("3", "4")),
        )
        self.assertMapEqual(expected_block_map, original_block_map)


class TestJoinReturns(MapComparator):
    def test_two_returns(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: []
        "2":
            jt: []
        """
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)
        original_block_map.join_returns()
        self.assertMapEqual(expected_block_map, original_block_map)


class TestJoinTailsAndExits(MapComparator):
    def test_join_tails_and_exits_case_00(self):
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        original_block_map = BlockMap.from_yaml(original)
        expected = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        expected_block_map = BlockMap.from_yaml(expected)

        tails = wrap_id(("0",))
        exits = wrap_id(("1",))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(
            tails, exits
        )

        self.assertMapEqual(expected_block_map, original_block_map)
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
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)

        tails = wrap_id(("0",))
        exits = wrap_id(("1", "2"))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(
            tails, exits
        )

        self.assertMapEqual(expected_block_map, original_block_map)
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
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)

        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3",))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(
            tails, exits
        )

        self.assertMapEqual(expected_block_map, original_block_map)
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
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)

        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3",))

        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(
            tails, exits
        )
        self.assertMapEqual(expected_block_map, original_block_map)
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
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)

        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3", "4"))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(
            tails, exits
        )
        self.assertMapEqual(expected_block_map, original_block_map)
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
        original_block_map = BlockMap.from_yaml(original)
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
        expected_block_map = BlockMap.from_yaml(expected)
        tails = wrap_id(("1", "2"))
        exits = wrap_id(("3", "4"))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(
            tails, exits
        )
        self.assertMapEqual(expected_block_map, original_block_map)
        self.assertEqual(SyntheticTail("6"), solo_tail_label)
        self.assertEqual(SyntheticExit("7"), solo_exit_label)


class TestLoopRestructure(MapComparator):

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
        original_block_map = self.from_yaml(original)
        expected_block_map = self.from_yaml(expected)
        loop_restructure_helper(original_block_map, set(self.wrap_id({"1"})))
        self.assertMapEqual(expected_block_map, original_block_map)


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
        original_block_map = self.from_yaml(original)
        expected_block_map = self.from_yaml(expected)
        loop_restructure_helper(original_block_map, set(self.wrap_id({"1", "2"})))
        self.assertMapEqual(expected_block_map, original_block_map)


if __name__ == "__main__":
    main()
