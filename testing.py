from typing import Set

import yaml
from unittest import TestCase, main

from byteflow2 import (Label, ControlLabel, BasicBlock, BlockMap,
                       ByteFlowRenderer, ByteFlow, ControlLabelGenerator,
                       loop_rotate, SyntheticTail, SyntheticExit)




class MapComparator(TestCase):

    def assertMapEqual(self, first_map, second_map):

        for key1, key2 in zip(sorted(first_map.graph.keys(), key=lambda x:
                                     x.index),
                              sorted(second_map.graph.keys(), key=lambda x:
                                     x.index)):
            # compare indices of labels
            self.assertEqual(key1.index, key2.index)
            # compare indices of jump_targets
            self.assertEqual(sorted([j.index for j in
                                     first_map[key1]._jump_targets]),
                             sorted([j.index for j in
                                     second_map[key2]._jump_targets]))
            # compare indices of backedges
            self.assertEqual(sorted([j.index for j in
                                     first_map[key1].backedges]),
                             sorted([j.index for j in
                                     second_map[key2].backedges]))

    def wrap_id(self, indices: Set[Label]):
        return tuple([ControlLabel(i) for i in indices])

    def from_yaml(self, yaml_string):
        # Convert to BlockMap
        data = yaml.safe_load(yaml_string)
        block_map_graph = {}
        clg = ControlLabelGenerator()
        for index, attributes in data.items():
            jump_targets = attributes["jt"]
            backedges = attributes.get("be", ())
            label = ControlLabel(str(clg.new_index()))
            block = BasicBlock(
                label=label,
                backedges=self.wrap_id(backedges),
                _jump_targets=self.wrap_id(jump_targets),
            )
            block_map_graph[label] = block
        return BlockMap(block_map_graph, clg=clg)


class TestInsertBlock(MapComparator):

    def test_linear(self):
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        original_block_map = self.from_yaml(original)
        expected = """
        "0":
            jt: ["2"]
        "1":
            jt: []
        "2":
            jt: ["1"]
        """
        expected_block_map = self.from_yaml(expected)
        original_block_map.insert_block(ControlLabel("2"),
                                        self.wrap_id(("0",)),
                                        self.wrap_id(("1",)))
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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)
        original_block_map.insert_block(ControlLabel("3"),
                                        self.wrap_id(("0", "1")),
                                        self.wrap_id(("2",)))
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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)
        original_block_map.insert_block(ControlLabel("3"),
                                        self.wrap_id(("0",)),
                                        self.wrap_id(("1", "2",)))
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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)
        original_block_map.insert_block(ControlLabel("5"),
                                        self.wrap_id(("1", "2")),
                                        self.wrap_id(("3", "4",)))
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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)
        original_block_map.insert_block(ControlLabel("5"),
                                        self.wrap_id(("1", "2")),
                                        self.wrap_id(("3", "4",)))
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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)
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
        original_block_map = self.from_yaml(original)
        expected = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        expected_block_map = self.from_yaml(expected)

        tails = self.wrap_id(("0",))
        exits = self.wrap_id(("1",))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)

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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)

        tails = self.wrap_id(("0",))
        exits = self.wrap_id(("1", "2"))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)

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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)

        tails = self.wrap_id(("1", "2"))
        exits = self.wrap_id(("3",))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)

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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)

        tails = self.wrap_id(("1", "2"))
        exits = self.wrap_id(("3",))

        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)
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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)

        tails = self.wrap_id(("1", "2"))
        exits = self.wrap_id(("3", "4"))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)
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
        original_block_map = self.from_yaml(original)
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
        expected_block_map = self.from_yaml(expected)
        tails = self.wrap_id(("1", "2"))
        exits = self.wrap_id(("3", "4"))
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)
        self.assertMapEqual(expected_block_map, original_block_map)
        self.assertEqual(SyntheticTail("6"), solo_tail_label)
        self.assertEqual(SyntheticExit("7"), solo_exit_label)


class TestLoopRotate(MapComparator):

    def test_basic_for_loop(self):

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
        original_block_map = self.from_yaml(original)
        expected = """
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
            be: ["2"]
        """
        expected_block_map = self.from_yaml(expected)

        loop_rotate(original_block_map, {ControlLabel("1"), ControlLabel("2")})
        print(original_block_map.compute_scc())
        self.assertMapEqual(expected_block_map, original_block_map)

    def test_basic_while_loop(self):
        original = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["1", "2"]
        "2":
            jt: []
        """
        original_block_map = self.from_yaml(original)
        expected = """
        "0":
            jt: ["1", "2"]
        "1":
            jt: ["1", "2"]
            be: ["1"]
        "2":
            jt: []
        """
        expected_block_map = self.from_yaml(expected)

        loop_rotate(original_block_map, {ControlLabel("1")})
        print(original_block_map.compute_scc())
        self.assertMapEqual(expected_block_map, original_block_map)


if __name__ == '__main__':
    main()
