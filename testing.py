import yaml

from byteflow2 import (ControlLabel, BasicBlock, BlockMap, ByteFlowRenderer,
                       ByteFlow, join_pre_exits, ControlLabelGenerator)

from unittest import TestCase, main


def from_yaml(yaml_string):
    # Convert to BlockMap
    data = yaml.safe_load(yaml_string)
    block_map_graph = {}
    clg = ControlLabelGenerator()
    for index, jump_targets in data.items():
        begin_label = ControlLabel(str(clg.new_index()))
        end_label = ControlLabel("end")
        block = BasicBlock(
            begin_label,
            end_label,
            fallthrough=len(jump_targets["jt"]) == 1,
            backedges=set(),
            jump_targets=set((ControlLabel(i) for i in jump_targets["jt"]))
        )
        block_map_graph[begin_label] = block
    return BlockMap(block_map_graph)


class TestJoinTailsAndExits(TestCase):

    def test_join_tails_and_exits_case_00(self):
        original = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        original_block_map = from_yaml(original)
        expected = """
        "0":
            jt: ["1"]
        "1":
            jt: []
        """
        expected_block_map = from_yaml(expected)

        tails = {ControlLabel(i) for i in ("0")}
        exits = {ControlLabel(i) for i in ("1")}
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)

        self.assertEqual(expected_block_map, original_block_map)
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
        original_block_map = from_yaml(original)
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
        expected_block_map = from_yaml(expected)

        tails = {ControlLabel(i) for i in ("0")}
        exits = {ControlLabel(i) for i in ("1", "2")}
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)

        self.assertEqual(expected_block_map, original_block_map)
        self.assertEqual(ControlLabel("0"), solo_tail_label)
        self.assertEqual(ControlLabel("4"), solo_exit_label)

    def test_join_tails_and_exits_case_02(self):
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
        original_block_map = from_yaml(original)
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
        expected_block_map = from_yaml(expected)

        tails = {ControlLabel(i) for i in ("1", "2")}
        exits = {ControlLabel(i) for i in ("3")}
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)

        self.assertEqual(expected_block_map, original_block_map)
        self.assertEqual(ControlLabel("4"), solo_tail_label)
        self.assertEqual(ControlLabel("3"), solo_exit_label)

    def test_join_tails_and_exits_case_02_01(self):
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
        original_block_map = from_yaml(original)
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
        expected_block_map = from_yaml(expected)

        tails = {ControlLabel(i) for i in ("1", "2")}
        exits = {ControlLabel(i) for i in ("3")}
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)
        self.assertEqual(expected_block_map, original_block_map)
        self.assertEqual(ControlLabel("4"), solo_tail_label)
        self.assertEqual(ControlLabel("3"), solo_exit_label)

    def test_join_tails_and_exits_case_03(self):

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
        original_block_map = from_yaml(original)
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
        expected_block_map = from_yaml(expected)

        tails = {ControlLabel(i) for i in ("1", "2")}
        exits = {ControlLabel(i) for i in ("3", "4")}
        solo_tail_label, solo_exit_label = original_block_map.join_tails_and_exits(tails, exits)
        self.assertEqual(expected_block_map, original_block_map)
        self.assertEqual(ControlLabel("6"), solo_tail_label)
        self.assertEqual(ControlLabel("7"), solo_exit_label)


if __name__ == '__main__':
    main()
