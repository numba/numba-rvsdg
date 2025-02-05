# mypy: ignore-errors

from unittest import main

from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.core.datastructures.basic_block import BasicBlock
from numba_scfg.core.transformations import loop_restructure_helper
from numba_scfg.tests.test_utils import SCFGComparator
from numba_scfg.core.datastructures import block_names


class TestInsertBlock(SCFGComparator):
    def test_linear(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
        edges:
            '0': ['1']
            '1': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
        edges:
            '0': ['2']
            '1': []
            '2': ['1']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)
        new_name = original_scfg.name_gen.new_block_name(block_names.BASIC)
        original_scfg.insert_block(
            new_name, (block_dict["0"],), (block_dict["1"],), BasicBlock
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
        edges:
            '0': ['2']
            '1': ['2']
            '2': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['3']
            '1': ['3']
            '2': []
            '3': ['2']
        backedges:
        """
        expected_scfg, expected_block_dict = SCFG.from_yaml(expected)
        new_name = original_scfg.name_gen.new_block_name(block_names.BASIC)
        original_scfg.insert_block(
            new_name,
            (block_dict["0"], block_dict["1"]),
            (block_dict["2"],),
            BasicBlock,
        )
        self.assertSCFGEqual(
            expected_scfg,
            original_scfg,
            {
                block_dict["0"]: expected_block_dict["0"],
                block_dict["1"]: expected_block_dict["1"],
            },
        )

    def test_dual_successor(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
        edges:
            '0': ['1', '2']
            '1': []
            '2': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['3']
            '1': []
            '2': []
            '3': ['1', '2']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            original_scfg.name_gen.new_block_name(block_names.BASIC),
            (block_dict["0"],),
            (block_dict["1"], block_dict["2"]),
            BasicBlock,
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor_and_dual_successor(self):
        original = """
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
            '3': []
            '4': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
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
            '2': ['5']
            '3': []
            '4': []
            '5': ['3', '4']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            original_scfg.name_gen.new_block_name(block_names.BASIC),
            (block_dict["1"], block_dict["2"]),
            (block_dict["3"], block_dict["4"]),
            BasicBlock,
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_dual_predecessor_and_dual_successor_with_additional_arcs(self):
        original = """
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
            '2': ['1', '4']
            '3': ['0']
            '4': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
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
            '3': ['0']
            '4': []
            '5': ['3', '4']
        backedges:
        """
        expected_scfg, expected_block_dict = SCFG.from_yaml(expected)
        original_scfg.insert_block(
            original_scfg.name_gen.new_block_name(block_names.BASIC),
            (block_dict["1"], block_dict["2"]),
            (block_dict["3"], block_dict["4"]),
            BasicBlock,
        )
        self.assertSCFGEqual(
            expected_scfg,
            original_scfg,
            {
                block_dict["0"]: expected_block_dict["0"],
                block_dict["2"]: expected_block_dict["2"],
            },
        )


class TestJoinReturns(SCFGComparator):
    def test_two_returns(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
        edges:
            '0': ['1', '2']
            '1': []
            '2': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1', '2']
            '1': ['3']
            '2': ['3']
            '3': []
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)
        original_scfg.join_returns()
        self.assertSCFGEqual(expected_scfg, original_scfg)


class TestJoinTailsAndExits(SCFGComparator):
    def test_join_tails_and_exits_case_00(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
        edges:
            '0': ['1']
            '1': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
        edges:
            '0': ['1']
            '1': []
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)

        tails = (block_dict["0"],)
        exits = (block_dict["1"],)
        solo_tail_name, solo_exit_name = original_scfg.join_tails_and_exits(
            tails, exits
        )

        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(block_dict["0"], solo_tail_name)
        self.assertEqual(block_dict["1"], solo_exit_name)

    def test_join_tails_and_exits_case_01(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1', '2']
            '1': ['3']
            '2': ['3']
            '3': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
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
            '0': ['4']
            '1': ['3']
            '2': ['3']
            '3': []
            '4': ['1', '2']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)

        tails = (block_dict["0"],)
        exits = (block_dict["1"], block_dict["2"])
        solo_tail_name, solo_exit_name = original_scfg.join_tails_and_exits(
            tails, exits
        )

        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(block_dict["0"], solo_tail_name)
        self.assertEqual(
            expected_scfg.name_gen.new_block_name(block_names.SYNTH_EXIT),
            solo_exit_name,
        )

    def test_join_tails_and_exits_case_02_01(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1', '2']
            '1': ['3']
            '2': ['3']
            '3': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
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
            '1': ['4']
            '2': ['4']
            '3': []
            '4': ['3']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)

        tails = (block_dict["1"], block_dict["2"])
        exits = (block_dict["3"],)
        solo_tail_name, solo_exit_name = original_scfg.join_tails_and_exits(
            tails, exits
        )

        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(
            expected_scfg.name_gen.new_block_name(block_names.SYNTH_TAIL),
            solo_tail_name,
        )
        self.assertEqual(block_dict["3"], solo_exit_name)

    def test_join_tails_and_exits_case_02_02(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1', '2']
            '1': ['3']
            '2': ['1', '3']
            '3': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
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
            '1': ['4']
            '2': ['1', '4']
            '3': []
            '4': ['3']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)

        tails = (block_dict["1"], block_dict["2"])
        exits = (block_dict["3"],)

        solo_tail_name, solo_exit_name = original_scfg.join_tails_and_exits(
            tails, exits
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(
            expected_scfg.name_gen.new_block_name(block_names.SYNTH_TAIL),
            solo_tail_name,
        )
        self.assertEqual(block_dict["3"], solo_exit_name)

    def test_join_tails_and_exits_case_03_01(self):
        original = """
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
            '1': ['3']
            '2': ['4']
            '3': ['5']
            '4': ['5']
            '5': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
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
            '6':
                type: basic
            '7':
                type: basic
        edges:
            '0': ['1', '2']
            '1': ['6']
            '2': ['6']
            '3': ['5']
            '4': ['5']
            '5': []
            '6': ['7']
            '7': ['3', '4']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)

        tails = (block_dict["1"], block_dict["2"])
        exits = (block_dict["3"], block_dict["4"])
        solo_tail_name, solo_exit_name = original_scfg.join_tails_and_exits(
            tails, exits
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(
            expected_scfg.name_gen.new_block_name(block_names.SYNTH_TAIL),
            solo_tail_name,
        )
        self.assertEqual(
            expected_scfg.name_gen.new_block_name(block_names.SYNTH_EXIT),
            solo_exit_name,
        )

    def test_join_tails_and_exits_case_03_02(self):
        original = """
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
            '1': ['3']
            '2': ['1', '4']
            '3': ['5']
            '4': ['5']
            '5': []
        backedges:
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected = """
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
            '6':
                type: basic
            '7':
                type: basic
        edges:
            '0': ['1', '2']
            '1': ['6']
            '2': ['1', '6']
            '3': ['5']
            '4': ['5']
            '5': []
            '6': ['7']
            '7': ['3', '4']
        backedges:
        """
        expected_scfg, _ = SCFG.from_yaml(expected)
        tails = (block_dict["1"], block_dict["2"])
        exits = (block_dict["3"], block_dict["4"])
        solo_tail_name, solo_exit_name = original_scfg.join_tails_and_exits(
            tails, exits
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)
        self.assertEqual(
            expected_scfg.name_gen.new_block_name(block_names.SYNTH_TAIL),
            solo_tail_name,
        )
        self.assertEqual(
            expected_scfg.name_gen.new_block_name(block_names.SYNTH_EXIT),
            solo_exit_name,
        )


class TestLoopRestructure(SCFGComparator):
    def test_no_op_mono(self):
        """Loop consists of a single Block."""
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
        edges:
            '0': ['1']
            '1': ['1', '2']
            '2': []
        backedges:
        """
        expected = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
        edges:
            '0': ['1']
            '1': ['1', '2']
            '2': []
        backedges:
            '1': ['1']
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected_scfg, _ = SCFG.from_yaml(expected)
        loop_restructure_helper(original_scfg, set({block_dict["1"]}))
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_no_op(self):
        """Loop consists of two blocks, but it's in form."""
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1']
            '1': ['2']
            '2': ['1', '3']
            '3': []
        backedges:
        """
        expected = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1']
            '1': ['2']
            '2': ['1', '3']
            '3': []
        backedges:
            '2': ['1']
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected_scfg, _ = SCFG.from_yaml(expected)
        loop_restructure_helper(
            original_scfg, set({block_dict["1"], block_dict["2"]})
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_backedge_not_exiting(self):
        """Loop has a backedge not coming from the exiting block.

        This is the situation with the standard Python for loop.
        """
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1']
            '1': ['2', '3']
            '2': ['1']
            '3': []
        backedges:
        """
        expected = """
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
            '6':
                type: basic
        edges:
            '0': ['1']
            '1': ['2', '5']
            '2': ['6']
            '3': []
            '4': ['1', '3']
            '5': ['4']
            '6': ['4']
        backedges:
            '4': ['1']
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected_scfg, _ = SCFG.from_yaml(expected)
        loop_restructure_helper(
            original_scfg, set({block_dict["1"], block_dict["2"]})
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_multi_back_edge_with_backedge_from_header(self):
        original = """
        blocks:
            '0':
                type: basic
            '1':
                type: basic
            '2':
                type: basic
            '3':
                type: basic
        edges:
            '0': ['1']
            '1': ['1', '2']
            '2': ['1', '3']
            '3': []
        backedges:
        """
        expected = """
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
            '6':
                type: basic
            '7':
                type: basic
        edges:
            '0': ['1']
            '1': ['5', '2']
            '2': ['6', '7']
            '3': []
            '4': ['1', '3']
            '5': ['4']
            '6': ['4']
            '7': ['4']
        backedges:
            '4': ['1']
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected_scfg, _ = SCFG.from_yaml(expected)
        loop_restructure_helper(
            original_scfg, set({block_dict["1"], block_dict["2"]})
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_exit(self):
        """Loop has two exiting blocks.

        For example a loop with a break.

        """
        original = """
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
            '1': ['2']
            '2': ['3', '4']
            '3': ['1', '4']
            '4': []
        backedges:
        """
        expected = """
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
            '6':
                type: basic
            '7':
                type: basic
            '8':
                type: basic
        edges:
            '0': ['1']
            '1': ['2']
            '2': ['3', '6']
            '3': ['7', '8']
            '4': []
            '5': ['1', '4']
            '6': ['5']
            '7': ['5']
            '8': ['5']
        backedges:
            '5': ['1']
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected_scfg, _ = SCFG.from_yaml(expected)
        loop_restructure_helper(
            original_scfg,
            set({block_dict["1"], block_dict["2"], block_dict["3"]}),
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_header(self):
        """This is like the example from Bahman2015 fig. 3 --
        but with one exiting block removed."""
        original = """
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
            '1': ['3']
            '2': ['4']
            '3': ['2', '5']
            '4': ['1']
            '5': []
        backedges:
        """
        expected = """
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
            '6':
                type: basic
            '7':
                type: basic
            '8':
                type: basic
            '9':
                type: basic
            '10':
                type: basic
            '11':
                type: basic
            '12':
                type: basic
        edges:
            '0': ['7', '8']
            '1': ['3']
            '2': ['4']
            '3': ['10', '11']
            '4': ['12']
            '5': []
            '6': ['1', '2']
            '7': ['6']
            '8': ['6']
            '9': ['5', '6']
            '10': ['9']
            '11': ['9']
            '12': ['9']
        backedges:
            '9': ['6']
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected_scfg, _ = SCFG.from_yaml(expected)
        loop_restructure_helper(
            original_scfg,
            set(
                {
                    block_dict["1"],
                    block_dict["2"],
                    block_dict["3"],
                    block_dict["4"],
                }
            ),
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)

    def test_double_header_double_exiting(self):
        """This is like the example from Bahman2015 fig. 3.

        Two headers that need to be multiplexed to, on additional branch that
        becomes the exiting latch and one branch that becomes the exit.

        """
        original = """
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
            '6':
                type: basic
            '7':
                type: basic
        edges:
            '0': ['1', '2']
            '1': ['3']
            '2': ['4']
            '3': ['2', '5']
            '4': ['1', '6']
            '5': ['7']
            '6': ['7']
            '7': []
        backedges:
        """
        expected = """
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
            '6':
                type: basic
            '7':
                type: basic
            '8':
                type: basic
            '9':
                type: basic
            '10':
                type: basic
            '11':
                type: basic
            '12':
                type: basic
            '13':
                type: basic
            '14':
                type: basic
            '15':
                type: basic
            '16':
                type: basic
        edges:
            '0': ['10', '9']
            '1': ['3']
            '2': ['4']
            '3': ['13', '14']
            '4': ['15', '16']
            '5': ['7']
            '6': ['7']
            '7': []
            '8': ['1', '2']
            '9': ['8']
            '10': ['8']
            '11': ['12', '8']
            '12': ['5', '6']
            '13': ['11']
            '14': ['11']
            '15': ['11']
            '16': ['11']
        backedges:
            '11': ['8']
        """
        original_scfg, block_dict = SCFG.from_yaml(original)
        expected_scfg, _ = SCFG.from_yaml(expected)
        loop_restructure_helper(
            original_scfg,
            set(
                {
                    block_dict["1"],
                    block_dict["2"],
                    block_dict["3"],
                    block_dict["4"],
                }
            ),
        )
        self.assertSCFGEqual(expected_scfg, original_scfg)


if __name__ == "__main__":
    main()
