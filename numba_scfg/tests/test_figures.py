# mypy: ignore-errors

from numba_scfg.core.datastructures.flow_info import FlowInfo
from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.tests.test_utils import SCFGComparator
import dis

fig_3_yaml = """
blocks:
    branch_region_0:
        type: region
        kind: branch
        contains: ['synth_asign_block_0']
        header: synth_asign_block_0
        exiting: synth_asign_block_0
        parent_region: meta_region_0
    branch_region_1:
        type: region
        kind: branch
        contains: ['synth_asign_block_1']
        header: synth_asign_block_1
        exiting: synth_asign_block_1
        parent_region: meta_region_0
    branch_region_2:
        type: region
        kind: branch
        contains: ['python_bytecode_block_2']
        header: python_bytecode_block_2
        exiting: python_bytecode_block_2
        parent_region: tail_region_0
    branch_region_3:
        type: region
        kind: branch
        contains: ['python_bytecode_block_4']
        header: python_bytecode_block_4
        exiting: python_bytecode_block_4
        parent_region: tail_region_0
    branch_region_4:
        type: region
        kind: branch
        contains: ['branch_region_6', 'branch_region_7', 'head_region_3', 'tail_region_3'] # noqa
        header: head_region_3
        exiting: tail_region_3
        parent_region: loop_region_0
    branch_region_5:
        type: region
        kind: branch
        contains: ['branch_region_8', 'branch_region_9', 'head_region_4', 'tail_region_4'] # noqa
        header: head_region_4
        exiting: tail_region_4
        parent_region: loop_region_0
    branch_region_6:
        type: region
        kind: branch
        contains: ['synth_asign_block_2']
        header: synth_asign_block_2
        exiting: synth_asign_block_2
        parent_region: branch_region_4
    branch_region_7:
        type: region
        kind: branch
        contains: ['synth_asign_block_3']
        header: synth_asign_block_3
        exiting: synth_asign_block_3
        parent_region: branch_region_4
    branch_region_8:
        type: region
        kind: branch
        contains: ['synth_asign_block_4']
        header: synth_asign_block_4
        exiting: synth_asign_block_4
        parent_region: branch_region_5
    branch_region_9:
        type: region
        kind: branch
        contains: ['synth_asign_block_5']
        header: synth_asign_block_5
        exiting: synth_asign_block_5
        parent_region: branch_region_5
    head_region_0:
        type: region
        kind: head
        contains: ['python_bytecode_block_0']
        header: python_bytecode_block_0
        exiting: python_bytecode_block_0
        parent_region: meta_region_0
    head_region_1:
        type: region
        kind: head
        contains: ['loop_region_0', 'synth_exit_block_0']
        header: loop_region_0
        exiting: synth_exit_block_0
        parent_region: tail_region_0
    head_region_2:
        type: region
        kind: head
        contains: ['synth_head_block_0']
        header: synth_head_block_0
        exiting: synth_head_block_0
        parent_region: loop_region_0
    head_region_3:
        type: region
        kind: head
        contains: ['python_bytecode_block_1']
        header: python_bytecode_block_1
        exiting: python_bytecode_block_1
        parent_region: branch_region_4
    head_region_4:
        type: region
        kind: head
        contains: ['python_bytecode_block_3']
        header: python_bytecode_block_3
        exiting: python_bytecode_block_3
        parent_region: branch_region_5
    loop_region_0:
        type: region
        kind: loop
        contains: ['branch_region_4', 'branch_region_5', 'head_region_2', 'tail_region_2']
        header: head_region_2
        exiting: tail_region_2
        parent_region: head_region_1
    python_bytecode_block_0:
        type: python_bytecode
        begin: 0
        end: 4
    python_bytecode_block_1:
        type: python_bytecode
        begin: 4
        end: 8
    python_bytecode_block_2:
        type: python_bytecode
        begin: 8
        end: 12
    python_bytecode_block_3:
        type: python_bytecode
        begin: 12
        end: 16
    python_bytecode_block_4:
        type: python_bytecode
        begin: 16
        end: 20
    python_bytecode_block_5:
        type: python_bytecode
        begin: 20
        end: 22
    synth_asign_block_0:
        type: synth_asign
        variable_assignment: {'__scfg_control_var_0__': 0}
    synth_asign_block_1:
        type: synth_asign
        variable_assignment: {'__scfg_control_var_0__': 1}
    synth_asign_block_2:
        type: synth_asign
        variable_assignment: {'__scfg_control_var_0__': 0, '__scfg_backedge_var_0__': 1}
    synth_asign_block_3:
        type: synth_asign
        variable_assignment: {'__scfg_backedge_var_0__': 0, '__scfg_control_var_0__': 1}
    synth_asign_block_4:
        type: synth_asign
        variable_assignment: {'__scfg_control_var_0__': 1, '__scfg_backedge_var_0__': 1}
    synth_asign_block_5:
        type: synth_asign
        variable_assignment: {'__scfg_backedge_var_0__': 0, '__scfg_control_var_0__': 0}
    synth_exit_block_0:
        type: synth_exit_branch
        branch_value_table: {0: 'branch_region_2', 1: 'branch_region_3'}
        variable: __scfg_control_var_0__
    synth_exit_latch_block_0:
        type: synth_exit_latch
        branch_value_table: {1: 'synth_exit_block_0', 0: 'head_region_2'}
        variable: __scfg_backedge_var_0__
    synth_head_block_0:
        type: synth_head
        branch_value_table: {0: 'branch_region_4', 1: 'branch_region_5'}
        variable: __scfg_control_var_0__
    synth_tail_block_0:
        type: synth_tail
    synth_tail_block_1:
        type: synth_tail
    tail_region_0:
        type: region
        kind: tail
        contains: ['branch_region_2', 'branch_region_3', 'head_region_1', 'tail_region_1']
        header: head_region_1
        exiting: tail_region_1
        parent_region: meta_region_0
    tail_region_1:
        type: region
        kind: tail
        contains: ['python_bytecode_block_5']
        header: python_bytecode_block_5
        exiting: python_bytecode_block_5
        parent_region: tail_region_0
    tail_region_2:
        type: region
        kind: tail
        contains: ['synth_exit_latch_block_0']
        header: synth_exit_latch_block_0
        exiting: synth_exit_latch_block_0
        parent_region: loop_region_0
    tail_region_3:
        type: region
        kind: tail
        contains: ['synth_tail_block_0']
        header: synth_tail_block_0
        exiting: synth_tail_block_0
        parent_region: branch_region_4
    tail_region_4:
        type: region
        kind: tail
        contains: ['synth_tail_block_1']
        header: synth_tail_block_1
        exiting: synth_tail_block_1
        parent_region: branch_region_5
edges:
    branch_region_0: ['tail_region_0']
    branch_region_1: ['tail_region_0']
    branch_region_2: ['tail_region_1']
    branch_region_3: ['tail_region_1']
    branch_region_4: ['tail_region_2']
    branch_region_5: ['tail_region_2']
    branch_region_6: ['tail_region_3']
    branch_region_7: ['tail_region_3']
    branch_region_8: ['tail_region_4']
    branch_region_9: ['tail_region_4']
    head_region_0: ['branch_region_0', 'branch_region_1']
    head_region_1: ['branch_region_2', 'branch_region_3']
    head_region_2: ['branch_region_4', 'branch_region_5']
    head_region_3: ['branch_region_6', 'branch_region_7']
    head_region_4: ['branch_region_8', 'branch_region_9']
    loop_region_0: ['synth_exit_block_0']
    python_bytecode_block_0: ['branch_region_0', 'branch_region_1']
    python_bytecode_block_1: ['branch_region_6', 'branch_region_7']
    python_bytecode_block_2: ['tail_region_1']
    python_bytecode_block_3: ['branch_region_8', 'branch_region_9']
    python_bytecode_block_4: ['tail_region_1']
    python_bytecode_block_5: []
    synth_asign_block_0: ['tail_region_0']
    synth_asign_block_1: ['tail_region_0']
    synth_asign_block_2: ['tail_region_3']
    synth_asign_block_3: ['tail_region_3']
    synth_asign_block_4: ['tail_region_4']
    synth_asign_block_5: ['tail_region_4']
    synth_exit_block_0: ['branch_region_2', 'branch_region_3']
    synth_exit_latch_block_0: ['head_region_2', 'synth_exit_block_0']
    synth_head_block_0: ['branch_region_4', 'branch_region_5']
    synth_tail_block_0: ['tail_region_2']
    synth_tail_block_1: ['tail_region_2']
    tail_region_0: []
    tail_region_1: []
    tail_region_2: ['synth_exit_block_0']
    tail_region_3: ['tail_region_2']
    tail_region_4: ['tail_region_2']
backedges:
    synth_exit_latch_block_0: ['head_region_2']"""

fig_4_yaml = """
blocks:
    branch_region_0:
        type: region
        kind: branch
        contains: ['branch_region_2', 'branch_region_3', 'head_region_1', 'tail_region_1'] # noqa
        header: head_region_1
        exiting: tail_region_1
        parent_region: meta_region_0
    branch_region_1:
        type: region
        kind: branch
        contains: ['synth_asign_block_0']
        header: synth_asign_block_0
        exiting: synth_asign_block_0
        parent_region: meta_region_0
    branch_region_2:
        type: region
        kind: branch
        contains: ['python_bytecode_block_2', 'synth_asign_block_1']
        header: python_bytecode_block_2
        exiting: synth_asign_block_1
        parent_region: branch_region_0
    branch_region_3:
        type: region
        kind: branch
        contains: ['python_bytecode_block_3', 'synth_asign_block_2']
        header: python_bytecode_block_3
        exiting: synth_asign_block_2
        parent_region: branch_region_0
    branch_region_4:
        type: region
        kind: branch
        contains: ['python_bytecode_block_4']
        header: python_bytecode_block_4
        exiting: python_bytecode_block_4
        parent_region: tail_region_0
    branch_region_5:
        type: region
        kind: branch
        contains: ['synth_fill_block_0']
        header: synth_fill_block_0
        exiting: synth_fill_block_0
        parent_region: tail_region_0
    head_region_0:
        type: region
        kind: head
        contains: ['python_bytecode_block_0']
        header: python_bytecode_block_0
        exiting: python_bytecode_block_0
        parent_region: meta_region_0
    head_region_1:
        type: region
        kind: head
        contains: ['python_bytecode_block_1']
        header: python_bytecode_block_1
        exiting: python_bytecode_block_1
        parent_region: branch_region_0
    head_region_2:
        type: region
        kind: head
        contains: ['synth_head_block_0']
        header: synth_head_block_0
        exiting: synth_head_block_0
        parent_region: tail_region_0
    python_bytecode_block_0:
        type: python_bytecode
        begin: 0
        end: 4
    python_bytecode_block_1:
        type: python_bytecode
        begin: 4
        end: 8
    python_bytecode_block_2:
        type: python_bytecode
        begin: 8
        end: 12
    python_bytecode_block_3:
        type: python_bytecode
        begin: 12
        end: 14
    python_bytecode_block_4:
        type: python_bytecode
        begin: 14
        end: 18
    python_bytecode_block_5:
        type: python_bytecode
        begin: 18
        end: 20
    synth_asign_block_0:
        type: synth_asign
        variable_assignment: {'__scfg_control_var_0__': 0}
    synth_asign_block_1:
        type: synth_asign
        variable_assignment: {'__scfg_control_var_0__': 1}
    synth_asign_block_2:
        type: synth_asign
        variable_assignment: {'__scfg_control_var_0__': 2}
    synth_fill_block_0:
        type: synth_fill
    synth_head_block_0:
        type: synth_head
        branch_value_table: {0: 'branch_region_4', 2: 'branch_region_4', 1: 'branch_region_5'} # noqa
        variable: __scfg_control_var_0__
    synth_tail_block_0:
        type: synth_tail
    tail_region_0:
        type: region
        kind: tail
        contains: ['branch_region_4', 'branch_region_5', 'head_region_2', 'tail_region_2'] # noqa
        header: head_region_2
        exiting: tail_region_2
        parent_region: meta_region_0
    tail_region_1:
        type: region
        kind: tail
        contains: ['synth_tail_block_0']
        header: synth_tail_block_0
        exiting: synth_tail_block_0
        parent_region: branch_region_0
    tail_region_2:
        type: region
        kind: tail
        contains: ['python_bytecode_block_5']
        header: python_bytecode_block_5
        exiting: python_bytecode_block_5
        parent_region: tail_region_0
edges:
    branch_region_0: ['tail_region_0']
    branch_region_1: ['tail_region_0']
    branch_region_2: ['tail_region_1']
    branch_region_3: ['tail_region_1']
    branch_region_4: ['tail_region_2']
    branch_region_5: ['tail_region_2']
    head_region_0: ['branch_region_0', 'branch_region_1']
    head_region_1: ['branch_region_2', 'branch_region_3']
    head_region_2: ['branch_region_4', 'branch_region_5']
    python_bytecode_block_0: ['branch_region_0', 'branch_region_1']
    python_bytecode_block_1: ['branch_region_2', 'branch_region_3']
    python_bytecode_block_2: ['synth_asign_block_1']
    python_bytecode_block_3: ['synth_asign_block_2']
    python_bytecode_block_4: ['tail_region_2']
    python_bytecode_block_5: []
    synth_asign_block_0: ['tail_region_0']
    synth_asign_block_1: ['tail_region_1']
    synth_asign_block_2: ['tail_region_1']
    synth_fill_block_0: ['tail_region_2']
    synth_head_block_0: ['branch_region_4', 'branch_region_5']
    synth_tail_block_0: ['tail_region_0']
    tail_region_0: []
    tail_region_1: ['tail_region_0']
    tail_region_2: []
backedges:"""


class TestBahmannFigures(SCFGComparator):
    def test_figure_3(self):
        # Figure 3 of the paper

        # fake bytecode just good enough for FlowInfo
        bc = [
            dis.Instruction("OP", 1, None, None, "", 0, None, False),
            dis.Instruction(
                "POP_JUMP_IF_TRUE", 2, None, 12, "", 2, None, False
            ),
            # label 4
            dis.Instruction("OP", 1, None, None, "", 4, None, False),
            dis.Instruction(
                "POP_JUMP_IF_TRUE", 2, None, 12, "", 6, None, False
            ),
            dis.Instruction("OP", 1, None, None, "", 8, None, False),
            dis.Instruction("JUMP_ABSOLUTE", 2, None, 20, "", 10, None, False),
            # label 12
            dis.Instruction("OP", 1, None, None, "", 12, None, False),
            dis.Instruction(
                "POP_JUMP_IF_TRUE", 2, None, 4, "", 14, None, False
            ),
            dis.Instruction("OP", 1, None, None, "", 16, None, False),
            dis.Instruction("JUMP_ABSOLUTE", 2, None, 20, "", 18, None, False),
            # label 20
            dis.Instruction(
                "RETURN_VALUE", 1, None, None, "", 20, None, False
            ),
        ]
        flow = FlowInfo.from_bytecode(bc)
        scfg = flow.build_basicblocks()
        scfg.restructure()

        x, _ = SCFG.from_yaml(fig_3_yaml)
        self.assertSCFGEqual(x, scfg)

    def test_figure_4(self):
        # Figure 4 of the paper

        # fake bytecode just good enough for FlowInfo
        bc = [
            dis.Instruction("OP", 1, None, None, "", 0, None, False),
            dis.Instruction(
                "POP_JUMP_IF_TRUE", 2, None, 14, "", 2, None, False
            ),
            # label 4
            dis.Instruction("OP", 1, None, None, "", 4, None, False),
            dis.Instruction(
                "POP_JUMP_IF_TRUE", 2, None, 12, "", 6, None, False
            ),
            dis.Instruction("OP", 1, None, None, "", 8, None, False),
            dis.Instruction("JUMP_ABSOLUTE", 2, None, 18, "", 10, None, False),
            # label 12
            dis.Instruction("OP", 1, None, None, "", 12, None, False),
            dis.Instruction("OP", 2, None, 4, "", 14, None, False),
            dis.Instruction("JUMP_ABSOLUTE", 2, None, 18, "", 16, None, False),
            # label 18
            dis.Instruction(
                "RETURN_VALUE", 1, None, None, "", 18, None, False
            ),
        ]
        flow = FlowInfo.from_bytecode(bc)
        scfg = flow.build_basicblocks()
        scfg.restructure()

        x, _ = SCFG.from_yaml(fig_4_yaml)
        self.assertSCFGEqual(x, scfg)
