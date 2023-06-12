# Figure 3 of the paper
import dis
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.rendering.rendering import render_flow, ByteFlowRenderer


expected = """digraph {
	subgraph cluster_python_bytecode_block_0 {
		color=red label=head
		python_bytecode_block_0 [label="python_bytecode_block_0\l  0: OP\l  2: POP_JUMP_IF_TRUE\l" shape=rect]
	}
	subgraph cluster_synth_asign_block_0 {
		color=green label=branch
		synth_asign_block_0 [label="synth_asign_block_0\lcontrol_var_0 = 0" shape=rect]
	}
	subgraph cluster_synth_asign_block_1 {
		color=green label=branch
		synth_asign_block_1 [label="synth_asign_block_1\lcontrol_var_0 = 1" shape=rect]
	}
	python_bytecode_block_1 -> synth_asign_block_2
	python_bytecode_block_1 -> synth_asign_block_3
	synth_asign_block_2 -> synth_tail_block_0
	synth_asign_block_3 -> synth_tail_block_0
	python_bytecode_block_3 -> synth_asign_block_4
	python_bytecode_block_3 -> synth_asign_block_5
	synth_asign_block_4 -> synth_tail_block_1
	synth_asign_block_5 -> synth_tail_block_1
	synth_exit_latch_block_0 -> synth_head_block_0 [color=grey constraint=0 style=dashed]
	synth_tail_block_0 -> synth_exit_latch_block_0
	synth_tail_block_1 -> synth_exit_latch_block_0
	synth_head_block_0 -> python_bytecode_block_1
	synth_head_block_0 -> python_bytecode_block_3
	synth_exit_latch_block_0 -> synth_exit_block_0
	subgraph cluster_synth_head_block_0 {
		color=purple label=tail
		subgraph cluster_python_bytecode_block_2 {
			color=green label=branch
			python_bytecode_block_2 [label="python_bytecode_block_2\l  8: OP\l 10: JUMP_ABSOLUTE\l" shape=rect]
		}
		subgraph cluster_python_bytecode_block_4 {
			color=green label=branch
			python_bytecode_block_4 [label="python_bytecode_block_4\l 16: OP\l 18: JUMP_ABSOLUTE\l" shape=rect]
		}
		subgraph cluster_python_bytecode_block_5 {
			color=purple label=tail
			python_bytecode_block_5 [label="python_bytecode_block_5\l 20: RETURN_VALUE\l" shape=rect]
		}
		subgraph cluster_synth_head_block_0 {
			color=red label=head
			synth_exit_block_0 [label="synth_exit_block_0\lvariable: control_var_0\l0=>python_bytecode_block_2\l1=>python_bytecode_block_4" shape=rect]
			subgraph cluster_synth_head_block_0 {
				color=blue label=loop
				subgraph cluster_python_bytecode_block_1 {
					color=green label=branch
					subgraph cluster_python_bytecode_block_1 {
						color=red label=head
						python_bytecode_block_1 [label="python_bytecode_block_1\l  4: OP\l  6: POP_JUMP_IF_TRUE\l" shape=rect]
					}
					subgraph cluster_synth_asign_block_2 {
						color=green label=branch
						synth_asign_block_2 [label="synth_asign_block_2\lcontrol_var_0 = 0\lbackedge_var_0 = 1" shape=rect]
					}
					subgraph cluster_synth_asign_block_3 {
						color=green label=branch
						synth_asign_block_3 [label="synth_asign_block_3\lbackedge_var_0 = 0\lcontrol_var_0 = 1" shape=rect]
					}
					subgraph cluster_synth_tail_block_0 {
						color=purple label=tail
						synth_tail_block_0 [label="synth_tail_block_0\l" shape=rect]
					}
				}
				subgraph cluster_python_bytecode_block_3 {
					color=green label=branch
					subgraph cluster_python_bytecode_block_3 {
						color=red label=head
						python_bytecode_block_3 [label="python_bytecode_block_3\l 12: OP\l 14: POP_JUMP_IF_TRUE\l" shape=rect]
					}
					subgraph cluster_synth_asign_block_4 {
						color=green label=branch
						synth_asign_block_4 [label="synth_asign_block_4\lcontrol_var_0 = 1\lbackedge_var_0 = 1" shape=rect]
					}
					subgraph cluster_synth_asign_block_5 {
						color=green label=branch
						synth_asign_block_5 [label="synth_asign_block_5\lbackedge_var_0 = 0\lcontrol_var_0 = 0" shape=rect]
					}
					subgraph cluster_synth_tail_block_1 {
						color=purple label=tail
						synth_tail_block_1 [label="synth_tail_block_1\l" shape=rect]
					}
				}
				subgraph cluster_synth_exit_latch_block_0 {
					color=purple label=tail
					synth_exit_latch_block_0 [label="synth_exit_latch_block_0\lvariable: backedge_var_0\l0=>synth_head_block_0\l1=>synth_exit_block_0" shape=rect]
				}
				subgraph cluster_synth_head_block_0 {
					color=red label=head
					synth_head_block_0 [label="synth_head_block_0\lvariable: control_var_0\l0=>python_bytecode_block_1\l1=>python_bytecode_block_3" shape=rect]
				}
			}
		}
	}
	python_bytecode_block_2 -> python_bytecode_block_5
	python_bytecode_block_4 -> python_bytecode_block_5
	synth_exit_block_0 -> python_bytecode_block_2
	synth_exit_block_0 -> python_bytecode_block_4
	python_bytecode_block_0 -> synth_asign_block_0
	python_bytecode_block_0 -> synth_asign_block_1
	synth_asign_block_0 -> synth_head_block_0
	synth_asign_block_1 -> synth_head_block_0
}"""


def make_flow():
    # fake bytecode just good enough for FlowInfo
    bc = [
        dis.Instruction("OP", 1, None, None, "", 0, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 12, "", 2, None, False),
        # label 4
        dis.Instruction("OP", 1, None, None, "", 4, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 12, "", 6, None, False),
        dis.Instruction("OP", 1, None, None, "", 8, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 20, "", 10, None, False),
        # label 12
        dis.Instruction("OP", 1, None, None, "", 12, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 4, "", 14, None, False),
        dis.Instruction("OP", 1, None, None, "", 16, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 20, "", 18, None, False),
        # label 20
        dis.Instruction("RETURN_VALUE", 1, None, None, "", 20, None, False),
    ]
    flow = FlowInfo.from_bytecode(bc)
    scfg = flow.build_basicblocks()
    return ByteFlow(bc=bc, scfg=scfg)


def test_fig3():
    flow = make_flow()
    restructured = flow.restructure()
    dot_output = ByteFlowRenderer().render_byteflow(restructured)
    assert expected == str(dot_output).strip()


if __name__ == "__main__":
    render_flow(make_flow())
