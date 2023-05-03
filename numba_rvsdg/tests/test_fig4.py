# Figure 4 of the paper
import dis
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.rendering.rendering import render_flow, ByteFlowRenderer


expected = """digraph {
	subgraph cluster_python_bytecode_block_0 {
		color=red label=head
		python_bytecode_block_0 [label="python_bytecode_block_0\l  0: OP\l  2: POP_JUMP_IF_TRUE\l" shape=rect]
	}
	python_bytecode_block_2 -> synth_asign_block_1
	python_bytecode_block_3 -> synth_asign_block_2
	subgraph cluster_python_bytecode_block_1 {
		color=green label=branch
		subgraph cluster_python_bytecode_block_1 {
			color=red label=head
			python_bytecode_block_1 [label="python_bytecode_block_1\l  4: OP\l  6: POP_JUMP_IF_TRUE\l" shape=rect]
		}
		subgraph cluster_python_bytecode_block_2 {
			color=green label=branch
			python_bytecode_block_2 [label="python_bytecode_block_2\l  8: OP\l 10: JUMP_ABSOLUTE\l" shape=rect]
			synth_asign_block_1 [label="synth_asign_block_1\lcontrol_var_0 = 1" shape=rect]
		}
		subgraph cluster_python_bytecode_block_3 {
			color=green label=branch
			python_bytecode_block_3 [label="python_bytecode_block_3\l 12: OP\l" shape=rect]
			synth_asign_block_2 [label="synth_asign_block_2\lcontrol_var_0 = 2" shape=rect]
		}
		subgraph cluster_synth_tail_block_0 {
			color=purple label=tail
			synth_tail_block_0 [label="synth_tail_block_0\l" shape=rect]
		}
	}
	python_bytecode_block_1 -> python_bytecode_block_2
	python_bytecode_block_1 -> python_bytecode_block_3
	synth_asign_block_1 -> synth_tail_block_0
	synth_asign_block_2 -> synth_tail_block_0
	subgraph cluster_synth_asign_block_0 {
		color=green label=branch
		synth_asign_block_0 [label="synth_asign_block_0\lcontrol_var_0 = 0" shape=rect]
	}
	subgraph cluster_synth_head_block_0 {
		color=purple label=tail
		subgraph cluster_python_bytecode_block_4 {
			color=green label=branch
			python_bytecode_block_4 [label="python_bytecode_block_4\l 14: OP\l 16: JUMP_ABSOLUTE\l" shape=rect]
		}
		subgraph cluster_python_bytecode_block_5 {
			color=purple label=tail
			python_bytecode_block_5 [label="python_bytecode_block_5\l 18: RETURN_VALUE\l" shape=rect]
		}
		subgraph cluster_synth_fill_block_0 {
			color=green label=branch
			synth_fill_block_0 [label="synth_fill_block_0\l" shape=rect]
		}
		subgraph cluster_synth_head_block_0 {
			color=red label=head
			synth_head_block_0 [label="synth_head_block_0\lvariable: control_var_0\l0=>python_bytecode_block_4\l2=>python_bytecode_block_4\l1=>synth_fill_block_0" shape=rect]
		}
	}
	python_bytecode_block_4 -> python_bytecode_block_5
	synth_fill_block_0 -> python_bytecode_block_5
	synth_head_block_0 -> python_bytecode_block_4
	synth_head_block_0 -> synth_fill_block_0
	python_bytecode_block_0 -> python_bytecode_block_1
	python_bytecode_block_0 -> synth_asign_block_0
	synth_tail_block_0 -> synth_head_block_0
	synth_asign_block_0 -> synth_head_block_0
}"""


def make_flow():
    # fake bytecode just good enough for FlowInfo
    bc = [
        dis.Instruction("OP", 1, None, None, "", 0, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 14, "", 2, None, False),
        # label 4
        dis.Instruction("OP", 1, None, None, "", 4, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 12, "", 6, None, False),
        dis.Instruction("OP", 1, None, None, "", 8, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 18, "", 10, None, False),
        # label 12
        dis.Instruction("OP", 1, None, None, "", 12, None, False),
        dis.Instruction("OP", 2, None, 4, "", 14, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 18, "", 16, None, False),
        # label 18
        dis.Instruction("RETURN_VALUE", 1, None, None, "", 18, None, False),
    ]
    flow = FlowInfo.from_bytecode(bc)
    scfg = flow.build_basicblocks()
    return ByteFlow(bc=bc, scfg=scfg)

def test_fig4():
    flow = make_flow()
    restructured = flow.restructure()
    dot_output = ByteFlowRenderer().render_byteflow(restructured)
    assert expected == str(dot_output).strip()


if __name__ == "__main__":
    render_flow(make_flow())
