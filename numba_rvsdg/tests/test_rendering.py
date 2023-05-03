
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.rendering.rendering import ByteFlowRenderer


def test_simple():
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
    original_scfg, block_dict = SCFG.from_yaml(original)
    restructured = original_scfg.restructure()
    dot = ByteFlowRenderer().render_scfg(restructured)
    expected= """digraph {
	subgraph cluster_basic_block_0 {
		color=red label=head
		basic_block_0 [label="basic_block_0\l" shape=rect]
	}
	subgraph cluster_synth_asign_block_0 {
		color=green label=branch
		synth_asign_block_0 [label="synth_asign_block_0\lcontrol_var_0 = 0" shape=rect]
	}
	subgraph cluster_synth_asign_block_1 {
		color=green label=branch
		synth_asign_block_1 [label="synth_asign_block_1\lcontrol_var_0 = 1" shape=rect]
	}
	basic_block_1 -> basic_block_3
	basic_block_3 -> synth_asign_block_2
	basic_block_3 -> synth_asign_block_3
	synth_asign_block_2 -> synth_tail_block_0
	synth_asign_block_3 -> synth_tail_block_0
	basic_block_2 -> basic_block_4
	basic_block_4 -> synth_asign_block_4
	synth_exit_latch_block_0 -> synth_head_block_0 [color=grey constraint=0 style=dashed]
	synth_tail_block_0 -> synth_exit_latch_block_0
	synth_asign_block_4 -> synth_exit_latch_block_0
	synth_head_block_0 -> basic_block_1
	synth_head_block_0 -> basic_block_2
	subgraph cluster_synth_head_block_0 {
		color=purple label=tail
		basic_block_5 [label="basic_block_5\l" shape=rect]
		subgraph cluster_synth_head_block_0 {
			color=blue label=loop
			subgraph cluster_basic_block_1 {
				color=green label=branch
				subgraph cluster_basic_block_1 {
					color=red label=head
					basic_block_1 [label="basic_block_1\l" shape=rect]
					basic_block_3 [label="basic_block_3\l" shape=rect]
				}
				subgraph cluster_synth_asign_block_2 {
					color=green label=branch
					synth_asign_block_2 [label="synth_asign_block_2\lbackedge_var_0 = 0" shape=rect]
				}
				subgraph cluster_synth_asign_block_3 {
					color=green label=branch
					synth_asign_block_3 [label="synth_asign_block_3\lbackedge_var_0 = 1" shape=rect]
				}
				subgraph cluster_synth_tail_block_0 {
					color=purple label=tail
					synth_tail_block_0 [label="synth_tail_block_0\l" shape=rect]
				}
			}
			subgraph cluster_basic_block_2 {
				color=green label=branch
				basic_block_2 [label="basic_block_2\l" shape=rect]
				basic_block_4 [label="basic_block_4\l" shape=rect]
				synth_asign_block_4 [label="synth_asign_block_4\lbackedge_var_0 = 0" shape=rect]
			}
			subgraph cluster_synth_exit_latch_block_0 {
				color=purple label=tail
				synth_exit_latch_block_0 [label="synth_exit_latch_block_0\lvariable: backedge_var_0\l0=>synth_head_block_0\l1=>basic_block_5" shape=rect]
			}
			subgraph cluster_synth_head_block_0 {
				color=red label=head
				synth_head_block_0 [label="synth_head_block_0\lvariable: control_var_0\l0=>basic_block_1\l1=>basic_block_2" shape=rect]
			}
		}
	}
	synth_exit_latch_block_0 -> basic_block_5
	basic_block_0 -> synth_asign_block_0
	basic_block_0 -> synth_asign_block_1
	synth_asign_block_0 -> synth_head_block_0
	synth_asign_block_1 -> synth_head_block_0
}"""
    assert expected == str(dot).strip()
