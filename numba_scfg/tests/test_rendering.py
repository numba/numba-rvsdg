# This file conatins expected dot output, which uses tabs for indentation.
# Flake8 will fail, so we just ignore the whole file.
# flake8: noqa
# Also ignore types, since we don't type annotate tests.
# mypy: ignore-errors

from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.rendering.rendering import SCFGRenderer

expected_original = r"""digraph {
	0 [label="0\n
jump targets: ('1', '2')" shape=rect style=rounded]
	1 [label="1\n
jump targets: ('3',)" shape=rect style=rounded]
	2 [label="2\n
jump targets: ('4',)" shape=rect style=rounded]
	3 [label="3\n
jump targets: ('2', '5')" shape=rect style=rounded]
	4 [label="4\n
jump targets: ('1',)" shape=rect style=rounded]
	5 [label="5\n" shape=rect style=rounded]
	0 -> 1
	0 -> 2
	1 -> 3
	2 -> 4
	3 -> 2
	3 -> 5
	4 -> 1
}"""

expected_restructured = r"""digraph {
	subgraph cluster_head_region_0 {
		color="#DC267F" label="head_region_0\n
jump targets: ('branch_region_0', 'branch_region_1')" shape=rect style=rounded
		0 [label="0\n
jump targets: ('branch_region_0', 'branch_region_1')" shape=rect style=rounded]
	}
	subgraph cluster_branch_region_0 {
		color="#FFB000" label="branch_region_0\n
jump targets: ('tail_region_0',)" shape=rect style=rounded
		synth_asign_block_0 [label="synth_asign_block_0\n\l__scfg_control_var_0__ = 0\l
jump targets: ('tail_region_0',)" shape=rect style=rounded]
	}
	subgraph cluster_branch_region_1 {
		color="#FFB000" label="branch_region_1\n
jump targets: ('tail_region_0',)" shape=rect style=rounded
		synth_asign_block_1 [label="synth_asign_block_1\n\l__scfg_control_var_0__ = 1\l
jump targets: ('tail_region_0',)" shape=rect style=rounded]
	}
	subgraph cluster_tail_region_0 {
		color="#785EF0" label="tail_region_0\n" shape=rect style=rounded
		5 [label="5\n" shape=rect style=rounded]
		subgraph cluster_loop_region_0 {
			color="#648FFF" label="loop_region_0\n
jump targets: ('5',)" shape=rect style=rounded
			subgraph cluster_head_region_1 {
				color="#DC267F" label="head_region_1\n
jump targets: ('branch_region_2', 'branch_region_3')" shape=rect style=rounded
				synth_head_block_0 [label="synth_head_block_0\n\lvariable: __scfg_control_var_0__\l0 → branch_region_2\l1 → branch_region_3\l
jump targets: ('branch_region_2', 'branch_region_3')" shape=rect style=rounded]
			}
			subgraph cluster_branch_region_2 {
				color="#FFB000" label="branch_region_2\n
jump targets: ('tail_region_1',)" shape=rect style=rounded
				subgraph cluster_head_region_2 {
					color="#DC267F" label="head_region_2\n
jump targets: ('branch_region_4', 'branch_region_5')" shape=rect style=rounded
					1 [label="1\n
jump targets: ('3',)" shape=rect style=rounded]
					3 [label="3\n
jump targets: ('branch_region_4', 'branch_region_5')" shape=rect style=rounded]
				}
				subgraph cluster_branch_region_4 {
					color="#FFB000" label="branch_region_4\n
jump targets: ('tail_region_2',)" shape=rect style=rounded
					synth_asign_block_2 [label="synth_asign_block_2\n\l__scfg_backedge_var_0__ = 0\l__scfg_control_var_0__ = 1\l
jump targets: ('tail_region_2',)" shape=rect style=rounded]
				}
				subgraph cluster_branch_region_5 {
					color="#FFB000" label="branch_region_5\n
jump targets: ('tail_region_2',)" shape=rect style=rounded
					synth_asign_block_3 [label="synth_asign_block_3\n\l__scfg_backedge_var_0__ = 1\l
jump targets: ('tail_region_2',)" shape=rect style=rounded]
				}
				subgraph cluster_tail_region_2 {
					color="#785EF0" label="tail_region_2\n
jump targets: ('tail_region_1',)" shape=rect style=rounded
					synth_tail_block_0 [label="synth_tail_block_0\n
jump targets: ('tail_region_1',)" shape=rect style=rounded]
				}
			}
			subgraph cluster_branch_region_3 {
				color="#FFB000" label="branch_region_3\n
jump targets: ('tail_region_1',)" shape=rect style=rounded
				2 [label="2\n
jump targets: ('4',)" shape=rect style=rounded]
				4 [label="4\n
jump targets: ('synth_asign_block_4',)" shape=rect style=rounded]
				synth_asign_block_4 [label="synth_asign_block_4\n\l__scfg_backedge_var_0__ = 0\l__scfg_control_var_0__ = 0\l
jump targets: ('tail_region_1',)" shape=rect style=rounded]
			}
			subgraph cluster_tail_region_1 {
				color="#785EF0" label="tail_region_1\n
jump targets: ('5',)" shape=rect style=rounded
				synth_exit_latch_block_0 [label="synth_exit_latch_block_0\n\lvariable: __scfg_backedge_var_0__\l1 → 5\l0 → head_region_1\l
jump targets: ('5',)
back edges: ('head_region_1',)" shape=rect style=rounded]
			}
		}
	}
	0 -> synth_asign_block_0
	0 -> synth_asign_block_1
	synth_asign_block_0 -> synth_head_block_0
	synth_asign_block_1 -> synth_head_block_0
	synth_head_block_0 -> 1
	synth_head_block_0 -> 2
	1 -> 3
	3 -> synth_asign_block_2
	3 -> synth_asign_block_3
	synth_asign_block_2 -> synth_tail_block_0
	synth_asign_block_3 -> synth_tail_block_0
	synth_tail_block_0 -> synth_exit_latch_block_0
	2 -> 4
	4 -> synth_asign_block_4
	synth_asign_block_4 -> synth_exit_latch_block_0
	synth_exit_latch_block_0 -> 5
	synth_exit_latch_block_0 -> synth_head_block_0 [color=grey constraint=0 style=dashed]
}"""


def test_simple():
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
    scfg, _ = SCFG.from_yaml(original)

    dot_original = str(SCFGRenderer(scfg).render_scfg()).strip()
    assert expected_original == dot_original

    scfg.restructure()

    dot_restructured = str(SCFGRenderer(scfg).render_scfg()).strip()
    assert expected_restructured == dot_restructured
