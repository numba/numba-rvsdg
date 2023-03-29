from collections import defaultdict
from typing import Set, Dict, List

from numba_rvsdg.core.datastructures.labels import (
    Label,
    SyntheticBranch,
    SyntheticHead,
    SyntheticExitingLatch,
    SyntheticExit,
    SyntheticReturn,
    SyntheticTail,
    SynthenticAssignment,
    PythonBytecodeLabel,
    BlockName,
)
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    ControlVariableBlock,
    BranchBlock,
)

from numba_rvsdg.core.utils import _logger


def loop_restructure_helper(scfg: SCFG, loop: Set[BlockName]):
    """Loop Restructuring

    Applies the algorithm LOOP RESTRUCTURING from section 4.1 of Bahmann2015.

    Note that this will modify both the `scfg` and the `loop` in-place.

    Parameters
    ----------
    scfg: SCFG
        The SCFG containing the loop
    loop: List[BlockName]
        The loop (strongly connected components) that is to be restructured

    """

    headers, entries = scfg.find_headers_and_entries(loop)
    exiting_blocks, exit_blocks = scfg.find_exiting_and_exits(loop)
    headers_were_unified = False
    exit_blocks = list(sorted(exit_blocks))

    # If there are multiple headers, insert assignment and control blocks,
    # such that only a single loop header remains.
    if len(headers) > 1:
        headers_were_unified = True
        solo_head_label = SyntheticHead()
        loop_head: BlockName = insert_block_and_control_blocks(scfg, list(sorted(entries)), list(sorted(headers)), block_label=solo_head_label)
        loop.add(loop_head)
    else:
        loop_head: BlockName = next(iter(headers))
    # If there is only a single exiting latch (an exiting block that also has a
    # backedge to the loop header) we can exit early, since the condition for
    # SCFG is fullfilled.
    backedge_blocks = [
        block for block in loop if set(headers).intersection(scfg.out_edges[block])
    ]
    if (
        len(backedge_blocks) == 1
        and len(exiting_blocks) == 1
        and backedge_blocks[0] == next(iter(exiting_blocks))
    ):
        scfg.back_edges[backedge_blocks[0]].append(loop_head)
        return

    doms = _doms(scfg)
    # The synthetic exiting latch and synthetic exit need to be created
    # based on the state of the cfg. If there are multiple exits, we need a
    # SyntheticExit, otherwise we only need a SyntheticExitingLatch

    # Set a flag, this will determine the variable assignment and block
    # insertion later on
    needs_synth_exit = len(exit_blocks) > 1

    # This sets up the various control variables.
    # If there were multiple headers, we must re-use the variable that was used
    # for looping as the exit variable
    if headers_were_unified:
        exit_variable = scfg[loop_head].variable
    else:
        exit_variable = scfg.name_gen.new_var_name()

    exit_value_table = dict(((i, j) for i, j in enumerate(exit_blocks)))
    if needs_synth_exit:
        synth_exit_label = SyntheticExit()
        synth_exit = scfg.add_block("branch", block_label=synth_exit_label, variable=exit_variable, branch_value_table=exit_value_table)

    # Now we setup the lookup tables for the various control variables,
    # depending on the state of the CFG and what is needed
    if needs_synth_exit:
        backedge_value_table = dict(
            (i, j) for i, j in enumerate((loop_head, synth_exit))
        )
    else:
        backedge_value_table = dict(
            (i, j) for i, j in enumerate((loop_head, next(iter(exit_blocks))))
        )
    if headers_were_unified:
        header_value_table = scfg[loop_head].branch_value_table
    else:
        header_value_table = {}

    synth_latch_label = SyntheticExitingLatch()
    # This variable denotes the backedge
    backedge_variable = scfg.name_gen.new_var_name()
    synth_exiting_latch = scfg.add_block("branch", block_label=synth_latch_label, variable=backedge_variable, branch_value_table=backedge_value_table)

    # This does a dictionary reverse lookup, to determine the key for a given
    # value.
    def reverse_lookup(d, value):
        for k, v in d.items():
            if v == value:
                return k
        else:
            return "UNUSED"

    # Now that everything is in place, we can start to insert blocks, depending
    # on what is needed

    # For every block in the loop:
    for _name in sorted(loop):
        # If the block is an exiting block or a backedge block
        if _name in exiting_blocks or _name in backedge_blocks:
            # For each jump_target in the block
            for jt in scfg.out_edges[_name]:
                # If the target is an exit block
                if jt in exit_blocks:
                    # Create a new assignment name and record it
                    synth_assign = SynthenticAssignment()

                    # Setup the table for the variable assignment
                    variable_assignment = {}
                    # Setup the variables in the assignment table to point to
                    # the correct blocks
                    if needs_synth_exit:
                        variable_assignment[exit_variable] = reverse_lookup(
                            exit_value_table, jt
                        )
                    variable_assignment[backedge_variable] = reverse_lookup(
                        backedge_value_table,
                        synth_exit if needs_synth_exit else next(iter(exit_blocks)),
                    )
                    # Insert the assignment to the block map
                    synth_assign = scfg.add_block("control_variable", synth_assign, variable_assignment=variable_assignment)
                    scfg.add_connections(synth_assign, [synth_exiting_latch], [])
                    loop.add(synth_assign)

                    scfg.out_edges[_name][scfg.out_edges[_name].index(jt)] = synth_assign
                # If the target is the loop_head
                elif jt in headers and _name not in doms[jt]:
                    # Create the assignment and record it
                    synth_assign = SynthenticAssignment()
                    # Setup the variables in the assignment table to point to
                    # the correct blocks
                    variable_assignment = {}
                    variable_assignment[backedge_variable] = reverse_lookup(
                        backedge_value_table, loop_head
                    )
                    if needs_synth_exit:
                        variable_assignment[exit_variable] = reverse_lookup(
                            header_value_table, jt
                        )
                    synth_assign = scfg.add_block("control_variable", synth_assign, variable_assignment=variable_assignment)
                    scfg.add_connections(synth_assign, [synth_exiting_latch], [])
                    loop.add(synth_assign)

                    scfg.out_edges[_name][scfg.out_edges[_name].index(jt)] = synth_assign
    loop.add(synth_exiting_latch)

    # Add the back_edge
    scfg.out_edges[synth_exiting_latch].append(loop_head)
    scfg.back_edges[synth_exiting_latch].append(loop_head)

    # If an exit is to be created, we do so too, but only add it to the scfg,
    # since it isn't part of the loop
    if needs_synth_exit:
        scfg.insert_block_between(synth_exit, [synth_exiting_latch], list(exit_blocks))
    else:
        scfg.out_edges[synth_exiting_latch].append(list(exit_blocks)[0])


def restructure_loop(scfg: SCFG):
    """Inplace restructuring of the given graph to extract loops using
    strongly-connected components
    """
    # obtain a List of Sets of Labels, where all labels in each set are strongly
    # connected, i.e. all reachable from one another by traversing the subset
    scc: List[Set[SCFG]] = scfg.compute_scc()
    # loops are defined as strongly connected subsets who have more than a
    # single label and single label loops that point back to to themselves.
    loops: List[Set[SCFG]] = [
        nodes
        for nodes in scc
        if len(nodes) > 1 or next(iter(nodes)) in scfg.out_edges[next(iter(nodes))]
    ]

    _logger.debug(
        "restructure_loop found %d loops in %s", len(loops), scfg.blocks.keys()
    )
    # rotate and extract loop
    for loop in loops:
        loop_restructure_helper(scfg, loop)
        extract_region(scfg, loop, "loop")


def find_head_blocks(scfg: SCFG, begin: BlockName) -> Set[BlockName]:
    head = scfg.find_head()
    head_region_blocks = set()
    current_block = head
    # Start at the head block and traverse the graph linearly until
    # reaching the begin block.
    while True:
        head_region_blocks.add(current_block)
        if current_block == begin:
            break
        else:
            jt = scfg.out_edges[current_block]
            assert len(jt) == 1
            current_block = next(iter(jt))
    return head_region_blocks


def find_branch_regions(scfg: SCFG, begin: BlockName, end: BlockName) -> Set[BlockName]:
    # identify branch regions
    doms = _doms(scfg)
    postdoms = _post_doms(scfg)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    branch_regions = []
    jump_targets = scfg.out_edges[begin]
    for bra_start in jump_targets:
        for jt in jump_targets:
            if jt != bra_start and scfg.is_reachable_dfs(jt, bra_start):
                branch_regions.append(tuple())
                break
        else:
            sub_keys: Set[PythonBytecodeLabel] = set()
            branch_regions.append((bra_start, sub_keys))
            # a node is part of the branch if
            # - the start of the branch is a dominator; and,
            # - the tail of the branch is not a dominator
            for k, kdom in doms.items():
                if bra_start in kdom and end not in kdom:
                    sub_keys.add(k)
    return branch_regions


def _find_branch_regions(scfg: SCFG, begin: BlockName, end: BlockName) -> Set[BlockName]:
    # identify branch regions
    branch_regions = []
    for bra_start in scfg.out_edges[begin]:
        region = []
        region.append(bra_start)
    return branch_regions


def find_tail_blocks(scfg: SCFG, begin: Set[BlockName], head_region_blocks, branch_regions):
    tail_subregion = set((b for b in scfg.blocks.keys()))
    tail_subregion.difference_update(head_region_blocks)
    for reg in branch_regions:
        if not reg:
            continue
        b, sub = reg
        tail_subregion.discard(b)
        for s in sub:
            tail_subregion.discard(s)
    # exclude parents
    tail_subregion.discard(begin)
    return tail_subregion


def extract_region(scfg: SCFG, region_blocks, region_kind):
    headers, entries = scfg.find_headers_and_entries(region_blocks)
    exiting_blocks, exit_blocks = scfg.find_exiting_and_exits(region_blocks)
    assert len(headers) == 1
    assert len(exiting_blocks) == 1
    region_header = next(iter(headers))
    region_exiting = next(iter(exiting_blocks))

    scfg.add_region(region_header, region_exiting, region_kind)


def restructure_branch(scfg: SCFG):
    print("restructure_branch", scfg.blocks)
    doms = _doms(scfg)
    postdoms = _post_doms(scfg)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    regions = [r for r in _iter_branch_regions(scfg, immdoms, postimmdoms)]

    # Early exit when no branching regions are found.
    # TODO: the whole graph should become a linear mono head
    if not regions:
        return

    # Compute initial regions.
    begin, end = regions[0]
    head_region_blocks = find_head_blocks(scfg, begin)
    branch_regions = find_branch_regions(scfg, begin, end)
    tail_region_blocks = find_tail_blocks(
        scfg, begin, head_region_blocks, branch_regions
    )

    # Unify headers of tail subregion if need be.
    headers, entries = scfg.find_headers_and_entries(tail_region_blocks)
    if len(headers) > 1:
        end = SyntheticHead()
        insert_block_and_control_blocks(scfg, entries, headers, end)

    # Recompute regions.
    head_region_blocks = find_head_blocks(scfg, begin)
    branch_regions = find_branch_regions(scfg, begin, end)
    tail_region_blocks = find_tail_blocks(
        scfg, begin, head_region_blocks, branch_regions
    )

    # Branch region processing:
    # Close any open branch regions by inserting a SyntheticTail.
    # Populate any empty branch regions by inserting a SyntheticBranch.
    for region in branch_regions:
        if region:
            bra_start, inner_nodes = region
            if inner_nodes:
                # Insert SyntheticTail
                exiting_blocks, _ = scfg.find_exiting_and_exits(inner_nodes)
                tail_headers, _ = scfg.find_headers_and_entries(tail_region_blocks)
                _, _ = join_tails_and_exits(scfg, exiting_blocks, tail_headers)

        else:
            # Insert SyntheticBranch
            tail_headers, _ = scfg.find_headers_and_entries(tail_region_blocks)
            synthetic_branch_block_label = SyntheticBranch()
            scfg.add_block(block_label=synthetic_branch_block_label)
            scfg.insert_block_between(synthetic_branch_block_label, (begin,), tail_headers)

    # Recompute regions.
    head_region_blocks = find_head_blocks(scfg, begin)
    branch_regions = find_branch_regions(scfg, begin, end)
    tail_region_blocks = find_tail_blocks(
        scfg, begin, head_region_blocks, branch_regions
    )

    # extract subregions
    extract_region(scfg, head_region_blocks, "head")
    for region in branch_regions:
        if region:
            bra_start, inner_nodes = region
            if inner_nodes:
                extract_region(scfg, inner_nodes, "branch")
    extract_region(scfg, tail_region_blocks, "tail")


def _iter_branch_regions(
    scfg: SCFG, immdoms: Dict[BlockName, BlockName], postimmdoms: Dict[BlockName, BlockName]
):
    for begin, node in [i for i in scfg.blocks.items()]:
        if len(scfg.out_edges[begin]) > 1:
            # found branch
            if begin in postimmdoms:
                end = postimmdoms[begin]
                if immdoms[end] == begin:
                    yield begin, end


def _imm_doms(doms: Dict[BlockName, Set[BlockName]]) -> Dict[BlockName, BlockName]:
    idoms = {k: v - {k} for k, v in doms.items()}
    changed = True
    while changed:
        changed = False
        for k, vs in idoms.items():
            nstart = len(vs)
            for v in list(vs):
                vs -= idoms[v]
            if len(vs) < nstart:
                changed = True
    # fix output
    out = {}
    for k, vs in idoms.items():
        if vs:
            [v] = vs
            out[k] = v
    return out


def _doms(scfg: SCFG):
    # compute dom
    entries = set()
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    node: BasicBlock
    for src, node in scfg.blocks.items():
        for dst in scfg.out_edges[src]:
            # check dst is in subgraph
            if dst in scfg.blocks:
                preds_table[dst].add(src)
                succs_table[src].add(dst)

    for k in scfg.blocks:
        if not preds_table[k]:
            entries.add(k)
    return _find_dominators_internal(
        entries, list(scfg.blocks.keys()), preds_table, succs_table
    )


def _post_doms(scfg: SCFG):
    # compute post dom
    entries = set()
    for k in scfg.blocks.keys():
        targets = set(scfg.out_edges[k]) & set(scfg.blocks)
        if not targets:
            entries.add(k)
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    for src in scfg.blocks.keys():
        for dst in scfg.out_edges[src]:
            # check dst is in subgraph
            if dst in scfg.blocks:
                preds_table[src].add(dst)
                succs_table[dst].add(src)

    return _find_dominators_internal(
        entries, list(scfg.blocks.keys()), preds_table, succs_table
    )


def _find_dominators_internal(entries, nodes, preds_table, succs_table):
    # From NUMBA
    # See theoretical description in
    # http://en.wikipedia.org/wiki/Dominator_%28graph_theory%29
    # The algorithm implemented here uses a todo-list as described
    # in http://pages.cs.wisc.edu/~fischer/cs701.f08/finding.loops.html

    # if post:
    #     entries = set(scfg._exit_points)
    #     preds_table = scfg._succs
    #     succs_table = scfg._preds
    # else:
    #     entries = set([scfg._entry_point])
    #     preds_table = scfg._preds
    #     succs_table = scfg._succs

    import functools

    if not entries:
        raise RuntimeError("no entry points: dominator algorithm " "cannot be seeded")

    doms = {}
    for e in entries:
        doms[e] = set([e])

    todo = []
    for n in nodes:
        if n not in entries:
            doms[n] = set(nodes)
            todo.append(n)

    while todo:
        n = todo.pop()
        if n in entries:
            continue
        new_doms = set([n])
        preds = preds_table[n]
        if preds:
            new_doms |= functools.reduce(set.intersection, [doms[p] for p in preds])
        if new_doms != doms[n]:
            assert len(new_doms) < len(doms[n])
            doms[n] = new_doms
            todo.extend(succs_table[n])
    return doms


def insert_block_and_control_blocks(
    scfg: SCFG,
    predecessors: List[BlockName],
    successors: List[BlockName],
    block_label: Label = Label()
) -> BlockName:
    # TODO: needs a diagram and documentaion
    # name of the variable for this branching assignment
    branch_variable = scfg.name_gen.new_var_name()
    # initial value of the assignment
    branch_variable_value = 0
    # store for the mapping from variable value to blockname
    branch_value_table = {}
    # initialize new block, which will hold the branching table
    branch_block_name = scfg.add_block(
        "branch",
        block_label,
        variable=branch_variable,
        branch_value_table=branch_value_table
    )

    control_blocks = {}
    # Replace any arcs from any of predecessors to any of successors with
    # an arc through the to be inserted block instead.

    for pred_name in predecessors:
        pred_outs = scfg.out_edges[pred_name]
        # Need to create synthetic assignments for each arc from a
        # predecessors to a successor and insert it between the predecessor
        # and the newly created block
        for s in successors:
            if s in pred_outs:
                synth_assign = SynthenticAssignment()
                variable_assignment = {}
                variable_assignment[branch_variable] = branch_variable_value

                # add block
                control_block_name = scfg.add_block(
                    "control_variable",
                    synth_assign,
                    variable_assignment=variable_assignment,
                )
                # update branching table
                branch_value_table[branch_variable_value] = s
                # update branching variable
                branch_variable_value += 1
                control_blocks[control_block_name] = pred_name

    scfg.insert_block_between(branch_block_name, predecessors, successors)

    for _synth_assign, _pred in control_blocks.items():
        scfg.insert_block_between(_synth_assign, [_pred], [branch_block_name])

    return branch_block_name


def join_returns(scfg: SCFG):
    """Close the CFG.

    A closed CFG is a CFG with a unique entry and exit node that have no
    predescessors and no successors respectively.
    """
    # for all nodes that contain a return
    return_nodes = [node for node in scfg.blocks.keys() if scfg.is_exiting(node)]
    # close if more than one is found
    if len(return_nodes) > 1:
        return_solo_label = SyntheticReturn()
        new_block = scfg.add_block(block_label=return_solo_label)
        scfg.insert_block_between(new_block, return_nodes, [])


def join_tails_and_exits(scfg: SCFG, tails: Set[BlockName], exits: Set[BlockName]):
    if len(tails) == 1 and len(exits) == 1:
        # no-op
        solo_tail_name = next(iter(tails))
        solo_exit_name = next(iter(exits))
        return solo_tail_name, solo_exit_name

    if len(tails) == 1 and len(exits) == 2:
        # join only exits
        solo_tail_name = next(iter(tails))
        solo_exit_label = SyntheticExit()
        solo_exit_name = scfg.add_block(block_label=solo_exit_label)
        scfg.insert_block_between(solo_exit_name, tails, exits)
        return solo_tail_name, solo_exit_name

    if len(tails) >= 2 and len(exits) == 1:
        # join only tails
        solo_tail_label = SyntheticTail()
        solo_exit_name = next(iter(exits))
        solo_tail_name = scfg.add_block(block_label=solo_tail_label)
        scfg.insert_block_between(solo_tail_name, tails, exits)
        return solo_tail_name, solo_exit_name

    if len(tails) >= 2 and len(exits) >= 2:
        # join both tails and exits
        solo_tail_label = SyntheticTail()
        solo_exit_label = SyntheticExit()

        solo_tail_name = scfg.add_block(block_label=solo_tail_label)
        scfg.insert_block_between(solo_tail_name, tails, exits)

        solo_exit_name = scfg.add_block(block_label=solo_exit_label)
        scfg.insert_block_between(solo_exit_name, set((solo_tail_name,)), exits)
        return solo_tail_name, solo_exit_name
