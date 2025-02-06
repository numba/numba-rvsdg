from collections import defaultdict
from typing import Set, Dict, List, Tuple, Optional, Mapping, Iterator

from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.core.datastructures.basic_block import (
    BasicBlock,
    SyntheticAssignment,
    SyntheticBranch,
    SyntheticExitingLatch,
    SyntheticExitBranch,
    RegionBlock,
)
from numba_scfg.core.datastructures import block_names

from numba_scfg.core.utils import _logger


def loop_restructure_helper(scfg: SCFG, loop: Set[str]) -> None:
    """Loop Restructuring

    Applies the algorithm LOOP RESTRUCTURING from section 4.1 of Bahmann2015.

    Note that this will modify both the `scfg` and the `loop` in-place.

    Parameters
    ----------
    scfg: SCFG
        The SCFG containing the loop
    loop: Set[str]
        The loop (strongly connected components) that is to be restructured

    """

    headers, entries = scfg.find_headers_and_entries(loop)
    exiting_blocks, exit_blocks = scfg.find_exiting_and_exits(loop)
    # assert len(entries) == 1
    headers_were_unified = False

    # If there are multiple headers, insert assignment and control blocks,
    # such that only a single loop header remains.
    loop_head = None
    if len(headers) > 1:
        headers_were_unified = True
        solo_head_name = scfg.name_gen.new_block_name(block_names.SYNTH_HEAD)
        scfg.insert_block_and_control_blocks(solo_head_name, entries, headers)
        loop.add(solo_head_name)
        loop_head = solo_head_name
    else:
        loop_head = next(iter(headers))
    # If there is only a single exiting latch (an exiting block that also has a
    # backedge to the loop header) we can exit early, since the condition for
    # SCFG is fullfilled.
    backedge_blocks = [
        block
        for block in loop
        if set(headers).intersection(scfg[block].jump_targets)
    ]
    if (
        len(backedge_blocks) == 1
        and len(exiting_blocks) == 1
        and backedge_blocks[0] == next(iter(exiting_blocks))
    ):
        scfg.add_block(
            scfg.graph.pop(backedge_blocks[0]).declare_backedge(loop_head)
        )
        return

    # The synthetic exiting latch and synthetic exit need to be created
    # based on the state of the cfg. If there are multiple exits, we need a
    # SyntheticExit, otherwise we only need a SyntheticExitingLatch
    synth_exiting_latch = scfg.name_gen.new_block_name(
        block_names.SYNTH_EXIT_LATCH
    )
    # Set a flag, this will determine the variable assignment and block
    # insertion later on
    needs_synth_exit = len(exit_blocks) > 1
    if needs_synth_exit:
        synth_exit = scfg.name_gen.new_block_name(block_names.SYNTH_EXIT)

    # This sets up the various control variables.
    # If there were multiple headers, we must re-use the variable that was used
    # for looping as the exit variable
    if headers_were_unified:
        bb_branch = scfg[solo_head_name]
        assert isinstance(bb_branch, SyntheticBranch)
        exit_variable = bb_branch.variable
    else:
        exit_variable = scfg.name_gen.new_var_name("exit")
    # This variable denotes the backedge
    backedge_variable = scfg.name_gen.new_var_name("backedge")
    # Now we setup the lookup tables for the various control variables,
    # depending on the state of the CFG and what is needed
    exit_value_table = {i: j for i, j in enumerate(exit_blocks)}
    if needs_synth_exit:
        backedge_value_table = {
            i: j for i, j in enumerate((loop_head, synth_exit))
        }
    else:
        backedge_value_table = {
            i: j for i, j in enumerate((loop_head, next(iter(exit_blocks))))
        }
    if headers_were_unified:
        bb_branch = scfg[solo_head_name]
        assert isinstance(bb_branch, SyntheticBranch)
        header_value_table = bb_branch.branch_value_table
    else:
        header_value_table = {}

    # This does a dictionary reverse lookup, to determine the key for a given
    # value.
    def reverse_lookup(d: Mapping[int, str], value: str) -> int:
        for k, v in d.items():
            if v == value:
                return k
        else:
            return -1

    # Now that everything is in place, we can start to insert blocks, depending
    # on what is needed
    # All new blocks are recorded for later insertion into the loop set
    new_blocks = set()
    doms = _doms(scfg)
    # The loop is a set of blocks, but order matters for reproducibility when
    # assigning synth blocks and so we iterate over the sorted blocks since
    # sets in Python are unorderd.
    for name in sorted(loop):
        # If the block is an exiting block or a backedge block
        if name in exiting_blocks or name in backedge_blocks:
            # Copy the jump targets, these will be modified
            new_jt = list(scfg[name].jump_targets)
            # For each jump_target in the blockj
            for jt in scfg[name].jump_targets:
                # If the target is an exit block
                if jt in exit_blocks:
                    # Create a new assignment name and record it
                    synth_assign = scfg.name_gen.new_block_name(
                        block_names.SYNTH_ASSIGN
                    )
                    new_blocks.add(synth_assign)
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
                        (
                            synth_exit
                            if needs_synth_exit
                            else next(iter(exit_blocks))
                        ),
                    )
                    # Create the actual control variable block
                    synth_assign_block = SyntheticAssignment(
                        name=synth_assign,
                        _jump_targets=(synth_exiting_latch,),
                        backedges=(),
                        variable_assignment=variable_assignment,
                    )
                    # Insert the assignment to the scfg
                    scfg.add_block(synth_assign_block)
                    # Insert the new block into the new jump_targets making
                    # sure, that it replaces the correct jump_target, order
                    # matters in this case.
                    new_jt[new_jt.index(jt)] = synth_assign
                # If the target is the loop_head
                elif jt in headers and (name not in doms[jt] or name == jt):
                    # Create the assignment and record it
                    synth_assign = scfg.name_gen.new_block_name(
                        block_names.SYNTH_ASSIGN
                    )
                    new_blocks.add(synth_assign)
                    # Setup the variables in the assignment table to point to
                    # the correct blocks
                    variable_assignment = {}
                    variable_assignment[backedge_variable] = reverse_lookup(
                        backedge_value_table, loop_head
                    )
                    if needs_synth_exit or headers_were_unified:
                        variable_assignment[exit_variable] = reverse_lookup(
                            header_value_table, jt
                        )
                    # Update the backedge block - remove any existing backedges
                    # that point to the headers, no need to add a backedge,
                    # since it will be contained in the SyntheticExitingLatch
                    # later on.
                    block = scfg.graph.pop(name)
                    jts = list(block.jump_targets)
                    for h in headers:
                        if h in jts:
                            jts.remove(h)
                    scfg.add_block(
                        block.replace_jump_targets(jump_targets=tuple(jts))
                    )
                    # Setup the assignment block and initialize it with the
                    # correct jump_targets and variable assignment.
                    synth_assign_block = SyntheticAssignment(
                        name=synth_assign,
                        _jump_targets=(synth_exiting_latch,),
                        backedges=(),
                        variable_assignment=variable_assignment,
                    )
                    # Add the new block to the SCFG
                    scfg.add_block(synth_assign_block)
                    # Update the jump targets again, order matters
                    new_jt[new_jt.index(jt)] = synth_assign
            # finally, replace the jump_targets for this block with the new
            # ones
            scfg.add_block(
                scfg.graph.pop(name).replace_jump_targets(
                    jump_targets=tuple(new_jt)
                )
            )
    # Add any new blocks to the loop.
    loop.update(new_blocks)

    # Insert the exiting latch, add it to the loop and to the graph.
    synth_exiting_latch_block = SyntheticExitingLatch(
        name=synth_exiting_latch,
        _jump_targets=(
            synth_exit if needs_synth_exit else next(iter(exit_blocks)),
            loop_head,
        ),
        backedges=(loop_head,),
        variable=backedge_variable,
        branch_value_table=backedge_value_table,
    )
    loop.add(synth_exiting_latch)
    scfg.add_block(synth_exiting_latch_block)
    # If an exit is to be created, we do so too, but only add it to the scfg,
    # since it isn't part of the loop
    if needs_synth_exit:
        synth_exit_block = SyntheticExitBranch(
            name=synth_exit,
            _jump_targets=tuple(exit_blocks),
            backedges=(),
            variable=exit_variable,
            branch_value_table=exit_value_table,
        )
        scfg.add_block(synth_exit_block)


def restructure_loop(parent_region: RegionBlock) -> None:
    """Inplace restructuring of the given graph to extract loops using
    strongly-connected components
    """
    assert parent_region.subregion is not None
    scfg = parent_region.subregion
    # obtain a List of Sets of names, where all names in each set are strongly
    # connected, i.e. all reachable from one another by traversing the subset
    scc: List[Set[str]] = scfg.compute_scc()
    # loops are defined as strongly connected subsets who have more than a
    # single name and single name loops that point back to to themselves.
    loops: List[Set[str]] = [
        nodes
        for nodes in scc
        if len(nodes) > 1
        or next(iter(nodes)) in scfg[next(iter(nodes))].jump_targets
    ]

    _logger.debug(
        "restructure_loop found %d loops in %s", len(loops), scfg.graph.keys()
    )
    # rotate and extract loop
    for loop in loops:
        loop_restructure_helper(scfg, loop)
        extract_region(scfg, loop, "loop", parent_region)


def find_head_blocks(scfg: SCFG, begin: str) -> Set[str]:
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
            jt = scfg.graph[current_block].jump_targets
            assert len(jt) == 1
            current_block = next(iter(jt))
    return head_region_blocks


def find_branch_regions(
    scfg: SCFG, begin: str, end: str
) -> List[Optional[Tuple[str, Set[str]]]]:
    # identify branch regions
    doms = _doms(scfg)
    branch_regions: List[Optional[Tuple[str, Set[str]]]] = []
    jump_targets = scfg.graph[begin].jump_targets
    for bra_start in jump_targets:
        for jt in jump_targets:
            if jt != bra_start and scfg.is_reachable_dfs(jt, bra_start):
                # placeholder for empty branch region
                branch_regions.append(None)
                break
        else:
            sub_keys: Set[str] = set()
            branch_regions.append((bra_start, sub_keys))
            # a node is part of the branch if
            # - the start of the branch is a dominator; and,
            # - the tail of the branch is not a dominator
            for k, kdom in doms.items():
                if bra_start in kdom and end not in kdom:
                    sub_keys.add(k)
    return branch_regions


def find_tail_blocks(
    scfg: SCFG,
    begin: str,
    head_region_blocks: Set[str],
    branch_regions: List[Optional[Tuple[str, Set[str]]]],
) -> Set[str]:
    tail_subregion = {b for b in scfg.graph.keys()}
    tail_subregion.difference_update(head_region_blocks)
    for reg in branch_regions:
        if reg is None:
            # empty branch region
            continue
        b, sub = reg
        tail_subregion.discard(b)
        for s in sub:
            tail_subregion.discard(s)
    # exclude parents
    tail_subregion.discard(begin)
    return tail_subregion


def update_exiting(
    region_block: RegionBlock, new_region_header: str, new_region_name: str
) -> RegionBlock:
    # Recursively updates the exiting blocks of a regionblock
    region_exiting = region_block.exiting
    assert region_block.subregion is not None
    region_exiting_block: BasicBlock = region_block.subregion.graph.pop(
        region_exiting
    )
    jt = list(region_exiting_block._jump_targets)
    for idx, s in enumerate(jt):
        if s == new_region_header:
            jt[idx] = new_region_name
    region_exiting_block = region_exiting_block.replace_jump_targets(
        jump_targets=tuple(jt)
    )
    be = list(region_exiting_block.backedges)
    for idx, s in enumerate(be):
        if s == new_region_header:
            be[idx] = new_region_name
    region_exiting_block = region_exiting_block.replace_backedges(
        backedges=tuple(be)
    )
    if isinstance(region_exiting_block, RegionBlock):
        region_exiting_block = update_exiting(
            region_exiting_block, new_region_header, new_region_name
        )
    region_block.subregion.add_block(region_exiting_block)
    return region_block


def extract_region(
    scfg: SCFG,
    region_blocks: Set[str],
    region_kind: str,
    parent_region: RegionBlock,
) -> None:
    headers, entries = scfg.find_headers_and_entries(region_blocks)
    exiting_blocks, exit_blocks = scfg.find_exiting_and_exits(region_blocks)
    assert len(headers) == 1
    assert len(exiting_blocks) == 1
    region_header = next(iter(headers))
    region_exiting = next(iter(exiting_blocks))

    # Generate a new region name
    region_name = scfg.name_gen.new_region_name(region_kind)
    # Create the subregion, make sure that blocks are inserted in a predictable
    # order by sorting the set of region blocks.
    head_subgraph = SCFG(
        {name: scfg.graph[name] for name in sorted(region_blocks)},
        name_gen=scfg.name_gen,
    )

    # For all entries, replace the header as a jump target
    # with the newly created region as a jump target.
    for name in entries:
        # Case in which entry is outside the given sub-graph
        if name not in scfg.graph.keys():
            # If it's actually outside the graph, a check to see
            # if it's a valid assumption, is that the region that
            # the SCFG represents should not be the meta region.
            assert scfg.region.kind != "meta"
            continue
        entry = scfg.graph.pop(name)
        jt = list(entry._jump_targets)
        for idx, s in enumerate(jt):
            if s == region_header:
                jt[idx] = region_name
        entry = entry.replace_jump_targets(jump_targets=tuple(jt))
        be = list(entry.backedges)
        for idx, s in enumerate(be):
            if s == region_header:
                be[idx] = region_name
        entry = entry.replace_backedges(backedges=tuple(be))
        # If the entry itself is a region, update it's
        # exiting blocks too, recursively
        if isinstance(entry, RegionBlock):
            entry = update_exiting(entry, region_header, region_name)
        scfg.add_block(entry)

    region = RegionBlock(
        name=region_name,
        _jump_targets=scfg[region_exiting].jump_targets,
        backedges=(),
        kind=region_kind,
        header=region_header,
        subregion=head_subgraph,
        exiting=region_exiting,
        parent_region=parent_region,
    )
    scfg.remove_blocks(region_blocks)
    scfg.graph[region_name] = region

    # Set the parent region of the newly generated regional
    # graph as the current region.
    object.__setattr__(region.subregion, "region", region)
    # If the region is a header of the parent region replace
    # it accordingly.
    if region_header == parent_region.header:
        parent_region.replace_header(region_name)
    # If the region is a header of the parent region replace
    # it accordingly.
    if region_exiting == parent_region.exiting:
        parent_region.replace_exiting(region_name)
    # For every region block inside the newly created region,
    # update the parent region
    assert region.subregion is not None
    for k, v in region.subregion.graph.items():
        if isinstance(v, RegionBlock):
            object.__setattr__(v, "parent_region", region)


def restructure_branch(parent_region: RegionBlock) -> None:
    assert parent_region.subregion is not None
    scfg: SCFG = parent_region.subregion
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
        end = scfg.name_gen.new_block_name(block_names.SYNTH_HEAD)
        scfg.insert_block_and_control_blocks(end, entries, headers)

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
                tail_headers, _ = scfg.find_headers_and_entries(
                    tail_region_blocks
                )
                _, _ = scfg.join_tails_and_exits(exiting_blocks, tail_headers)

        else:
            # Insert SyntheticFill, a placeholder for an empty branch region
            tail_headers, _ = scfg.find_headers_and_entries(tail_region_blocks)
            synthetic_branch_block_name = scfg.name_gen.new_block_name(
                block_names.SYNTH_FILL
            )
            scfg.insert_SyntheticFill(
                synthetic_branch_block_name, [begin], tail_headers
            )

    # Recompute regions.
    head_region_blocks = find_head_blocks(scfg, begin)
    branch_regions = find_branch_regions(scfg, begin, end)
    tail_region_blocks = find_tail_blocks(
        scfg, begin, head_region_blocks, branch_regions
    )

    # extract subregions
    extract_region(scfg, head_region_blocks, "head", parent_region)
    for region in branch_regions:
        if region:
            bra_start, inner_nodes = region
            if inner_nodes:
                extract_region(scfg, inner_nodes, "branch", parent_region)
    extract_region(scfg, tail_region_blocks, "tail", parent_region)


def _iter_branch_regions(
    scfg: SCFG, immdoms: Dict[str, str], postimmdoms: Dict[str, str]
) -> Iterator[Tuple[str, str]]:
    for begin, node in scfg.concealed_region_view.items():
        if len(node.jump_targets) > 1:
            # found branch
            if begin in postimmdoms:
                end = postimmdoms[begin]
                if immdoms[end] == begin:
                    yield begin, end


def _imm_doms(doms: Dict[str, Set[str]]) -> Dict[str, str]:
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


def _doms(scfg: SCFG) -> Dict[str, Set[str]]:
    # compute dom
    entries = set()
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    node: BasicBlock
    for src, node in scfg.graph.items():
        for dst in node.jump_targets:
            # check dst is in subgraph
            if dst in scfg.graph:
                preds_table[dst].add(src)
                succs_table[src].add(dst)

    for k in scfg.graph:
        if not preds_table[k]:
            entries.add(k)
    return _find_dominators_internal(
        entries, list(scfg.graph.keys()), preds_table, succs_table
    )


def _post_doms(scfg: SCFG) -> Dict[str, Set[str]]:
    # compute post dom
    entries = set()
    for k, v in scfg.graph.items():
        targets = set(v.jump_targets) & set(scfg.graph)
        if not targets:
            entries.add(k)
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    node: BasicBlock
    for src, node in scfg.graph.items():
        for dst in node.jump_targets:
            # check dst is in subgraph
            if dst in scfg.graph:
                preds_table[src].add(dst)
                succs_table[dst].add(src)

    return _find_dominators_internal(
        entries, list(scfg.graph.keys()), preds_table, succs_table
    )


def _find_dominators_internal(
    entries: Set[str],
    nodes: List[str],
    preds_table: Dict[str, Set[str]],
    succs_table: Dict[str, Set[str]],
) -> Dict[str, Set[str]]:
    # From NUMBA
    # See theoretical description in
    # http://en.wikipedia.org/wiki/Dominator_%28graph_theory%29
    # The algorithm implemented here uses a todo-list as described
    # in http://pages.cs.wisc.edu/~fischer/cs701.f08/finding.loops.html

    # if post:
    #     entries = set(self._exit_points)
    #     preds_table = self._succs
    #     succs_table = self._preds
    # else:
    #     entries = set([self._entry_point])
    #     preds_table = self._preds
    #     succs_table = self._succs

    import functools

    if not entries:
        raise RuntimeError(
            "no entry points: dominator algorithm " "cannot be seeded"
        )

    doms = {}
    for e in entries:
        doms[e] = {e}

    todo = []
    for n in nodes:
        if n not in entries:
            doms[n] = set(nodes)
            todo.append(n)

    while todo:
        n = todo.pop()
        if n in entries:
            continue
        new_doms = {n}
        preds = preds_table[n]
        if preds:
            new_doms |= functools.reduce(
                set.intersection, [doms[p] for p in preds]  # type: ignore
            )
        if new_doms != doms[n]:
            assert len(new_doms) < len(doms[n])
            doms[n] = new_doms
            todo.extend(succs_table[n])
    return doms
