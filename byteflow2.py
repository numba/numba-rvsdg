import dis
from copy import deepcopy
from collections import deque, ChainMap, defaultdict
from typing import Optional, Set, Tuple, Dict, List, Sequence
from pprint import pprint
from dataclasses import dataclass, field, replace
import logging

_logger = logging.getLogger(__name__)


class _LogWrap:
    def __init__(self, fn):
        self._fn = fn

    def __str__(self):
        return self._fn()


@dataclass(frozen=True, order=True)
class Label:
    ...


@dataclass(frozen=True, order=True)
class BCLabel(Label):
    offset: int


@dataclass(frozen=True, order=True)
class ControlLabel(Label):
    index: int


class ControlLabelGenerator():

    def __init__(self, index=0):
        self.index = index

    def new_index(self):
        ret = self.index
        self.index += 1
        return ret


clg = ControlLabelGenerator()


@dataclass(frozen=True)
class BasicBlock:
    begin: Label
    """The starting bytecode offset.
    """

    end: Label
    """The bytecode offset immediate after the last bytecode of the block.
    """

    fallthrough: bool
    """Set to True when the block has no terminator. The control should just
    fallthrough to the next block.
    """

    jump_targets: Set[Label]
    """The destination block offsets."""

    backedges: Set[Label]
    """Backedges offsets"""

    def is_exiting(self) -> bool:
        return not self.jump_targets

    def get_instructions(
        self, bcmap: Dict[Label, dis.Instruction]
    ) -> List[dis.Instruction]:
        begin = self.begin
        end = self.end
        it = begin
        out = []
        while it < end:
            out.append(bcmap[it])
            # increment
            it = _next_inst_offset(it)
        return out

    def replace_backedge(self, loop_head: Label) -> "BasicBlock":
        if loop_head in self.jump_targets:
            assert not self.backedges
            # remove backede from jump_targets and add to backedges
            jt = set(self.jump_targets)
            jt.remove(loop_head)
            return replace(self, jump_targets=tuple(jt), backedges=(loop_head,))
        return self

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        fallthrough = len(jump_targets) == 1
        return replace(self, fallthrough=fallthrough, jump_targets=jump_targets)


@dataclass(frozen=True)
class RegionBlock(BasicBlock):
    kind: str
    headers: Dict[Label, BasicBlock]
    """The header of the region"""
    subregion: "BlockMap"
    """The subgraph excluding the headers
    """
    exit: Label
    """The exit node.
    """

    def get_full_graph(self):
        graph = ChainMap(self.subregion.graph, self.headers)
        return graph


_cond_jump = {
    "FOR_ITER",
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
}
_uncond_jump = {"JUMP_ABSOLUTE", "JUMP_FORWARD"}
_terminating = {"RETURN_VALUE"}


def is_conditional_jump(opname: str) -> bool:
    return opname in _cond_jump


def is_unconditional_jump(opname: str) -> bool:
    return opname in _uncond_jump


def is_exiting(opname: str) -> bool:
    return opname in _terminating


def _next_inst_offset(offset: Label) -> BCLabel:
    # Fix offset
    assert isinstance(offset, BCLabel)
    return BCLabel(offset.offset + 2)


def _prev_inst_offset(offset: Label) -> BCLabel:
    # Fix offset
    assert isinstance(offset, BCLabel)
    return BCLabel(offset.offset - 2)


@dataclass()
class FlowInfo:
    """ FlowInfo converts Bytecode into a ByteFlow object (CFG). """

    block_offsets: Set[Label] = field(default_factory=set)
    """Marks starting offset of basic-block
    """

    jump_insts: Dict[Label, Tuple[Label, ...]] = field(default_factory=dict)
    """Contains jump instructions and their target offsets.
    """

    last_offset: int = field(default=0)
    """Offset of the last bytecode instruction.
    """

    def _add_jump_inst(self, offset: Label, targets: Sequence[Label]):
        """Add jump instruction to FlowInfo."""
        for off in targets:
            assert isinstance(off, Label)
            self.block_offsets.add(off)
        self.jump_insts[offset] = tuple(targets)

    @staticmethod
    def from_bytecode(bc: dis.Bytecode) -> "FlowInfo":
        """
        Build control-flow information that marks start of basic-blocks and
        jump instructions.
        """
        flowinfo = FlowInfo()

        for inst in bc:
            # Handle jump-target instruction
            if inst.offset == 0 or inst.is_jump_target:
                flowinfo.block_offsets.add(BCLabel(inst.offset))
            # Handle by op
            if is_conditional_jump(inst.opname):
                flowinfo._add_jump_inst(
                    BCLabel(inst.offset),
                    (BCLabel(inst.argval),
                     _next_inst_offset(BCLabel(inst.offset))),
                )
            elif is_unconditional_jump(inst.opname):
                flowinfo._add_jump_inst(BCLabel(inst.offset),
                                        (BCLabel(inst.argval),))
            elif is_exiting(inst.opname):
                flowinfo._add_jump_inst(BCLabel(inst.offset), ())

        flowinfo.last_offset = inst.offset
        return flowinfo

    def build_basicblocks(self: "FlowInfo", end_offset=None) -> "BlockMap":
        """
        Build a graph of basic-blocks
        """
        offsets = sorted(self.block_offsets)
        if end_offset is None:
            end_offset = _next_inst_offset(BCLabel(self.last_offset))
        bbmap = BlockMap()
        for begin, end in zip(offsets, [*offsets[1:], end_offset]):
            targets: Tuple[Label, ...]
            term_offset = _prev_inst_offset(end)
            if term_offset not in self.jump_insts:
                # implicit jump
                targets = (end,)
                fallthrough = True
            else:
                targets = self.jump_insts[term_offset]
                fallthrough = False
            bb = BasicBlock(begin=begin,
                            end=end,
                            jump_targets=targets,
                            fallthrough=fallthrough,
                            backedges=(),
                            )
            bbmap.add_node(bb)
        return bbmap


@dataclass(frozen=True)
class BlockMap:
    """ Map of Labels to Blocks. """
    graph: Dict[Label, BasicBlock] = field(default_factory=dict)
    clg: ControlLabelGenerator = field(default_factory=ControlLabelGenerator,
                                       compare=False)

    def add_node(self, basicblock: BasicBlock):
        self.graph[basicblock.begin] = basicblock

    def remove_blocks(self, labels: Set[Label]):
        for label in labels:
            del self.graph[label]

    def __getitem__(self, index):
        return self.graph[index]

    def exclude_nodes(self, exclude_nodes: Set[Label]):
        """Iterator over all nodes not in exclude_nodes. """
        for node in self.graph:
            if node not in exclude_nodes:
                yield node

    def find_head(self):
        heads = set(self.graph.keys())
        for block in self.graph.keys():
            for jt in self.graph[block].jump_targets:
                heads.discard(jt)
        assert len(heads) == 1
        return next(iter(heads))

    def compute_scc(self) -> List[Set[Label]]:
        """
        Strongly-connected component for detecting loops.
        """
        from scc import scc

        class GraphWrap:
            def __init__(self, graph):
                self.graph = graph

            def __getitem__(self, vertex):
                out = self.graph[vertex].jump_targets
                # Exclude node outside of the subgraph
                return [k for k in out if k in self.graph]

            def __iter__(self):
                return iter(self.graph.keys())

        return list(scc(GraphWrap(self.graph)))

    def find_headers_and_entries(self, loop: Set[Label]):
        """Find entries and headers in a given loop.

        Entries are nodes outside the loop that have an edge pointing to the
        loop header. Headers are nodes that are part of the strongly connected
        subset, that have incoming edges from outside the loop. Entries point
        to headers and headers are pointed to by entries.

        """
        node: Label
        entries: Set[Label] = set()
        headers: Set[Label] = set()

        for node in self.exclude_nodes(loop):
            nodes_jump_in_loop = set(self.graph[node].jump_targets) & loop
            headers |= nodes_jump_in_loop
            if nodes_jump_in_loop:
                entries.add(node)

        return headers, entries

    def find_exits(self, subgraph: Set[Label]):
        """Find pre-exits and post-exits in a given subgraph.

        Pre-exits are nodes inside the subgraph that have edges to nodes
        outside of the subgraph. Post-exits are nodes  outside the subgraph
        that have incoming edges from within the subgraph.

        """
        node: Label
        pre_exits: Set[Label] = set()
        post_exits: Set[Label] = set()
        for inside in subgraph:
            # any node inside that points outside the loop
            for jt in self.graph[inside].jump_targets:
                if jt not in subgraph:
                    pre_exits.add(inside)
                    post_exits.add(jt)
            # any returns
            if self.graph[inside].is_exiting():
                pre_exits.add(inside)
        return pre_exits, post_exits

    def join_returns(self):
        """ Close the CFG.

        A closed CFG is a CFG with a unique entry and exit node that have no
        predescessors and no successors respectively.
        """
        # for all nodes that contain a return
        return_nodes = [node for node in self.graph
                        if self.graph[node].is_exiting()]
        # if there is more than one, we may need to close it
        if len(return_nodes) > 1:
            # create label and block and add to graph
            return_solo_label = ControlLabel(str(self.clg.new_index()))
            return_solo_block = BasicBlock(
                begin=return_solo_label,
                end=ControlLabel("end"),
                fallthrough=False,
                jump_targets=set(),
                backedges=set()
                )
            self.add_node(return_solo_block)
            # re-wire all previous exit nodes to the synthetic one
            for rnode in return_nodes:
                self.add_node(self.graph.pop(rnode).replace_jump_targets(
                            jump_targets=set((return_solo_label,))))

    def is_reachable_dfs(self, begin, end):
        """Is end reachable from begin. """
        seen = set()
        to_vist = list(self.graph[begin].jump_targets)
        while True:
            if to_vist:
                block = to_vist.pop()
            else:
                return False

            if block in seen:
                continue
            elif block == end:
                return True
            elif block not in seen:
                seen.add(block)
                if block in self.graph:
                    to_vist.extend(self.graph[block].jump_targets)

    def join_tails_and_exits(self, tails: Set[Label], exits: Set[Label]):
        if len(tails) == 1 and len(exits) == 1:
            # no-op
            solo_tail_label = next(iter(tails))
            solo_exit_label = next(iter(exits))
            return solo_tail_label, solo_exit_label

        if len(tails) == 1 and len(exits) == 2:
            # join only exits
            solo_tail_label = next(iter(tails))
            solo_exit_label = ControlLabel(str(self.clg.new_index()))
            # The solo exit block points to the exits
            solo_exit_block = BasicBlock(begin=solo_exit_label,
                                         end=ControlLabel("end"),
                                         fallthrough=False,
                                         jump_targets=set(exits),
                                         backedges=set()
                                         )
            self.add_node(solo_exit_block)
            # Update the solo tail block to point to the solo exit block
            self.add_node(self.graph.pop(solo_tail_label).replace_jump_targets(
                          jump_targets=set((solo_exit_label,))))

            return solo_tail_label, solo_exit_label

        if len(tails) >= 2 and len(exits) == 1:
            # join only tails
            solo_tail_label = ControlLabel(str(self.clg.new_index()))
            solo_exit_label = next(iter(exits))
            # The solo tail block points to the solo exit block
            solo_tail_block = BasicBlock(begin=solo_tail_label,
                                         end=ControlLabel("end"),
                                         fallthrough=True,
                                         jump_targets=set((solo_exit_label,)),
                                         backedges=set()
                                         )
            self.add_node(solo_tail_block)
            # replace the exit label in all tails blocks with the solo tail label
            for label in tails:
                block = self.graph.pop(label)
                jt = set(block.jump_targets)
                jt.remove(solo_exit_label)
                jt.add(solo_tail_label)
                self.add_node(block.replace_jump_targets(jump_targets=set(jt)))

            return solo_tail_label, solo_exit_label

        if len(tails) >= 2 and len(exits) >= 2:
            # join both tails and exits
            solo_tail_label = ControlLabel(str(self.clg.new_index()))
            solo_exit_label = ControlLabel(str(self.clg.new_index()))
            # The solo tail block points to the solo exit block
            solo_tail_block = BasicBlock(begin=solo_tail_label,
                                         end=ControlLabel("end"),
                                         fallthrough=True,
                                         jump_targets=set((solo_exit_label,)),
                                         backedges=set()
                                         )
            # The solo exit block points to the exits
            solo_exit_block = BasicBlock(begin=solo_exit_label,
                                         end=ControlLabel("end"),
                                         fallthrough=False,
                                         jump_targets=set(exits),
                                         backedges=set()
                                         )
            self.add_node(solo_tail_block)
            self.add_node(solo_exit_block)
            # Replace all previous jump targets that went outside the loop to
            # the new tail label.
            for label in tails:
                block = self.graph.pop(label)
                jt = set(block.jump_targets)
                jt.difference_update(jt.intersection(exits))
                jt.add(solo_tail_label)
                self.add_node(block.replace_jump_targets(jump_targets=set(jt)))

            return solo_tail_label, solo_exit_label


@dataclass(frozen=True)
class ByteFlow:
    bc: dis.Bytecode
    bbmap: "BlockMap"

    @staticmethod
    def from_bytecode(code) -> "ByteFlow":
        bc = dis.Bytecode(code)
        _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

        flowinfo = FlowInfo.from_bytecode(bc)
        bbmap = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, bbmap=bbmap)

    def _join_returns(self):
        bbmap = deepcopy(self.bbmap)
        bbmap.join_returns()
        return ByteFlow(bc=self.bc, bbmap=bbmap)

    def _restructure_loop(self):
        bbmap = deepcopy(self.bbmap)
        restructure_loop(bbmap)
        return ByteFlow(bc=self.bc, bbmap=bbmap)

    def _restructure_branch(self):
        bbmap = deepcopy(self.bbmap)
        restructure_branch(bbmap)
        for region in _iter_subregions(bbmap):
            restructure_branch(region.subregion)
        return ByteFlow(bc=self.bc, bbmap=bbmap)

    def restructure(self):
        bbmap = deepcopy(self.bbmap)
        # close
        bbmap.join_returns()
        # handle loop
        restructure_loop(bbmap)
        # handle branch
        restructure_branch(bbmap)
        for region in _iter_subregions(bbmap):
            restructure_branch(region.subregion)
        return ByteFlow(bc=self.bc, bbmap=bbmap)


def _iter_subregions(bbmap: "BlockMap"):
    for node in bbmap.graph.values():
        if isinstance(node, RegionBlock):
            yield node
            yield from _iter_subregions(node.subregion)


def join_exits(loop: Set[Label], bbmap: BlockMap, exits: Set[Label]):
    # create a single exit label and add it to the loop
    pre_exit_label = ControlLabel(clg.new_index())
    post_exit_label = ControlLabel(clg.new_index())
    loop.add(pre_exit_label)
    # create the exit block and add it to the block map
    post_exit_block = BasicBlock(begin=post_exit_label,
                                 end="end",
                                 fallthrough=False,
                                 jump_targets=tuple(exits),
                                 backedges=tuple()
                                 )
    pre_exit_block = BasicBlock(begin=pre_exit_label,
                                end="end",
                                fallthrough=False,
                                jump_targets=(post_exit_label,),
                                backedges=tuple()
                                )
    bbmap.add_node(pre_exit_block)
    bbmap.add_node(post_exit_block)
    # for all exits, find the nodes that jump to this exit
    # this is effectively finding the exit vertices
    for exit_node in exits:
        for loop_node in loop:
            if loop_node == pre_exit_label:
                continue
            if exit_node in bbmap.graph[loop_node].jump_targets:
                # update the jump_targets to point to the new exitnode
                # by replacing the original node with updates
                new_jump_targets = tuple(
                    [t for t in bbmap.graph[loop_node].jump_targets
                     if t != exit_node]
                    + [pre_exit_label])
                bbmap.add_node(
                    bbmap.graph.pop(loop_node).replace_jump_targets(
                        jump_targets=new_jump_targets))
    return pre_exit_label, post_exit_label


def join_pre_exits(exits: Set[Label], post_exit: Label,
                   inner_nodes: Set[Label], bbmap: BlockMap, ):
    pre_exit_label = ControlLabel(clg.new_index())
    # create the exit block and add it to the block map
    pre_exit_block = BasicBlock(begin=pre_exit_label,
                                end="end",
                                fallthrough=False,
                                jump_targets=(post_exit,),
                                backedges=tuple()
                                )
    bbmap.add_node(pre_exit_block)
    inner_nodes.add(pre_exit_label)
    # for all exits, find the nodes that jump to this exit
    # this is effectively finding the exit vertices
    for exit_node in exits:
        bbmap.add_node(
            bbmap.graph.pop(exit_node).replace_jump_targets(
                jump_targets=(pre_exit_label,)))
    return pre_exit_label


def join_headers(headers, entries, bbmap):
    assert len(headers) > 1
    # create the synthetic entry block
    synth_entry_label = ControlLabel(bbmap.clg.new_index())
    synth_entry_block = BasicBlock(begin=synth_entry_label,
                                   end="end",
                                   fallthrough=False,
                                   jump_targets=tuple(headers),
                                   backedges=tuple()
                                   )
    bbmap.add_node(synth_entry_block)
    # rewire headers
    for label in entries:
        block = bbmap.graph.pop(label)
        jt = set(block.jump_targets)
        jt.difference_update(jt.intersection(headers))
        jt.add(synth_entry_label)
        bbmap.add_node(block.replace_jump_targets(
                        jump_targets=jt))
    return synth_entry_label, synth_entry_block


def loop_rotate(bbmap: BlockMap, loop: Set[Label]):
    """ Rotate loop.

    This will convert the loop into "loop canonical form".
    """
    headers, entries = bbmap.find_headers_and_entries(loop)
    pre_exits, post_exits = bbmap.find_exits(loop)

    # find the loop head
    assert len(headers) == 1  # TODO join entries
    loop_head: Label = next(iter(headers))

    # the loop head should have two jump targets
    loop_head_jt = bbmap[loop_head].jump_targets
    assert len(loop_head_jt) == 2
    llhjt_list = list(loop_head_jt)
    if llhjt_list[0] in loop:
        loop_body_start = llhjt_list[0]
        loop_head_exit = llhjt_list[1]
    else:
        loop_body_start = llhjt_list[1]
        loop_head_exit = llhjt_list[0]

    # find the backedge that points to the loop head
    backedge_blocks = [block for block in loop
                       if loop_head in bbmap[block].jump_targets]
    assert len(backedge_blocks) == 1
    backedge_block = backedge_blocks[0]
    bbmap.add_node(bbmap.graph.pop(backedge_block).replace_jump_targets(
        jump_targets=loop_head_jt))

    headers, entries = bbmap.find_headers_and_entries(loop)
    pre_exits, post_exits = bbmap.find_exits(loop)
    #pre_exit_label, post_exit_label = bbmap.join_tails_and_exits(pre_exits,
    #                                                             post_exits)
    #loop.add(pre_exit_label)

    # recompute the scc
    subgraph_bbmap = BlockMap({label: bbmap[label] for label in loop})
    new_loops = []
    for scc in subgraph_bbmap.compute_scc():
        if len(scc) == 1:
            solo_block = next(iter(scc))
            if solo_block in bbmap[solo_block].jump_targets:
                new_loops.append(scc)
        else:
            new_loops.append(scc)
    assert(len(new_loops) == 1)
    new_loop = next(iter(new_loops))
    loop.clear()
    loop.update(new_loop)

    headers, entries = bbmap.find_headers_and_entries(loop)
    pre_exits, post_exits = bbmap.find_exits(loop)
    #pre_exit_label = next(iter(pre_exits))
    #post_exit_label = next(iter(post_exits))
    pre_exit_label, post_exit_label = bbmap.join_tails_and_exits(pre_exits,
                                                                 post_exits)
    loop.add(pre_exit_label)
    loop_head = next(iter(headers))

    # fixup backedges
    for label in loop:
        bbmap.add_node(
            bbmap.graph.pop(label).replace_backedge(loop_head))

    loop_body_start = next(iter(bbmap[loop_head].jump_targets))

    return headers, loop_head, loop_body_start, pre_exit_label, post_exit_label


def restructure_loop(bbmap: BlockMap):
    """Inplace restructuring of the given graph to extract loops using
    strongly-connected components
    """
    # obtain a List of Sets of Labels, where all labels in each set are strongly
    # connected, i.e. all reachable from one another by traversing the subset
    scc: List[Set[Label]] = bbmap.compute_scc()
    # loops are defined as strongly connected subsets who have more than a
    # single label
    loops: List[Set[Label]] = [nodes for nodes in scc if len(nodes) > 1]
    _logger.debug("restructure_loop found %d loops in %s",
                  len(loops), bbmap.graph.keys())

    # extract loop
    for loop in loops:
        headers, loop_head, loop_body_start, pre_exit_label, post_exit_label = loop_rotate(bbmap, loop)
        loop_subregion = BlockMap({
            label: bbmap[label] for label in loop}, clg=bbmap.clg)

        # create a subregion
        blk = RegionBlock(
            begin=loop_head,
            end="end",
            fallthrough=False,
            jump_targets=(post_exit_label,),
            backedges=(),
            kind="loop",
            subregion=loop_subregion,
            headers={header: bbmap.graph[header] for header in headers},
            exit=pre_exit_label
        )
        # Remove the nodes in the subregion
        bbmap.remove_blocks(loop)
        # insert subregion back into original
        bbmap.graph[loop_head] = blk
        # process subregions
        restructure_loop(blk.subregion)


def restructure_branch(bbmap: BlockMap):
    print("restructure_branch", bbmap.graph)
    doms = _doms(bbmap)
    postdoms = _post_doms(bbmap)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    recursive_subregions = []
    # TODO what are begin and end exactly? The assumtion is that they are
    # unique, is this true?
    for begin, end in _iter_branch_regions(bbmap, immdoms, postimmdoms):
        _logger.debug("branch region: %s -> %s", begin, end)
        # find exiting nodes from branch
        # exits = {k for k, node in bbmap.graph.items()
        #          if begin <= k < end and end in node.jump_targets}
        # partition the branches

        # partition head subregion
        head = bbmap.find_head()
        head_region_blocks = []
        current_block = head
        # Start at the head block and traverse the graph linearly until
        # reaching the begin block.
        while True:
            head_region_blocks.append(current_block)
            if current_block == begin:
                break
            else:
                jt = bbmap.graph[current_block].jump_targets
                assert len(jt) == 1
                current_block = jt[0]
        # Extract the head subregion
        head_subgraph = BlockMap({block: bbmap.graph[block]
                                  for block in head_region_blocks},
                                  clg=bbmap.clg)
        head_subregion = RegionBlock(
            begin=head,
            end=begin,
            fallthrough=False,
            jump_targets=bbmap.graph[begin].jump_targets,
            backedges=(),
            kind="head",
            headers=(head,),
            subregion=head_subgraph,
            exit=begin,
        )
        begin = head
        bbmap.remove_blocks(head_region_blocks)
        bbmap.graph[begin] = head_subregion

        # insert synthetic branch blocks in case of empty branch regions
        jump_targets = list(bbmap.graph[begin].jump_targets)
        new_jump_targets = []
        jump_targets_changed = False
        for a in jump_targets:
            for b in jump_targets:
                if a == b:
                    continue
                elif bbmap.is_reachable_dfs(a, b):

                    # If one of the jump targets is reachable from the other,
                    # it means a branch region is empty and the reachable jump
                    # target becoms part of the tail. In this case,
                    # create a new snythtic block to fill the empty branch
                    # region and rewire the jump targets.
                    synthetic_branch_block_label = ControlLabel(bbmap.clg.new_index())
                    synthetic_branch_block_block = BasicBlock(
                            begin=synthetic_branch_block_label,
                            end="end",
                            jump_targets=(b,),
                            fallthrough=True,
                            backedges=(),
                            )
                    bbmap.add_node(synthetic_branch_block_block)
                    new_jump_targets.append(synthetic_branch_block_label)
                    jump_targets_changed = True
                else:
                    # otherwise just re-use the existing block
                    new_jump_targets.append(b)

        # update the begin block with new jump_targets
        if jump_targets_changed:
            bbmap.add_node(
                bbmap.graph.pop(begin).replace_jump_targets(
                    jump_targets=new_jump_targets))
            # and recompute doms
            doms = _doms(bbmap)
            postdoms = _post_doms(bbmap)
            postimmdoms = _imm_doms(postdoms)
            immdoms = _imm_doms(doms)

        # identify branch regions
        branch_regions = []
        for bra_start in bbmap.graph[begin].jump_targets:
            sub_keys: Set[BCLabel] = set()
            branch_regions.append((bra_start, sub_keys))
            # a node is part of the branch if
            # - the start of the branch is a dominator; and,
            # - the tail of the branch is not a dominator
            for k, kdom in doms.items():
                if bra_start in kdom and end not in kdom:
                    sub_keys.add(k)

        # identify and close tail
        #
        # find all nodes that are left
        # TODO this could be a set operation
        tail_subregion = set((b for b in bbmap.graph.keys()))
        for b, sub in branch_regions:
            tail_subregion.discard(b)
            for s in sub:
                tail_subregion.discard(s)
        # exclude parents
        tail_subregion.discard(begin)

        headers, entries = bbmap.find_headers_and_entries(tail_subregion)
        exits, _ = bbmap.find_exits(tail_subregion)
        #if len(returns) == 0:
        #    returns = set([block for block in tail_subregion if bbmap.graph[block].])

        #assert len(returns) == 1

        if len(headers) > 1:
            entry_label, entry_block = join_headers(
                headers, entries, bbmap)
            tail_subregion.add(entry_label)
        else:
            entry_label = next(iter(headers))

        # end of the graph
        #assert len(pre_exits) == 1
        #assert len(post_exits) == 0

        subgraph = BlockMap(clg=bbmap.clg)
        for block in tail_subregion:
            subgraph.add_node(bbmap.graph[block])
        subregion = RegionBlock(
            begin=entry_label,
            end=next(iter(exits)),
            fallthrough=False,
            jump_targets=(),
            backedges=(),
            kind="tail",
            headers=headers,
            subregion=subgraph,
            exit=None,
        )
        bbmap.remove_blocks(tail_subregion)
        bbmap.graph[entry_label] = subregion

        if subregion.subregion.graph:
            recursive_subregions.append(subregion.subregion)

        # extract the subregion
        for bra_start, inner_nodes in branch_regions:
            if inner_nodes:  # and len(inner_nodes) > 1:
                pre_exits, post_exits = bbmap.find_exits(inner_nodes)
                #if len(pre_exits) != 1 and len(post_exits) == 1:
                #    post_exit_label = next(iter(post_exits))
                #    pre_exit_label = join_pre_exits(
                #        pre_exits,
                #        post_exit_label,
                #        inner_nodes,
                #        bbmap)
                #elif len(pre_exits) != 1 and len(post_exits) > 1:
                #    pre_exit_label, post_exit_label = join_exits(
                #        inner_nodes,
                #        bbmap,
                #        post_exits)
                #else:
                #    pre_exit_label = next(iter(pre_exits))
                #    post_exit_label = next(iter(bbmap.graph[pre_exit_label].jump_targets))
                pre_exit_label, post_exit_label = bbmap.join_tails_and_exits(pre_exits, post_exits)

                if isinstance(bbmap[pre_exit_label], RegionBlock):
                    pre_exit_label = bbmap[pre_exit_label].exit

                subgraph = BlockMap(clg=bbmap.clg)
                for k in inner_nodes:
                    subgraph.add_node(bbmap.graph[k])

                subregion = RegionBlock(
                    begin=bra_start,
                    end=end,
                    fallthrough=False,
                    jump_targets=(post_exit_label,),
                    backedges=(),
                    kind="branch",
                    headers={},
                    subregion=subgraph,
                    exit=pre_exit_label,
                )
                bbmap.remove_blocks(inner_nodes)
                bbmap.graph[bra_start] = subregion

                if subregion.subregion.graph:
                    recursive_subregions.append(subregion.subregion)

    # recurse into subregions as necessary
    for region in recursive_subregions:
        restructure_branch(region)


def _iter_branch_regions(bbmap: BlockMap,
                         immdoms: Dict[Label, Label],
                         postimmdoms: Dict[Label, Label]):
    for begin, node in [i for i in bbmap.graph.items()]:
        if len(node.jump_targets) > 1:
            # found branch
            if begin in postimmdoms:
                end = postimmdoms[begin]
                if immdoms[end] == begin:
                    yield begin, end


def _imm_doms(doms: Dict[Label, Set[Label]]) -> Dict[Label, Label]:
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


def _doms(bbmap: BlockMap):
    # compute dom
    entries = set()
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    node: BasicBlock
    for src, node in bbmap.graph.items():
        for dst in node.jump_targets:
            # check dst is in subgraph
            if dst in bbmap.graph:
                preds_table[dst].add(src)
                succs_table[src].add(dst)

    for k in bbmap.graph:
        if not preds_table[k]:
            entries.add(k)
    return _find_dominators_internal(entries, list(bbmap.graph.keys()), preds_table, succs_table)


def _post_doms(bbmap: BlockMap):
    # compute post dom
    entries = set()
    for k, v in bbmap.graph.items():
        targets = set(v.jump_targets) & set(bbmap.graph)
        if not targets:
            entries.add(k)
    preds_table = defaultdict(set)
    succs_table = defaultdict(set)

    node: BasicBlock
    for src, node in bbmap.graph.items():
        for dst in node.jump_targets:
            # check dst is in subgraph
            if dst in bbmap.graph:
                preds_table[src].add(dst)
                succs_table[dst].add(src)

    return _find_dominators_internal(entries, list(bbmap.graph.keys()), preds_table, succs_table)


def _find_dominators_internal(entries, nodes, preds_table, succs_table):
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
        raise RuntimeError("no entry points: dominator algorithm "
                           "cannot be seeded")

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
            new_doms |= functools.reduce(set.intersection,
                                         [doms[p] for p in preds])
        if new_doms != doms[n]:
            assert len(new_doms) < len(doms[n])
            doms[n] = new_doms
            todo.extend(succs_table[n])
    return doms


class ByteFlowRenderer(object):

    def __init__(self):
        from graphviz import Digraph
        self.g = Digraph()

    def render_region_block(self, digraph: "Digraph", label: Label, regionblock: RegionBlock):
        # render subgraph
        graph = regionblock.get_full_graph()
        with digraph.subgraph(name=f"cluster_{label}") as subg:
            color = 'blue'
            if regionblock.kind == 'branch':
                color = 'green'
            if regionblock.kind == 'tail':
                color = 'purple'
            if regionblock.kind == 'head':
                color = 'red'
            subg.attr(color=color, label=regionblock.kind)
            for label, block in graph.items():
                self.render_block(subg, label, block)
        # render edges within this region
        self.render_edges(graph)

    def render_basic_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        if isinstance(label, BCLabel):
            instlist = block.get_instructions(self.bcmap)
            body = "\l".join(
                [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
            )
        elif isinstance(label, ControlLabel):
            body = "Control Label: " + str(label.index)
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        if type(block) == BasicBlock:
            self.render_basic_block(digraph, label, block)
        elif type(block) == RegionBlock:
            self.render_region_block(digraph, label, block)
        else:
            raise Exception("unreachable")

    def render_edges(self, blocks: Dict[Label, BasicBlock]):
        for label, block in blocks.items():
            for dst in block.jump_targets:
                if dst in blocks:
                    if type(block) == BasicBlock:
                        self.g.edge(str(label), str(dst))
                    elif type(block) == RegionBlock:
                        if block.exit is not None:
                            self.g.edge(str(block.exit), str(dst))
                        else:
                            self.g.edge(str(label), str(dst))
                    else:
                        raise Exception("unreachable")
            for dst in block.backedges:
                #assert dst in blocks
                self.g.edge(str(label), str(dst),
                            style="dashed", color="grey", constraint="0")

    def render_byteflow(self, byteflow: ByteFlow):
        self.bcmap_from_bytecode(byteflow.bc)

        # render nodes
        for label, block in byteflow.bbmap.graph.items():
            self.render_block(self.g, label, block)
        self.render_edges(byteflow.bbmap.graph)
        return self.g

    def bcmap_from_bytecode(self, bc: dis.Bytecode):
        self.bcmap: Dict[Label, dis.Instruction] = {BCLabel(inst.offset): inst
                                                    for inst in bc}
