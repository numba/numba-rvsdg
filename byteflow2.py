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


def bcmap_from_bytecode(bc: dis.Bytecode) -> Dict[Label, dis.Instruction]:
    bcmap: Dict[Label, dis.Instruction] = {BCLabel(inst.offset): inst
                                           for inst in bc}
    return bcmap


@dataclass(frozen=True)
class Block:
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

    jump_targets: Tuple[Label, ...]
    """The destination block offsets."""

    backedges: Tuple[Label, ...]
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

    def render_dot(self,
                   g,
                   node_offset: Label,
                   bcmap: Dict[Label, dis.Instruction]):
        # The node may not have instructions, insert a synthetic string instead
        # if that is the case.
        try:
            instlist = self.get_instructions(bcmap)
        except KeyError:
            instlist = []
        if instlist:
            body = "\l".join(
                [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
            )
        else:
            body = "SYNTHETIC"
        g.node(str(node_offset), shape="rect", label=body)


@dataclass()
class FlowInfo:
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
            bb = Block(begin=begin,
                       end=end,
                       jump_targets=targets,
                       fallthrough=fallthrough,
                       backedges=(),
                       )
            bbmap.add_node(bb)
        return bbmap


@dataclass(frozen=True)
class ByteFlow:
    bc: dis.Bytecode
    bbmap: "BlockMap"

    def render_dot(self):
        return render_dot(self.bc, self.bbmap)

    @staticmethod
    def from_bytecode(code) -> "ByteFlow":
        bc = dis.Bytecode(code)
        _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

        flowinfo = FlowInfo.from_bytecode(bc)
        bbmap = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, bbmap=bbmap)

    def restructure(self):
        bbmap = deepcopy(self.bbmap)
        # handle loop
        restructure_loop(bbmap)
        # handle branch
        #restructure_branch(bbmap)
        #for region in _iter_subregions(bbmap):
        #    restructure_branch(region.subregion)
        return ByteFlow(bc=self.bc, bbmap=bbmap)


def _iter_subregions(bbmap: "BlockMap"):
    for node in bbmap.graph.values():
        if isinstance(node, RegionBlock):
            yield node
            yield from _iter_subregions(node.subregion)




@dataclass(frozen=True)
class RegionBlock(Block):
    kind: str
    headers: Dict[Label, Block]
    """The header of the region"""
    subregion: "BlockMap"
    """The subgraph excluding the headers
    """

    def render_dot(self, g, node_offset: Label, bcmap: Dict[Label, dis.Instruction]):
        # render subgraph
        graph = self.get_full_graph()
        with g.subgraph(name=f"cluster_{node_offset}") as subg:
            color = 'blue'
            if self.kind == 'branch':
                color = 'green'
            subg.attr(color=color, label=self.kind)
            for k, node in graph.items():
                node.render_dot(subg, k, bcmap)
        # render edges within this region
        render_edges(g, graph)

    def get_full_graph(self):
        graph = ChainMap(self.subregion.graph, self.headers)
        return graph

@dataclass(frozen=True)
class BlockMap:
    graph: Dict[Label, Block] = field(default_factory=dict)

    def add_node(self, bb: Block):
        self.graph[bb.begin] = bb


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


def compute_scc(bbmap: BlockMap) -> List[Set[Label]]:
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

    return list(scc(GraphWrap(bbmap.graph)))


def render_dot(bc: dis.Bytecode, bbmap: BlockMap):
    from graphviz import Digraph

    bcmap: Dict[Label, dis.Instruction] = {BCLabel(inst.offset): inst for inst in bc}

    g = Digraph()
    # render nodes
    for k, node in bbmap.graph.items():
        node.render_dot(g, k, bcmap)
    # render edges
    render_edges(g, bbmap.graph)
    return g


def render_edges(g, nodes: Dict[Label, Block]):
    for k, node in nodes.items():
        for dst in node.jump_targets:
            if dst in nodes:
                g.edge(str(k), str(dst))
        for dst in node.backedges:
            assert dst in nodes
            g.edge(str(k), str(dst), style="dashed", color="grey", constraint="0")


def restructure_loop(bbmap: BlockMap):
    """Inplace restructuring of the given graph to extract loops using
    strongly-connected components
    """
    # obtain a List of Sets of Labels, where all labels in each set are strongly
    # connected, i.e. all reachable from one another by traversing the subset
    scc: List[Set[Label]] = compute_scc(bbmap)
    # loops are defined as strongly connected subsets who have more than a
    # single label
    loops: List[Set[Label]] = [nodes for nodes in scc if len(nodes) > 1]
    _logger.debug("restructure_loop found %d loops in %s",
                  len(loops), bbmap.graph.keys())

    node: Label
    # extract loop
    for loop in loops:
        _logger.debug("loop nodes %s", loop)
        # find entries and headers
        # entries are nodes outside the loop that have an edge pointing to the
        # loop header
        # headers are nodes that are part of the strongly connected subset,
        # that have incoming edges from outside the loop
        # entries point to headers and headers are pointed to by entries
        entries: Set[Label] = set()
        headers: Set[Label] = set()

        for node in _exclude_nodes(bbmap.graph, loop):
            nodes_jump_in_loop = set(bbmap.graph[node].jump_targets) & loop
            headers |= nodes_jump_in_loop
            if nodes_jump_in_loop:
                entries.add(node)

        # find exits
        # exits are nodes outside the loop that have incoming edges from
        # within the loop
        exits = set()
        for node in loop:
            for outside in _exclude_nodes(bbmap.graph, loop):
                if outside in bbmap.graph[node].jump_targets:
                    exits.add(outside)

        # remove loop nodes from cfg/bbmap
        # use the set of labels to remove/pop Blocks into a set of blocks
        insiders: Set[Block] = {bbmap.graph.pop(k) for k in loop}

        assert len(headers) == 1, headers  # TODO join entries and exits
        # turn singleton set into single element
        loop_head: Label = next(iter(headers))
        # construct the loop body, identifying backedges as we go
        loop_body: Dict[Label, Block] = {
            node.begin: _replace_backedge(node, loop_head)
            for node in insiders if node.begin not in headers}
        # create a subregion
        blk = RegionBlock(
            begin=loop_head,
            end=_next_inst_offset(loop_head),
            fallthrough=False,
            jump_targets=tuple(exits),
            backedges=(),
            kind="loop",
            subregion=BlockMap(loop_body),
            headers={node.begin: node for node in insiders
                     if node.begin in headers},
        )
        # process subregions
        restructure_loop(blk.subregion)
        # insert subregion back into original
        bbmap.graph[loop_head] = blk


def _replace_backedge(node: Block, loop_head: Label) -> Block:
    if loop_head in node.jump_targets:
        assert len(node.jump_targets) == 1
        assert not node.backedges
        return replace(node, jump_targets=(), backedges=(loop_head,))
    return node


def restructure_branch(bbmap: BlockMap):
    print("restructure_branch", bbmap.graph)
    doms = _doms(bbmap)
    postdoms = _post_doms(bbmap)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    for begin, end in _iter_branch_regions(bbmap, immdoms, postimmdoms):
        _logger.debug("branch region: %s -> %s", begin, end)
        # find exiting nodes from branch
        # exits = {k for k, node in bbmap.graph.items()
        #          if begin <= k < end and end in node.jump_targets}

        # partition the branches
        branch_regions = []
        for bra_start in bbmap.graph[begin].jump_targets:
            sub_keys: Set[int]  = set()
            branch_regions.append((bra_start, sub_keys))
            # a node is part of the branch if
            # - the start of the branch is a dominator; and,
            # - the tail of the branch is not a dominator
            for k, kdom in doms.items():
                if bra_start in kdom and end not in kdom:
                    sub_keys.add(k)

        # extract the subregion
        for bra_start, inner_nodes in branch_regions:
            if inner_nodes:
                subgraph = BlockMap()
                for k in inner_nodes:
                    subgraph.add_node(bbmap.graph[k])

                subregion = RegionBlock(
                    begin=bra_start,
                    end=end,
                    fallthrough=False,
                    jump_targets=(end,),
                    backedges=(),
                    kind="branch",
                    headers={},
                    subregion=subgraph,
                )
                for k in inner_nodes:
                    del bbmap.graph[k]
                bbmap.graph[bra_start] = subregion

                # recursive
                if subregion.subregion.graph:
                    restructure_branch(subregion.subregion)
        break

def _iter_branch_regions(bbmap: BlockMap,
                         immdoms: Dict[Label, Label],
                         postimmdoms: Dict[Label, Label]):
    for begin, node in bbmap.graph.items():
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

    node : Block
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

    node : Block
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



def _exclude_nodes(iter, nodeset: Set):
    for it in iter:
        if it not in nodeset:
            yield it
