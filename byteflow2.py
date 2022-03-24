import dis
from collections import deque, ChainMap, defaultdict
from typing import Optional, Set, Tuple, Dict, List
from pprint import pprint
from dataclasses import dataclass, field, replace
import logging

_logger = logging.getLogger(__name__)


class _LogWrap:
    def __init__(self, fn):
        self._fn = fn

    def __str__(self):
        return self._fn()

@dataclass(frozen=True)
class ByteFlow:
    bc: dis.Bytecode
    bbmap: "BlockMap"

    def render_dot(self):
        return render_dot(self.bc, self.bbmap)


def parse_bytecode(code) -> ByteFlow:
    bc = dis.Bytecode(code)
    _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

    end_offset = _next_inst_offset([inst.offset for inst in bc][-1])
    flowinfo = build_flowinfo(bc)
    bbmap = build_basicblocks(flowinfo, end_offset)
    # handle loop
    restructure_loop(bbmap)
    # handle branch
    restructure_branch(bbmap)
    # render
    return ByteFlow(bc=bc, bbmap=bbmap)


def build_flowinfo(bc: dis.Bytecode) -> "FlowInfo":
    """
    Build control-flow information that marks start of basic-blocks and jump
    instructions.
    """
    flowinfo = FlowInfo()

    for inst in bc:
        # Handle jump-target instruction
        if inst.offset == 0 or inst.is_jump_target:
            flowinfo.block_offsets.add(inst.offset)
        # Handle by op
        if is_conditional_jump(inst.opname):
            flowinfo.add_jump_inst(
                inst.offset, (inst.argval, _next_inst_offset(inst.offset)),
            )
        elif is_unconditional_jump(inst.opname):
            flowinfo.add_jump_inst(inst.offset, (inst.argval,))
        elif is_exiting(inst.opname):
            flowinfo.add_jump_inst(inst.offset, ())
    return flowinfo


def build_basicblocks(flowinfo: "FlowInfo", end_offset) -> "BlockMap":
    """
    Build a graph of basic-blocks
    """
    offsets = sorted(flowinfo.block_offsets)
    bbmap = BlockMap()
    for begin, end in zip(offsets, [*offsets[1:], end_offset]):
        term_offset = _prev_inst_offset(end)
        if term_offset not in flowinfo.jump_insts:
            # implicit jump
            targets = (end,)
            fallthrough = True
        else:
            targets = flowinfo.jump_insts[term_offset]
            fallthrough = False
        bb = Block(
            begin=begin, end=end, jump_targets=targets, fallthrough=fallthrough
        )
        bbmap.add_node(bb)
    return bbmap


@dataclass(frozen=True)
class FlowInfo:
    block_offsets: Set[int] = field(default_factory=set)
    """Marks starting offset of basic-block
    """

    jump_insts: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    """Contains jump instructions and their target offsets.
    """

    def add_jump_inst(self, offset, targets):
        for off in targets:
            self.block_offsets.add(off)
        self.jump_insts[offset] = tuple(targets)


@dataclass(frozen=True)
class Block:
    begin: int
    """The starting bytecode offset.
    """

    end: int
    """The bytecode offset immediate after the last bytecode of the block.
    """

    fallthrough: bool
    """Set to True when the block has no terminator. The control should just
    fallthrough to the next block.
    """

    jump_targets: Set[int]
    """The destination block offsets."""

    def is_exiting(self) -> bool:
        return not self.jump_targets

    def get_instructions(
        self, bcmap: Dict[int, dis.Instruction]
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

    def render_dot(self, g, node_offset: int, bcmap: "BlockMap"):
        instlist = self.get_instructions(bcmap)
        body = "\l".join(
            [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
        )
        g.node(hex(node_offset), shape="rect", label=body)


@dataclass(frozen=True)
class RegionBlock(Block):
    kind: str
    headers: Dict[int, Block]
    """The header of the region"""
    subregion: "BlockMap"
    """The subgraph excluding the headers
    """

    def render_dot(self, g, node_offset: int, bbmap: "BlockMap"):
        # render subgraph
        graph = self.get_full_graph()
        with g.subgraph(name=f"cluster_{node_offset}") as subg:
            color = 'blue'
            if self.kind == 'branch':
                color = 'green'
            subg.attr(color=color, label=self.kind)
            for k, node in graph.items():
                node.render_dot(subg, k, bbmap)
        # render edges within this region
        for k, node in graph.items():
            for dst in node.jump_targets:
                if dst in graph:
                    g.edge(hex(k), hex(dst))

    def get_full_graph(self):
        graph = ChainMap(self.subregion.graph, self.headers)
        return graph

@dataclass(frozen=True)
class BlockMap:
    graph: Dict[int, Block] = field(default_factory=dict)

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


def _next_inst_offset(offset: int) -> int:
    # Fix offset
    return offset + 2


def _prev_inst_offset(offset: int) -> int:
    # Fix offset
    return offset - 2


def compute_scc(bbmap: BlockMap) -> List[Set[int]]:
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

    bcmap = {inst.offset: inst for inst in bc}

    g = Digraph()
    # render nodes
    for k, node in bbmap.graph.items():
        node.render_dot(g, k, bcmap)
    # render edges
    for k, node in bbmap.graph.items():
        for dst in node.jump_targets:
            g.edge(hex(k), hex(dst))
    return g


def restructure_loop(bbmap: BlockMap):
    """Inplace restructuring of the given graph to extract loops using
    strongly-connected components
    """
    scc = compute_scc(bbmap)
    loops = [nodes for nodes in scc if len(nodes) > 1]
    _logger.debug("restructure_loop found %d loops in %s",
                  len(loops), bbmap.graph.keys())

    node: Block
    # extract loop
    for loop in loops:
        _logger.debug("loop nodes %s", loop)
        # find entries
        entries = set()
        headers = set()

        for node in _exclude_nodes(bbmap.graph, loop):
            nodes_jump_in_loop = set(bbmap.graph[node].jump_targets) & loop
            headers |= nodes_jump_in_loop
            if nodes_jump_in_loop:
                entries.add(node)

        # find exits
        exits = set()
        for node in loop:
            for outside in _exclude_nodes(bbmap.graph, loop):
                if outside in bbmap.graph[node].jump_targets:
                    exits.add(outside)

        # remove loop nodes
        insiders = {bbmap.graph.pop(k) for k in loop}

        assert len(headers) == 1, headers  # TODO join entries and exits
        [loop_head] = headers
        blk = RegionBlock(
            begin=loop_head,
            end=_next_inst_offset(loop_head),
            fallthrough=False,
            jump_targets=tuple(exits),
            kind="loop",
            subregion=BlockMap({node.begin: node for node in insiders
                                   if node.begin not in headers}),
            headers={node.begin: node for node in insiders
                     if node.begin in headers},
        )
        # process subregions
        restructure_loop(blk.subregion)
        bbmap.graph[loop_head] = blk


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
            subregion = set()
            branch_regions.append((bra_start, subregion))
            # a node is part of the branch if
            # - the start of the branch is a dominator; and,
            # - the tail of the branch is not a dominator
            for k, kdom in doms.items():
                if bra_start in kdom and end not in kdom:
                    subregion.add(k)

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
                    kind="branch",
                    headers=(),
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
                         immdoms: Dict[int, int],
                         postimmdoms: Dict[int, int]):
    for begin, node in bbmap.graph.items():
        if len(node.jump_targets) > 1:
            # found branch
            end = postimmdoms[begin]
            if immdoms[end] == begin:
                yield begin, end


def _imm_doms(doms: Dict[int, Set[int]]) -> Dict[int, int]:
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
