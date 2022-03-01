import dis
from collections import deque
from typing import Optional, Set, Tuple, Dict, List
from pprint import pprint
from dataclasses import dataclass, field


def parse_bytecode(code):
    bc = dis.Bytecode(code)
    out = bc.dis()
    print(out)

    end_offset = _next_inst_offset([inst.offset for inst in bc][-1])
    flowinfo = build_flowinfo(bc)
    print(flowinfo)
    bbmap = build_basicblocks(flowinfo, end_offset)
    pprint(bbmap.graph)
    # handle loop
    restructure_loop(bbmap)
    pprint(bbmap)
    # render
    render_dot(bc, bbmap).view()


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
    """
    Marks starting offset of basic-block
    """

    jump_insts: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    """
    Contains jump instructions and their target offsets.
    """

    def add_jump_inst(self, offset, targets):
        for off in targets:
            self.block_offsets.add(off)
        self.jump_insts[offset] = tuple(targets)


@dataclass(frozen=True)
class Block:
    begin: int
    end: int
    fallthrough: bool
    jump_targets: Set[int]

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


@dataclass(frozen=True)
class RegionBlock(Block):
    kind: str
    region_nodes: Dict[int, Block]


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
_uncond_jump = {"JUMP_ABSOLUTE"}
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
            return out

        def __iter__(self):
            return iter(self.graph.keys())

    return list(scc(GraphWrap(bbmap.graph)))


def render_dot(bc: dis.Bytecode, bbmap: BlockMap):
    from graphviz import Digraph

    bcmap = {inst.offset: inst for inst in bc}

    g = Digraph()
    # render nodes
    for k, node in bbmap.graph.items():
        instlist = node.get_instructions(bcmap)
        body = "\l".join(
            [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
        )
        g.node(hex(k), shape="rect", label=body)
    # render edges
    for k, node in bbmap.graph.items():
        for dst in node.jump_targets:
            g.edge(hex(k), hex(dst))
    return g


def restructure_loop(bbmap: BlockMap):
    scc = compute_scc(bbmap)
    loops = [nodes for nodes in scc if len(nodes) > 1]

    node: Block
    # extract loop
    for loop in loops:
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

        assert len(headers) == 1
        [loop_head] = headers
        blk = RegionBlock(
            begin=loop_head,
            end=_next_inst_offset(loop_head),
            fallthrough=False,
            jump_targets=tuple(exits),
            kind="loop",
            region_nodes={node.begin: node for node in insiders},
        )
        bbmap.graph[loop_head] = blk


def _exclude_nodes(iter, nodeset: Set):
    for it in iter:
        if it not in nodeset:
            yield it
