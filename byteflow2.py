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


@dataclass(frozen=True, order=True)
class SyntheticBranch(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SyntheticTail(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SyntheticExit(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SynthenticHead(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SynthenticReturn(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SyntheticLatch(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SyntheticExitingLatch(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SynthenticAssignment(ControlLabel):
    index: int


@dataclass(frozen=True, order=True)
class SyntheticForIter(ControlLabel):
    index: int


class ControlLabelGenerator():

    def __init__(self, index=0, variable=97):
        self.index = index
        self.variable = variable

    def new_index(self):
        ret = self.index
        self.index += 1
        return ret

    def new_variable(self):
        ret = chr(self.variable)
        self.variable += 1
        return ret


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
class ControlVariableBlock(BasicBlock):
    variable_assignment: dict


@dataclass(frozen=True)
class BranchBlock(BasicBlock):
    variable: str
    branch_value_table: dict

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        fallthrough = len(jump_targets) == 1
        old_branch_value_table = self.branch_value_table
        new_branch_value_table = {}
        for target in self.jump_targets:
            if target not in jump_targets:
                # ASSUMPTION: only one jump_target is being updated
                diff = jump_targets.difference(self.jump_targets)
                assert len(diff) == 1
                new_target = next(iter(diff))
                for k, v in old_branch_value_table.items():
                    if v == target:
                        new_branch_value_table[k] = new_target
            else:
                # copy all old values
                for k, v in old_branch_value_table.items():
                    if v == target:
                        new_branch_value_table[k] = v

        return replace(self,
                       fallthrough=fallthrough,
                       jump_targets=jump_targets,
                       branch_value_table=new_branch_value_table,
                       )


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
            bbmap.add_block(bb)
        return bbmap


@dataclass(frozen=True)
class BlockMap:
    """ Map of Labels to Blocks. """
    graph: Dict[Label, BasicBlock] = field(default_factory=dict)
    clg: ControlLabelGenerator = field(default_factory=ControlLabelGenerator,
                                       compare=False)

    def add_block(self, basicblock: BasicBlock):
        self.graph[basicblock.begin] = basicblock

    def insert_block(self, new_label: Label,
                     predecessors: Set[Label],
                     successors: Set[Label]):
        # initialize new block
        new_block = BasicBlock(begin=new_label,
                               end=ControlLabel("end"),
                               fallthrough=len(successors) == 1,
                               jump_targets=successors,
                               backedges=set()
                               )
        # add block to self
        self.add_block(new_block)
        # Replace any arcs from any of predecessors to any of successors with
        # an arc through the inserted block instead.
        for label in predecessors:
            block = self.graph.pop(label)
            jt = set(block.jump_targets)
            jt.difference_update(jt.intersection(successors))
            jt.add(new_label)
            self.add_block(block.replace_jump_targets(jump_targets=set(jt)))

    def insert_block_and_control_blocks(self, new_label: Label,
                                        predecessors: Set[Label],
                                        successors: Set[Label]):
        # name of the variable for this branching assignment
        branch_variable = self.clg.new_variable()
        # initial value of the assignment
        branch_variable_value = 0
        # store for the mapping from variable value to label
        branch_value_table = {}
        # Replace any arcs from any of predecessors to any of successors with
        # an arc through the to be inserted block instead.
        for label in predecessors:
            block = self.graph[label]
            jt = set(block.jump_targets)
            # Need to create synthetic assignments for each arc from a
            # predecessors to a successor and insert it between the predecessor
            # and the newly created block
            for s in jt.intersection(successors):
                synth_assign = SynthenticAssignment(self.clg.new_index())
                variable_assignment = {}
                variable_assignment[branch_variable] = branch_variable_value
                synth_assign_block = ControlVariableBlock(
                               begin=synth_assign,
                               end=ControlLabel("end"),
                               fallthrough=True,
                               jump_targets=set((new_label,)),
                               backedges=set(),
                               variable_assignment=variable_assignment,
                )
                # add block
                self.add_block(synth_assign_block)
                # update branching table
                branch_value_table[branch_variable_value] = s
                # update branching variable
                branch_variable_value += 1
                # remove previous successors
                jt.discard(s)
                # add the new assignment block to the jump targets
                jt.add(synth_assign)
            # finally, replace the jump_targets
            self.add_block(
                self.graph.pop(label).replace_jump_targets(jump_targets=jt))
        # initialize new block, which will hold the branching table
        new_block = BranchBlock(begin=new_label,
                                end=ControlLabel("end"),
                                fallthrough=len(successors) <= 1,
                                jump_targets=successors,
                                backedges=set(),
                                variable=branch_variable,
                                branch_value_table=branch_value_table,
                                )
        # add block to self
        self.add_block(new_block)

    def remove_blocks(self, labels: Set[Label]):
        for label in labels:
            del self.graph[label]

    def __getitem__(self, index):
        return self.graph[index]

    def __contains__(self, index):
        return index in self.graph

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

    def compute_scc_subgraph(self, subgraph) -> List[Set[Label]]:
        from scc import scc

        class GraphWrap:
            def __init__(self, graph, subgraph):
                self.graph = graph
                self.subgraph = subgraph

            def __getitem__(self, vertex):
                out = self.graph[vertex].jump_targets
                # Exclude node outside of the subgraph
                return [k for k in out if k in subgraph]

            def __iter__(self):
                return iter(self.graph.keys())

        return list(scc(GraphWrap(self.graph, subgraph)))

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
        # If the loop has no headers or entries, the only header is the head of
        # the CFG.
        if not headers:
            headers = {self.find_head()}
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
        # close if more than one is found
        if len(return_nodes) > 1:
            return_solo_label = SynthenticReturn(str(self.clg.new_index()))
            self.insert_block(return_solo_label, return_nodes, set())

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
            solo_exit_label = SyntheticExit(str(self.clg.new_index()))
            self.insert_block(solo_exit_label, tails, exits)
            return solo_tail_label, solo_exit_label

        if len(tails) >= 2 and len(exits) == 1:
            # join only tails
            solo_tail_label = SyntheticTail(str(self.clg.new_index()))
            solo_exit_label = next(iter(exits))
            self.insert_block(solo_tail_label, tails, exits)
            return solo_tail_label, solo_exit_label

        if len(tails) >= 2 and len(exits) >= 2:
            # join both tails and exits
            solo_tail_label = SyntheticTail(str(self.clg.new_index()))
            solo_exit_label = SyntheticExit(str(self.clg.new_index()))
            self.insert_block(solo_tail_label, tails, exits)
            self.insert_block(solo_exit_label, set((solo_tail_label,)), exits)
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
        for region in _iter_subregions(bbmap):
            restructure_loop(region.subregion)
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
        for region in _iter_subregions(bbmap):
            restructure_loop(region.subregion)
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


def loop_rotate_for_loop(bbmap: BlockMap, loop: Set[Label]):
    """ "Loop-rotate" a Python for loop.

    This function will convert a header controlled Python for loop into a
    tail-controlled Python foor loop. The approach taken here is to reuse the
    `FOR_ITER` bytecode as a loop guard. To implement this, the existing
    instruction is replicated into a `SyntheticForIter` block and the backedges
    are updated accordingly.

    Example
    -------

    Consider the following Python for loop:

    .. code-block::

        def foo(n):
            for i in range(n):
                pass


    This is the initial CFG.

      ┌─────────┐
      │  ENTRY  │
      └────┬────┘
           │
           │
           │
      ┌────▼────┐
    ┌─►FOR_ITER │
    │ └────┬────┘
    │      │
    │      ├─────────────┐
    │      │             │
    │ ┌────▼────┐   ┌────▼────┐
    └─┤  BODY   │   │  RETURN │
      └─────────┘   └─────────┘

    After loop-rotation, the backedge no longer points to the FOR_ITER and the
    SYNTHETIC for-iter block handles the loop condition. The initial FOR_ITER
    now serves as a loop-guard to determine if the loop should be entred or
    not.

      ┌─────────┐
      │  ENTRY  │
      └────┬────┘
           │
           │
           │
      ┌────▼────┐
      │FOR_ITER │
      └────┬────┘
           │
           ├─────────────┐
           │             │
      ┌────▼────┐        │
    ┌─►  BODY   │        │
    │ └────┬────┘        │
    │      │             │
    │      │             │
    │      │             │
    │ ┌────▼────┐   ┌────▼────┐
    └─┤SYNTHETIC├───►  RETURN │
      └─────────┘   └─────────┘

    """
    headers, entries = bbmap.find_headers_and_entries(loop)
    exiting_blocks, exit_blocks = bbmap.find_exits(loop)
    # Make sure there is only a single header, which is the FOR_ITER block.
    assert len(headers) == 1
    for_iter: Label = next(iter(headers))
    # Create the SyntheticForIter block and replicate the jump targets from the
    # original FOR_ITER block.
    synth_for_iter = SyntheticForIter(bbmap.clg.new_index())
    new_block = BasicBlock(begin=synth_for_iter,
                           end=ControlLabel("end"),
                           fallthrough=False,
                           jump_targets=set(bbmap[for_iter].jump_targets),
                           backedges=set()
                           )
    bbmap.add_block(new_block)

    # Remove the FOR_ITER from the set of loop blocks. 
    loop.remove(for_iter)
    # Rewire incoming edges for FOR_ITER to SyntheticForIter instead.
    for label in loop:
        block = bbmap.graph.pop(label)
        jt = set(block.jump_targets)
        if for_iter in jt:
            jt.remove(for_iter)
            jt.add(synth_for_iter)
        bbmap.add_block(block.replace_jump_targets(
                        jump_targets=jt))

    # Finally, add the SYNTHETIC_FOR_ITER to the loop.
    loop.add(synth_for_iter)

    ## find the new loop_head
    #for_iter_jump_targets = bbmap[for_iter].jump_targets
    #assert len(for_iter_jump_targets) == 2
    #llhjt_list = list(for_iter_jump_targets)
    #if llhjt_list[0] in loop:
    #    loop_head = llhjt_list[0]
    #    loop_exit = llhjt_list[1]
    #else:
    #    loop_head = llhjt_list[1]
    #    loop_exit = llhjt_list[0]


def loop_rotate(bbmap: BlockMap, loop: Set[Label]):
    """ Rotate loop.

    This will convert the loop into "loop canonical form".
    """
    headers, entries = bbmap.find_headers_and_entries(loop)
    exiting_blocks, exit_blocks = bbmap.find_exits(loop)
    assert len(entries) == 1
    headers_were_unified = False
    if len(headers) > 1:
        headers_were_unified = True
        solo_head_label = SynthenticHead(bbmap.clg.new_index())
        bbmap.insert_block_and_control_blocks(solo_head_label, entries, headers)
        loop.add(solo_head_label)
        loop_head: Label = solo_head_label
        # need to rewire all backedges
        #for label in loop:
        #    if label == solo_head_label:
        #        continue
        #    block = bbmap.graph.pop(label)
        #    jt = set(block.jump_targets)
        #    # remove any existing backedges that point to the headers
        #    jt.difference_update(headers)
        #    # add a backedge to the new solo header
        #    jt.add(solo_head_label)
        #    bbmap.add_block(block.replace_jump_targets(
        #                    jump_targets=jt))
    elif len(headers) == 1:
        assert len(headers) == 1  # TODO join entries
        # find the loop head
        loop_rotate_for_loop(bbmap, loop)
        headers, entries = bbmap.find_headers_and_entries(loop)
        exiting_blocks, exit_blocks = bbmap.find_exits(loop)
        loop_head: Label = next(iter(headers))
    else:
        raise Exception("unreachable")

    backedge_blocks = [block for block in loop
                       if headers.intersection(bbmap[block].jump_targets)]
    if len(backedge_blocks) == 1 and len(exiting_blocks) == 1:
        for label in loop:
            bbmap.add_block(
               bbmap.graph.pop(label).replace_backedge(loop_head))
        return headers, loop_head, next(iter(exiting_blocks)), next(iter(exit_blocks))

    # TODO: the synthetic exiting latch and synthetic exit need to be created
    # based on the state of the cfg
    synth_exiting_latch = SyntheticExitingLatch(bbmap.clg.new_index())
    synth_exit = SyntheticExit(bbmap.clg.new_index())

    # the entry variable and the exit variable will be re-used
    if headers_were_unified:
        exit_variable = bbmap[solo_head_label].variable
    else:
        exit_variable = bbmap.clg.new_variable()
    #exit_variable = bbmap.clg.new_variable()
    backedge_variable = bbmap.clg.new_variable()
    exit_value_table = dict(((i, j) for i, j in enumerate(exit_blocks)))
    backedge_value_table = dict((i, j) for i, j in enumerate((loop_head,
                                                              synth_exit)))
    if headers_were_unified:
        header_value_table = bbmap[solo_head_label].branch_value_table
    else:
        header_value_table = {}

    def reverse_lookup(d, value):
        for k, v in d.items():
            if v == value:
                return k
        else:
            return "UNUSED"

    new_blocks = set()
    doms = _doms(bbmap)
    for label in loop:
        if label in exiting_blocks or label in backedge_blocks:
            new_jt = set()
            for jt in bbmap[label].jump_targets:
                if jt in exit_blocks:
                    synth_assign = SynthenticAssignment(bbmap.clg.new_index())
                    new_blocks.add(synth_assign)
                    variable_assignment = dict((
                        (backedge_variable,
                         reverse_lookup(backedge_value_table, synth_exit)),
                        (exit_variable,
                         reverse_lookup(exit_value_table, jt)
                         ),
                    ))
                    synth_assign_block = ControlVariableBlock(
                               begin=synth_assign,
                               end=ControlLabel("end"),
                               fallthrough=True,
                               jump_targets=set((synth_exiting_latch,)),
                               backedges=set(),
                               variable_assignment=variable_assignment,
                    )
                    bbmap.add_block(synth_assign_block)
                    new_jt.add(synth_assign)
                elif jt in headers and label not in doms[jt]:
                    synth_assign = SynthenticAssignment(bbmap.clg.new_index())
                    new_blocks.add(synth_assign)
                    variable_assignment = dict((
                        (backedge_variable,
                         reverse_lookup(backedge_value_table, loop_head)),
                        (exit_variable,
                         reverse_lookup(header_value_table, jt)
                         ),
                    ))
                    # update the backedge block
                    block = bbmap.graph.pop(label)
                    jt = set(block.jump_targets)
                    # remove any existing backedges that point to the headers,
                    # no need to add a backedge, since it will be contained in
                    # the SyntheticExitingLatch later on.
                    jt.difference_update(headers)
                    bbmap.add_block(block.replace_jump_targets(
                                    jump_targets=jt))
                    synth_assign_block = ControlVariableBlock(
                               begin=synth_assign,
                               end=ControlLabel("end"),
                               fallthrough=True,
                               jump_targets=set((synth_exiting_latch,)),
                               backedges=set(),
                               variable_assignment=variable_assignment,
                    )
                    bbmap.add_block(synth_assign_block)
                    new_jt.add(synth_assign)
                else:
                    new_jt.add(jt)
            # finally, replace the jump_targets
            bbmap.add_block(
                bbmap.graph.pop(label).replace_jump_targets(jump_targets=new_jt))

    loop.update(new_blocks)

    synth_exiting_latch_block = BranchBlock(
        begin=synth_exiting_latch,
        end=ControlLabel("end"),
        fallthrough=False,
        jump_targets=set((synth_exit,)),
        backedges=set((loop_head, )),
        variable=backedge_variable,
        branch_value_table=backedge_value_table,
        )
    loop.add(synth_exiting_latch)
    bbmap.add_block(synth_exiting_latch_block)

    synth_exit_block = BranchBlock(
        begin=synth_exit,
        end=ControlLabel("end"),
        fallthrough=False,
        jump_targets=exit_blocks,
        backedges=set(),
        variable=exit_variable,
        branch_value_table=exit_value_table,
        )
    bbmap.add_block(synth_exit_block)

    return headers, loop_head, synth_exiting_latch, synth_exit

    # SyntheticExitingLatch
    # SyntheticExit
    # Will need multivariable assignment blocks
    # Make two variables, keep track of their state and iterate

    # THIS IS FOR PYTHON FOR LOOPS
    ## the loop head should have two jump targets
    #loop_head_jt = bbmap[loop_head].jump_targets
    #assert len(loop_head_jt) == 2
    #llhjt_list = list(loop_head_jt)
    #if llhjt_list[0] in loop:
    #    loop_body_start = llhjt_list[0]
    #    loop_head_exit = llhjt_list[1]
    #else:
    #    loop_body_start = llhjt_list[1]
    #    loop_head_exit = llhjt_list[0]

    #loop_head_jt = bbmap[loop_head].jump_targets

    ## find the backedge that points to the loop head
    #backedge_blocks = [block for block in loop
    #                   if loop_head in bbmap[block].jump_targets]
    #if len(backedge_blocks) == 1:
    #    backedge_block = backedge_blocks[0]
    #    # TODO replace jump_target, not overwrite
    #    bbmap.add_block(bbmap.graph.pop(backedge_block).replace_jump_targets(
    #        jump_targets=loop_head_jt))
    #elif len(backedge_blocks) > 1:
    #    # create new backedgeblock, that points to the loop_body_start
    #    synth_backedge_label = SyntheticLatch(bbmap.clg.new_index())
    #    synth_backedge_block = BasicBlock(begin=synth_backedge_label,
    #                                end="end",
    #                                fallthrough=False,
    #                                #jump_targets=set((loop_body_start,loop_head_exit)),
    #                                jump_targets=set((loop_head,)),
    #                                backedges=set()
    #                                )
    #    bbmap.add_block(synth_backedge_block)
    #    for label in backedge_blocks:
    #        block = bbmap.graph.pop(label)
    #        jt = set(block.jump_targets)
    #        jt.discard(loop_head)
    #        jt.add(synth_backedge_label)
    #        bbmap.add_block(block.replace_jump_targets(
    #                        jump_targets=jt))
    #    loop.add(synth_backedge_label)
    #else:
    #    raise Exception("unreachable")


    #headers, entries = bbmap.find_headers_and_entries(loop)
    #pre_exits, post_exits = bbmap.find_exits(loop)
    ##pre_exit_label, post_exit_label = bbmap.join_tails_and_exits(pre_exits,
    ##                                                             post_exits)
    ##loop.add(pre_exit_label)

    ## recompute the scc
    #subgraph_bbmap = BlockMap({label: bbmap[label] for label in loop})
    #new_loops = []
    #for scc in subgraph_bbmap.compute_scc():
    #    if len(scc) == 1:
    #        solo_block = next(iter(scc))
    #        if solo_block in bbmap[solo_block].jump_targets:
    #            new_loops.append(scc)
    #    else:
    #        new_loops.append(scc)
    #assert(len(new_loops) == 1)
    #new_loop = next(iter(new_loops))
    #loop.clear()
    #loop.update(new_loop)

    #headers, entries = bbmap.find_headers_and_entries(loop)
    #pre_exits, post_exits = bbmap.find_exits(loop)
    #pre_exit_label = next(iter(pre_exits))
    #post_exit_label = next(iter(post_exits))
    #pre_exit_label, post_exit_label = bbmap.join_tails_and_exits(pre_exits,
    #                                                             post_exits)
    #loop.add(pre_exit_label)
    #loop_head = next(iter(headers))

    ## fixup backedges
    #for label in loop:
    #    bbmap.add_block(
    #        bbmap.graph.pop(label).replace_backedge(loop_head))

    #loop_body_start = next(iter(bbmap[loop_head].jump_targets))

    #return headers, loop_head, loop_body_start, pre_exit_label, post_exit_label


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
        headers, loop_head, pre_exit_label, post_exit_label = loop_rotate(bbmap, loop)
        extract_region(bbmap, loop, "loop")



def _restructure_branch(bbmap: BlockMap):
    print("restructure_branch", bbmap.graph)
    doms = _doms(bbmap)
    postdoms = _post_doms(bbmap)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    recursive_subregions = []
    # TODO what are begin and end exactly? The assumtion is that they are
    # unique, is this true?
    regions = [r for r in _iter_branch_regions(bbmap, immdoms, postimmdoms)]
    for begin, end in regions:
        _logger.debug("branch region: %s -> %s", begin, end)
        # Extract the head subregion
        head, head_region_blocks = find_head_blocks(bbmap, begin)
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
                    synthetic_branch_block_label = SyntheticBranch(bbmap.clg.new_index())
                    synthetic_branch_block_block = BasicBlock(
                            begin=synthetic_branch_block_label,
                            end="end",
                            jump_targets=(b,),
                            fallthrough=True,
                            backedges=(),
                            )
                    bbmap.add_block(synthetic_branch_block_block)
                    new_jump_targets.append(synthetic_branch_block_label)
                    jump_targets_changed = True
                else:
                    # otherwise just re-use the existing block
                    new_jump_targets.append(b)

        # update the begin block with new jump_targets
        if jump_targets_changed:
            bbmap.add_block(
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

        if len(headers) > 1:
            # The tail region will have multiple incoming edges, but these need
            # to be funnled through a unique header. Hence we simply insert a
            # new block in case the tail subregion has multiple headers.
            synth_head_label = SynthenticHead(bbmap.clg.new_index())
            bbmap.insert_block(synth_head_label, entries, headers)
            tail_subregion.add(synth_head_label)
        else:
            synth_head_label = next(iter(headers))

        subgraph = BlockMap(clg=bbmap.clg)
        for block in tail_subregion:
            subgraph.add_block(bbmap.graph[block])
        subregion = RegionBlock(
            begin=synth_head_label,
            end="end",
            fallthrough=False,
            jump_targets=(),
            backedges=(),
            kind="tail",
            headers=headers,
            subregion=subgraph,
            exit=None,
        )
        bbmap.remove_blocks(tail_subregion)
        bbmap.graph[synth_head_label] = subregion

        #if subregion.subregion.graph:
        #    recursive_subregions.append(subregion.subregion)
        #break
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
                inner_nodes.add(pre_exit_label)
                #break

                if isinstance(bbmap[pre_exit_label], RegionBlock):
                    pre_exit_label = bbmap[pre_exit_label].exit

                subgraph = BlockMap(clg=bbmap.clg)
                for k in inner_nodes:
                    subgraph.add_block(bbmap.graph[k])

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

                #if subregion.subregion.graph:
                #    recursive_subregions.append(subregion.subregion)
        #break

    # recurse into subregions as necessary
    #for region in recursive_subregions:
    #    restructure_branch(region)


def find_head_blocks(bbmap: BlockMap, begin: Label) -> Set[Label]:
    head = bbmap.find_head()
    head_region_blocks = set()
    current_block = head
    # Start at the head block and traverse the graph linearly until
    # reaching the begin block.
    while True:
        head_region_blocks.add(current_block)
        if current_block == begin:
            break
        else:
            jt = bbmap.graph[current_block].jump_targets
            assert len(jt) == 1
            current_block = next(iter(jt))
    return head_region_blocks


def find_branch_regions(bbmap: BlockMap, begin: Label, end: Label) \
            -> Set[Label]:
    # identify branch regions
    doms = _doms(bbmap)
    postdoms = _post_doms(bbmap)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    branch_regions = []
    jump_targets = bbmap.graph[begin].jump_targets
    for bra_start in jump_targets:
        for jt in jump_targets:
            if jt != bra_start and bbmap.is_reachable_dfs(jt, bra_start):
                branch_regions.append(tuple())
                break
        else:
            sub_keys: Set[BCLabel] = set()
            branch_regions.append((bra_start, sub_keys))
            # a node is part of the branch if
            # - the start of the branch is a dominator; and,
            # - the tail of the branch is not a dominator
            for k, kdom in doms.items():
                if bra_start in kdom and end not in kdom:
                    sub_keys.add(k)
    return branch_regions


def _find_branch_regions(bbmap: BlockMap, begin: Label, end: Label) \
            -> Set[Label]:
    # identify branch regions
    branch_regions = []
    for bra_start in bbmap[begin].jump_targets:
        region = []
        region.append(bra_start)
    return branch_regions


def find_tail_blocks(bbmap: BlockMap, begin: Set[Label], head_region_blocks, branch_regions):
    tail_subregion = set((b for b in bbmap.graph.keys()))
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


def extract_region(bbmap, region_blocks, region_kind):
    headers, entries = bbmap.find_headers_and_entries(region_blocks)
    exiting_blocks, exit_blocks = bbmap.find_exits(region_blocks)
    assert len(headers) == 1
    assert len(exiting_blocks) == 1
    region_header = next(iter(headers))
    region_exiting = next(iter(exiting_blocks))

    head_subgraph = BlockMap({label: bbmap.graph[label]
                             for label in region_blocks},
                             clg=bbmap.clg)

    if isinstance(bbmap[region_exiting], RegionBlock):
        region_exit = bbmap[region_exiting].exit
    else:
        region_exit = region_exiting

    subregion = RegionBlock(
        begin=region_header,
        end=region_exiting,
        fallthrough=len(bbmap[region_exiting].jump_targets) > 1,
        jump_targets=bbmap[region_exiting].jump_targets,
        backedges=(),
        kind=region_kind,
        headers=headers,
        subregion=head_subgraph,
        exit=region_exit,
    )
    bbmap.remove_blocks(region_blocks)
    bbmap.graph[region_header] = subregion


def restructure_branch(bbmap: BlockMap):
    print("restructure_branch", bbmap.graph)
    doms = _doms(bbmap)
    postdoms = _post_doms(bbmap)
    postimmdoms = _imm_doms(postdoms)
    immdoms = _imm_doms(doms)
    regions = [r for r in _iter_branch_regions(bbmap, immdoms, postimmdoms)]

    # Early exit when no branching regions are found.
    # TODO: the whole graph should become a linear mono head
    if not regions:
        return

    # Compute initial regions.
    begin, end = regions[0]
    head_region_blocks = find_head_blocks(bbmap, begin)
    branch_regions = find_branch_regions(bbmap, begin, end)
    tail_region_blocks = find_tail_blocks(bbmap, begin, head_region_blocks, branch_regions)

    # Unify headers of tail subregion if need be.
    headers, entries = bbmap.find_headers_and_entries(tail_region_blocks)
    if len(headers) > 1:
        end = SynthenticHead(bbmap.clg.new_index())
        bbmap.insert_block_and_control_blocks(end, entries, headers)

    # Recompute regions.
    head_region_blocks = find_head_blocks(bbmap, begin)
    branch_regions = find_branch_regions(bbmap, begin, end)
    tail_region_blocks = find_tail_blocks(bbmap, begin, head_region_blocks, branch_regions)

    # Branch region processing:
    # Close any open branch regions by inserting a SyntheticTail.
    # Populate any empty branch regions by inserting a SyntheticBranch.
    for region in branch_regions:
        if region:
            bra_start, inner_nodes = region
            if inner_nodes:
                # Insert SyntheticTail
                exiting_blocks, _ = bbmap.find_exits(inner_nodes)
                tail_headers, _ = bbmap.find_headers_and_entries(tail_region_blocks)
                _, _ = bbmap.join_tails_and_exits(exiting_blocks, tail_headers)

        else:
            # Insert SyntheticBranch
            tail_headers, _ = bbmap.find_headers_and_entries(tail_region_blocks)
            synthetic_branch_block_label = SyntheticBranch(bbmap.clg.new_index())
            bbmap.insert_block(synthetic_branch_block_label, {begin}, tail_headers)

    # Recompute regions.
    head_region_blocks = find_head_blocks(bbmap, begin)
    branch_regions = find_branch_regions(bbmap, begin, end)
    tail_region_blocks = find_tail_blocks(bbmap, begin, head_region_blocks, branch_regions)

    # extract subregions
    extract_region(bbmap, head_region_blocks, "head")
    for region in branch_regions:
        if region:
            bra_start, inner_nodes = region
            if inner_nodes:
                extract_region(bbmap, inner_nodes, "branch")
    extract_region(bbmap, tail_region_blocks, "tail")


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
            body = label.__class__.__name__ + ": " + str(label.index)
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_control_variable_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        if isinstance(label, ControlLabel):
            body = label.__class__.__name__ + ": " + str(label.index) + '\n'
            body += "\n".join((f"{k} = {v}" for k, v in
                               block.variable_assignment.items()))
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_branching_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        if isinstance(label, ControlLabel):
            def find_index(v):
                if hasattr(v, "offset"):
                    return v.offset
                if hasattr(v, "index"):
                    return v.index
            body = label.__class__.__name__ + ": " + str(label.index) + '\n'
            body += f"variable: {block.variable}\n"
            body += "\n".join((f"{k}=>{find_index(v)}" for k, v in block.branch_value_table.items()))
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        if type(block) == BasicBlock:
            self.render_basic_block(digraph, label, block)
        elif type(block) == ControlVariableBlock:
            self.render_control_variable_block(digraph, label, block)
        elif type(block) == BranchBlock:
            self.render_branching_block(digraph, label, block)
        elif type(block) == RegionBlock:
            self.render_region_block(digraph, label, block)
        else:
            raise Exception("unreachable")

    def render_edges(self, blocks: Dict[Label, BasicBlock]):
        for label, block in blocks.items():
            for dst in block.jump_targets:
                if dst in blocks:
                    if type(block) in (BasicBlock,
                                       ControlVariableBlock,
                                       BranchBlock):
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
