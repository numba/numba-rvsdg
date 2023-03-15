import dis
from copy import deepcopy
from collections import ChainMap, defaultdict
from typing import (Set, Tuple, Dict, List, Sequence, Iterator)
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
    index: int
    ...


@dataclass(frozen=True, order=True)
class PythonBytecodeLabel(Label):
    pass


@dataclass(frozen=True, order=True)
class ControlLabel(Label):
    pass


@dataclass(frozen=True, order=True)
class SyntheticBranch(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticTail(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticExit(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticHead(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticReturn(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticLatch(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticExitingLatch(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SynthenticAssignment(ControlLabel):
    pass


@dataclass(frozen=True, order=True)
class SyntheticForIter(ControlLabel):
    pass


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
    label: Label
    """The corresponding Label for this block.  """

    _jump_targets: Tuple[Label]
    """Jump targets (branch destinations) for this block"""

    backedges: Tuple[Label]
    """Backedges for this block."""

    @property
    def is_exiting(self) -> bool:
        return not self.jump_targets

    @property
    def fallthrough(self) ->bool:
        return len(self._jump_targets) == 1

    @property
    def jump_targets(self) -> Tuple[Label]:
        acc = []
        for j in self._jump_targets:
            if j not in self.backedges:
                acc.append(j)
        return tuple(acc)


    def replace_backedge(self, target: Label) -> "BasicBlock":
        if target in self.jump_targets:
            assert not self.backedges
            return replace(self, backedges=(target,))
        return self

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        return replace(self, _jump_targets=jump_targets)

@dataclass(frozen=True)
class PythonBytecodeBlock(BasicBlock):
    begin: int
    """The starting bytecode offset.
    """

    end: int
    """The bytecode offset immediate after the last bytecode of the block.
    """

    def get_instructions(
        self, bcmap: Dict[int, dis.Instruction]
    ) -> List[dis.Instruction]:
        begin = self.begin
        end = self.end
        it = begin
        out = []
        while it < end:
            # Python 3.11 hack: account for gaps in the bytecode sequence
            try:
                out.append(bcmap[it])
            except KeyError:
                pass
            finally:
                it = _next_inst_offset(it)

        return out


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
                diff = set(jump_targets).difference(self.jump_targets)
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
                       _jump_targets=jump_targets,
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
    "JUMP_IF_FALSE_OR_POP",
    "POP_JUMP_IF_TRUE",
    "JUMP_IF_TRUE_OR_POP",
    "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_FORWARD_IF_NOT_NONE",
    "POP_JUMP_BACKWARD_IF_NOT_NONE",
    "POP_JUMP_FORWARD_IF_NONE",
    "POP_JUMP_BACKWARD_IF_NONE",

}
_uncond_jump = {"JUMP_ABSOLUTE", "JUMP_FORWARD", "JUMP_BACKWARD"}
_terminating = {"RETURN_VALUE"}


def is_conditional_jump(opname: str) -> bool:
    return opname in _cond_jump


def is_unconditional_jump(opname: str) -> bool:
    return opname in _uncond_jump


def is_exiting(opname: str) -> bool:
    return opname in _terminating


def _next_inst_offset(offset: int) -> int:
    # Fix offset
    assert isinstance(offset, int)
    return offset + 2


def _prev_inst_offset(offset: int) -> int:
    # Fix offset
    assert isinstance(offset, int)
    return offset - 2


@dataclass()
class FlowInfo:
    """ FlowInfo converts Bytecode into a ByteFlow object (CFG). """

    block_offsets: Set[int] = field(default_factory=set)
    """Marks starting offset of basic-block
    """

    jump_insts: Dict[int, Tuple[int, ...]] = field(default_factory=dict)
    """Contains jump instructions and their target offsets.
    """

    last_offset: int = field(default=0)
    """Offset of the last bytecode instruction.
    """

    clg: ControlLabelGenerator = field(default_factory=ControlLabelGenerator,
                                       compare=False)

    def _add_jump_inst(self, offset: int, targets: Sequence[int]):
        """Add jump instruction to FlowInfo."""
        for off in targets:
            assert isinstance(off, int)
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
                flowinfo.block_offsets.add(inst.offset)
            # Handle by op
            if is_conditional_jump(inst.opname):
                flowinfo._add_jump_inst(
                        inst.offset, (_next_inst_offset(inst.offset), inst.argval))
            elif is_unconditional_jump(inst.opname):
                flowinfo._add_jump_inst(inst.offset, (inst.argval,))
            elif is_exiting(inst.opname):
                flowinfo._add_jump_inst(inst.offset, ())

        flowinfo.last_offset = inst.offset
        return flowinfo

    def build_basicblocks(self: "FlowInfo", end_offset=None) -> "BlockMap":
        """
        Build a graph of basic-blocks
        """
        offsets = sorted(self.block_offsets)
        # enumerate labels
        labels = dict((offset, PythonBytecodeLabel(self.clg.new_index())) for offset in offsets)
        if end_offset is None:
            end_offset = _next_inst_offset(self.last_offset)
        bbmap = BlockMap(graph={}, clg=self.clg)
        for begin, end in zip(offsets, [*offsets[1:], end_offset]):
            label = labels[begin]
            targets: Tuple[Label, ...]
            term_offset = _prev_inst_offset(end)
            if term_offset not in self.jump_insts:
                # implicit jump
                targets = (labels[end],)
            else:
                targets = tuple(labels[o] for o in self.jump_insts[term_offset])
            block = PythonBytecodeBlock(
                label=label,
                begin=begin,
                end=end,
                _jump_targets=targets,
                backedges=(),
                )
            bbmap.add_block(block)
        return bbmap


@dataclass(frozen=True)
class BlockMap:
    """ Map of Labels to Blocks. """
    graph: Dict[Label, BasicBlock] = field(default_factory=dict)
    clg: ControlLabelGenerator = field(default_factory=ControlLabelGenerator,
                                       compare=False)

    def __getitem__(self, index):
        return self.graph[index]

    def __contains__(self, index):
        return index in self.graph

    def exclude_blocks(self, exclude_blocks: Set[Label]) -> Iterator[Label]:
        """Iterator over all nodes not in exclude_blocks. """
        for block in self.graph:
            if block not in exclude_blocks:
                yield block

    def find_head(self) -> Label:
        """Find the head block of the CFG.

        Assuming the CFG is closed, this will find the block
        that no other blocks are pointing to.

        """
        heads = set(self.graph.keys())
        for label in self.graph.keys():
            block = self.graph[label]
            for jt in block.jump_targets:
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
        """
        Strongly-connected component for detecting loops inside a subgraph.
        """
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

    def find_headers_and_entries(self, subgraph: Set[Label]) \
            -> Tuple[Set[Label], Set[Label]]:
        """Find entries and headers in a given subgraph.

        Entries are blocks outside the subgraph that have an edge pointing to
        the subgraph headers. Headers are blocks that are part of the strongly
        connected subset and that have incoming edges from outside the
        subgraph. Entries point to headers and headers are pointed to by
        entries.

        """
        outside: Label
        entries: Set[Label] = set()
        headers: Set[Label] = set()

        for outside in self.exclude_blocks(subgraph):
            nodes_jump_in_loop = subgraph.intersection(
                self.graph[outside].jump_targets)
            headers.update(nodes_jump_in_loop)
            if nodes_jump_in_loop:
                entries.add(outside)
        # If the loop has no headers or entries, the only header is the head of
        # the CFG.
        if not headers:
            headers = {self.find_head()}
        return headers, entries

    def find_exiting_and_exits(self, subgraph: Set[Label]) \
            -> Tuple[Set[Label], Set[Label]]:
        """Find exiting and exit blocks in a given subgraph.

        Existing blocks are blocks inside the subgraph that have edges to
        blocks outside of the subgraph. Exit blocks are blocks outside the
        subgraph that have incoming edges from within the subgraph. Exiting
        blocks point to exits and exits and pointed to by exiting blocks.

        """
        inside: Label
        exiting: Set[Label] = set()
        exits: Set[Label] = set()
        for inside in subgraph:
            # any node inside that points outside the loop
            for jt in self.graph[inside].jump_targets:
                if jt not in subgraph:
                    exiting.add(inside)
                    exits.add(jt)
            # any returns
            if self.graph[inside].is_exiting:
                exiting.add(inside)
        return exiting, exits

    def is_reachable_dfs(self, begin: Label, end: Label): # -> TypeGuard:
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

    def add_block(self, basicblock: BasicBlock):
        self.graph[basicblock.label] = basicblock

    def remove_blocks(self, labels: Set[Label]):
        for label in labels:
            del self.graph[label]

    def insert_block(self, new_label: Label,
                     predecessors: Set[Label],
                     successors: Set[Label]):
        # TODO: needs a diagram and documentaion
        # initialize new block
        new_block = BasicBlock(label=new_label,
                               _jump_targets=successors,
                               backedges=set()
                               )
        # add block to self
        self.add_block(new_block)
        # Replace any arcs from any of predecessors to any of successors with
        # an arc through the inserted block instead.
        for label in predecessors:
            block = self.graph.pop(label)
            jt = list(block.jump_targets)
            if successors:
                for s in successors:
                    if s in jt:
                        if new_label not in jt:
                            jt[jt.index(s)] = new_label
                        else:
                            jt.pop(jt.index(s))
            else:
                jt.append(new_label)
            self.add_block(block.replace_jump_targets(jump_targets=tuple(jt)))

    def insert_block_and_control_blocks(self, new_label: Label,
                                        predecessors: Set[Label],
                                        successors: Set[Label]):
        # TODO: needs a diagram and documentaion
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
            jt = list(block.jump_targets)
            # Need to create synthetic assignments for each arc from a
            # predecessors to a successor and insert it between the predecessor
            # and the newly created block
            for s in set(jt).intersection(successors):
                synth_assign = SynthenticAssignment(self.clg.new_index())
                variable_assignment = {}
                variable_assignment[branch_variable] = branch_variable_value
                synth_assign_block = ControlVariableBlock(
                               label=synth_assign,
                               _jump_targets=(new_label,),
                               backedges=(),
                               variable_assignment=variable_assignment,
                )
                # add block
                self.add_block(synth_assign_block)
                # update branching table
                branch_value_table[branch_variable_value] = s
                # update branching variable
                branch_variable_value += 1
                # replace previous successor with synth_assign
                jt[jt.index(s)] = synth_assign
            # finally, replace the jump_targets
            self.add_block(
                self.graph.pop(label).replace_jump_targets(
                    jump_targets=tuple(jt)))
        # initialize new block, which will hold the branching table
        new_block = BranchBlock(label=new_label,
                                _jump_targets=tuple(successors),
                                backedges=set(),
                                variable=branch_variable,
                                branch_value_table=branch_value_table,
                                )
        # add block to self
        self.add_block(new_block)

    def join_returns(self):
        """ Close the CFG.

        A closed CFG is a CFG with a unique entry and exit node that have no
        predescessors and no successors respectively.
        """
        # for all nodes that contain a return
        return_nodes = [node for node in self.graph
                        if self.graph[node].is_exiting]
        # close if more than one is found
        if len(return_nodes) > 1:
            return_solo_label = SyntheticReturn(str(self.clg.new_index()))
            self.insert_block(return_solo_label, return_nodes, tuple())

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
    exiting_blocks, exit_blocks = bbmap.find_exiting_and_exits(loop)
    # Make sure there is only a single header, which is the FOR_ITER block.
    assert len(headers) == 1
    for_iter: Label = next(iter(headers))
    # Create the SyntheticForIter block and replicate the jump targets from the
    # original FOR_ITER block.
    synth_for_iter = SyntheticForIter(str(bbmap.clg.new_index()))
    new_block = BasicBlock(label=synth_for_iter,
                           _jump_targets=bbmap[for_iter].jump_targets,
                           backedges=()
                           )
    bbmap.add_block(new_block)

    # Remove the FOR_ITER from the set of loop blocks.
    loop.remove(for_iter)
    # Rewire incoming edges for FOR_ITER to SyntheticForIter instead.
    for label in loop:
        block = bbmap.graph.pop(label)
        jt = list(block.jump_targets)
        if for_iter in jt:
            jt[jt.index(for_iter)] = synth_for_iter
        bbmap.add_block(block.replace_jump_targets(
                        jump_targets=tuple(jt)))

    # Finally, add the SYNTHETIC_FOR_ITER to the loop.
    loop.add(synth_for_iter)


def loop_rotate(bbmap: BlockMap, loop: Set[Label]):
    """ Rotate loop.

    This will convert the loop into "loop canonical form".
    """
    headers, entries = bbmap.find_headers_and_entries(loop)
    exiting_blocks, exit_blocks = bbmap.find_exiting_and_exits(loop)
    #if len(entries) == 0:
    #    return
    assert len(entries) == 1
    headers_were_unified = False
    if len(loop) == 1:
        # Probably a Python while loop, only find the loop_head
        loop_head: Label = next(iter(loop))
    elif len(headers) > 1:
        headers_were_unified = True
        solo_head_label = SyntheticHead(bbmap.clg.new_index())
        bbmap.insert_block_and_control_blocks(
            solo_head_label, entries, headers)
        loop.add(solo_head_label)
        loop_head: Label = solo_head_label
    elif len(headers) == 1:
        # TODO join entries
        # Probably a Python for loop
        loop_rotate_for_loop(bbmap, loop)
        headers, entries = bbmap.find_headers_and_entries(loop)
        exiting_blocks, exit_blocks = bbmap.find_exiting_and_exits(loop)
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
            new_jt = list(bbmap[label].jump_targets)
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
                               label=synth_assign,
                               _jump_targets=(synth_exiting_latch,),
                               backedges=(),
                               variable_assignment=variable_assignment,
                    )
                    bbmap.add_block(synth_assign_block)
                    new_jt[new_jt.index(jt)] = synth_assign
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
                    jts = list(block.jump_targets)
                    # remove any existing backedges that point to the headers,
                    # no need to add a backedge, since it will be contained in
                    # the SyntheticExitingLatch later on.
                    for h in headers:
                        if h in jts:
                            jts.remove(h)
                    bbmap.add_block(block.replace_jump_targets(
                                    jump_targets=tuple(jts)))
                    synth_assign_block = ControlVariableBlock(
                               label=synth_assign,
                               _jump_targets=(synth_exiting_latch,),
                               backedges=(),
                               variable_assignment=variable_assignment,
                    )
                    bbmap.add_block(synth_assign_block)
                    new_jt[new_jt.index(jt)] = synth_assign
            # finally, replace the jump_targets
            bbmap.add_block(
                bbmap.graph.pop(label).replace_jump_targets(jump_targets=tuple(new_jt)))

    loop.update(new_blocks)

    synth_exiting_latch_block = BranchBlock(
        label=synth_exiting_latch,
        _jump_targets=(synth_exit, loop_head),
        backedges=(loop_head, ),
        variable=backedge_variable,
        branch_value_table=backedge_value_table,
        )
    loop.add(synth_exiting_latch)
    bbmap.add_block(synth_exiting_latch_block)

    synth_exit_block = BranchBlock(
        label=synth_exit,
        _jump_targets=tuple(exit_blocks),
        backedges=(),
        variable=exit_variable,
        branch_value_table=exit_value_table,
        )
    bbmap.add_block(synth_exit_block)


def restructure_loop(bbmap: BlockMap):
    """Inplace restructuring of the given graph to extract loops using
    strongly-connected components
    """
    # obtain a List of Sets of Labels, where all labels in each set are strongly
    # connected, i.e. all reachable from one another by traversing the subset
    scc: List[Set[Label]] = bbmap.compute_scc()
    # loops are defined as strongly connected subsets who have more than a
    # single label and single label loops that point back to to themselves.
    loops: List[Set[Label]] = [nodes for nodes in scc
                               if len(nodes) > 1
                               or next(iter(nodes)) in bbmap[next(iter(nodes))].jump_targets]
                            
    _logger.debug("restructure_loop found %d loops in %s",
                  len(loops), bbmap.graph.keys())
    # rotate and extract loop
    for loop in loops:
        loop_rotate(bbmap, loop)
        extract_region(bbmap, loop, "loop")


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
            sub_keys: Set[PythonBytecodeLabel] = set()
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
    exiting_blocks, exit_blocks = bbmap.find_exiting_and_exits(region_blocks)
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
        label=region_header,
        _jump_targets=bbmap[region_exiting].jump_targets,
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
        end = SyntheticHead(bbmap.clg.new_index())
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
                exiting_blocks, _ = bbmap.find_exiting_and_exits(inner_nodes)
                tail_headers, _ = bbmap.find_headers_and_entries(tail_region_blocks)
                _, _ = bbmap.join_tails_and_exits(exiting_blocks, tail_headers)

        else:
            # Insert SyntheticBranch
            tail_headers, _ = bbmap.find_headers_and_entries(tail_region_blocks)
            synthetic_branch_block_label = SyntheticBranch(str(bbmap.clg.new_index()))
            bbmap.insert_block(synthetic_branch_block_label, (begin,), tail_headers)

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
