import dis
from typing import Set, Tuple, Dict, List, Iterator
from dataclasses import dataclass, field

from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    ControlVariableBlock,
    BranchBlock,
)
from numba_rvsdg.core.datastructures.labels import (
    Label,
    ControlLabelGenerator,
    SynthenticAssignment,
    SyntheticExit,
    SyntheticTail,
    SyntheticReturn,
)


@dataclass(frozen=True)
class BlockMap:
    """ Map of Labels to Blocks. """

    graph: Dict[Label, BasicBlock] = field(default_factory=dict)
    clg: ControlLabelGenerator = field(
        default_factory=ControlLabelGenerator, compare=False
    )

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
        from numba_rvsdg.networkx_vendored.scc import scc

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
        from numba_rvsdg.networkx_vendored.scc import scc

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

    def find_headers_and_entries(
        self, subgraph: Set[Label]
    ) -> Tuple[Set[Label], Set[Label]]:
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
            nodes_jump_in_loop = subgraph.intersection(self.graph[outside].jump_targets)
            headers.update(nodes_jump_in_loop)
            if nodes_jump_in_loop:
                entries.add(outside)
        # If the loop has no headers or entries, the only header is the head of
        # the CFG.
        if not headers:
            headers = {self.find_head()}
        return headers, entries

    def find_exiting_and_exits(
        self, subgraph: Set[Label]
    ) -> Tuple[Set[Label], Set[Label]]:
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

    def is_reachable_dfs(self, begin: Label, end: Label):  # -> TypeGuard:
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

    def insert_block(
        self, new_label: Label, predecessors: Set[Label], successors: Set[Label]
    ):
        # TODO: needs a diagram and documentaion
        # initialize new block
        new_block = BasicBlock(
            label=new_label, _jump_targets=successors, backedges=set()
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

    def insert_block_and_control_blocks(
        self, new_label: Label, predecessors: Set[Label], successors: Set[Label]
    ):
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
                self.graph.pop(label).replace_jump_targets(jump_targets=tuple(jt))
            )
        # initialize new block, which will hold the branching table
        new_block = BranchBlock(
            label=new_label,
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
        return_nodes = [node for node in self.graph if self.graph[node].is_exiting]
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

    @staticmethod
    def bcmap_from_bytecode(bc: dis.Bytecode):
        return {inst.offset: inst for inst in bc}
