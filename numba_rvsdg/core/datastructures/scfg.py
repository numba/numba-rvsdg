import dis
import yaml
from textwrap import dedent
from typing import Set, Tuple, Dict, List, Iterator
from dataclasses import dataclass, field
from collections import deque
from collections.abc import Mapping

from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    SyntheticBlock,
    SyntheticAssignment,
    SyntheticHead,
    SyntheticExit,
    SyntheticTail,
    SyntheticReturn,
    RegionBlock,
)
from numba_rvsdg.core.datastructures import block_names

@dataclass(frozen=True)
class NameGenerator:
    kinds: dict[str, int] = field(default_factory=dict)

    def new_block_name(self, kind: str) -> str:
        if kind in self.kinds.keys():
            idx = self.kinds[kind]
            name = str(kind) + '_block_' + str(idx)
            self.kinds[kind] = idx + 1
        else:
            idx = 0
            name = str(kind) + '_block_' + str(idx)
            self.kinds[kind] = idx + 1
        return name

    def new_region_name(self, kind: str) -> str:
        if kind in self.kinds.keys():
            idx = self.kinds[kind]
            name = str(kind) + '_region_' + str(idx)
            self.kinds[kind] = idx + 1
        else:
            idx = 0
            name = str(kind) + '_region_' + str(idx)
            self.kinds[kind] = idx + 1
        return name

    def new_var_name(self, kind: str) -> str:
        if kind in self.kinds.keys():
            idx = self.kinds[kind]
            name = str(kind) + '_var_' + str(idx)
            self.kinds[kind] = idx + 1
        else:
            idx = 0
            name = str(kind) + '_var_' + str(idx)
            self.kinds[kind] = idx + 1
        return name


@dataclass(frozen=True)
class SCFG:
    """Map of Names to Blocks."""

    blocks: dict[str, BasicBlock] = field(default_factory=dict, init=False)

    _jump_targets: dict[str, list[str]] = field(default_factory=dict, init=False)
    back_edges: dict[str, list[str]] = field(default_factory=dict, init=False)

    name_gen: NameGenerator = field(
        default_factory=NameGenerator, compare=False
    )

    def __getitem__(self, index):
        return self.blocks[index]

    def __contains__(self, index):
        return index in self.blocks

    def __iter__(self):
        """Graph Iterator"""
        # initialise housekeeping datastructures
        to_visit, seen = [self.find_head()], []
        while to_visit:
            # get the next name on the list
            name = to_visit.pop(0)
            # if we have visited this, we skip it
            if name in seen:
                continue
            else:
                seen.append(name)
            # get the corresponding block for the name
            block = self[name]
            # yield the name, block combo
            yield (name, block)
            # if this is a region, recursively yield everything from that region
            if type(block) == RegionBlock:
                for i in block.subregion:
                    yield i
            # finally add any jump_targets to the list of names to visit
            to_visit.extend(self.jump_targets[name])

    @property
    def concealed_region_view(self):
        return ConcealedRegionView(self)
    
    @property
    def jump_targets(self):
        jump_targets = {}
        for name in self._jump_targets.keys():
            jump_targets[name] = []
            for jt in self._jump_targets[name]:
                if jt not in self.back_edges[name]:
                    jump_targets[name].append(jt)
        return jump_targets

    def exclude_blocks(self, exclude_blocks: Set[str]) -> Iterator[str]:
        """Iterator over all nodes not in exclude_blocks."""
        for block in self.blocks:
            if block not in exclude_blocks:
                yield block

    def find_head(self) -> str:
        """Find the head block of the CFG.

        Assuming the CFG is closed, this will find the block
        that no other blocks are pointing to.

        """
        heads = set(self.blocks.keys())
        for name in self.blocks.keys():
            for jt in self.jump_targets[name]:
                heads.discard(jt)
        assert len(heads) == 1
        return next(iter(heads))

    def compute_scc(self) -> List[Set[str]]:
        """
        Strongly-connected component for detecting loops.
        """
        from numba_rvsdg.networkx_vendored.scc import scc

        scfg = self

        class GraphWrap:
            def __init__(self, graph):
                self.blocks = graph

            def __getitem__(self, vertex):
                out = scfg.jump_targets[vertex]
                # Exclude node outside of the subgraph
                return [k for k in out if k in self.blocks]

            def __iter__(self):
                return iter(self.blocks.keys())

        return list(scc(GraphWrap(self.blocks)))

    def compute_scc_subgraph(self, subgraph) -> List[Set[str]]:
        """
        Strongly-connected component for detecting loops inside a subgraph.
        """
        from numba_rvsdg.networkx_vendored.scc import scc
        scfg = self

        class GraphWrap:
            def __init__(self, graph, subgraph):
                self.blocks = graph
                self.subgraph = subgraph

            def __getitem__(self, vertex):
                out = scfg.jump_targets[vertex]
                # Exclude node outside of the subgraph
                return [k for k in out if k in subgraph]

            def __iter__(self):
                return iter(self.blocks.keys())

        return list(scc(GraphWrap(self.blocks, subgraph)))

    def find_headers_and_entries(
        self, subgraph: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Find entries and headers in a given subgraph.

        Entries are blocks outside the subgraph that have an edge pointing to
        the subgraph headers. Headers are blocks that are part of the strongly
        connected subset and that have incoming edges from outside the
        subgraph. Entries point to headers and headers are pointed to by
        entries.

        """
        outside: str
        entries: Set[str] = set()
        headers: Set[str] = set()

        for outside in self.exclude_blocks(subgraph):
            nodes_jump_in_loop = subgraph.intersection(self.jump_targets[outside])
            headers.update(nodes_jump_in_loop)
            if nodes_jump_in_loop:
                entries.add(outside)
        # If the loop has no headers or entries, the only header is the head of
        # the CFG.
        if not headers:
            headers = {self.find_head()}
        return sorted(headers), sorted(entries)

    def find_exiting_and_exits(
        self, subgraph: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Find exiting and exit blocks in a given subgraph.

        Existing blocks are blocks inside the subgraph that have edges to
        blocks outside of the subgraph. Exit blocks are blocks outside the
        subgraph that have incoming edges from within the subgraph. Exiting
        blocks point to exits and exits and pointed to by exiting blocks.

        """
        inside: str
        exiting: Set[str] = set()
        exits: Set[str] = set()
        for inside in subgraph:
            # any node inside that points outside the loop
            for jt in self.jump_targets[inside]:
                if jt not in subgraph:
                    exiting.add(inside)
                    exits.add(jt)
            # any returns
            if self.is_exiting(inside):
                exiting.add(inside)
        return sorted(exiting), sorted(exits)

    def is_reachable_dfs(self, begin: str, end: str):  # -> TypeGuard:
        """Is end reachable from begin."""
        seen = set()
        to_vist = list(self.jump_targets[begin])
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
                if block in self.blocks:
                    to_vist.extend(self.jump_targets[block])

    def add_block(self, basicblock: BasicBlock, jump_targets: List[str], back_edges: List[str]):
        self.blocks[basicblock.name] = basicblock
        self._jump_targets[basicblock.name] = jump_targets
        self.back_edges[basicblock.name] = back_edges

    def remove_blocks(self, names: Set[str]):
        for name in names:
            del self.blocks[name]
            del self._jump_targets[name]
            del self.back_edges[name]

    def is_exiting(self, block_name: str) -> bool:
        return not self.jump_targets[block_name]

    def is_fallthrough(self, block_name: str) -> bool:
        return len(self.jump_targets[block_name]) == 1

    def _insert_block(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
        block_type: SyntheticBlock
    ):
        # TODO: needs a diagram and documentaion
        # initialize new block
        new_block = block_type(
            name=new_name
        )
        # add block to self
        self.add_block(new_block, successors, [])
        # Replace any arcs from any of predecessors to any of successors with
        # an arc through the inserted block instead.
        for name in predecessors:
            if hasattr(self.blocks[name], 'branch_value_table'):
                for key, value in self.blocks[name].branch_value_table.items():
                    if value in successors:
                        self.blocks[name].branch_value_table[key] = new_name
            jt = list(self.jump_targets[name])
            if successors:
                for s in successors:
                    if s in jt:
                        if new_name not in jt:
                            jt[jt.index(s)] = new_name
                        else:
                            jt.pop(jt.index(s))
            else:
                jt.append(new_name)
            self._jump_targets[name] = jt

    def insert_SyntheticExit(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        self._insert_block(new_name, predecessors, successors, SyntheticExit)

    def insert_SyntheticTail(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        self._insert_block(new_name, predecessors, successors, SyntheticTail)

    def insert_SyntheticReturn(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        self._insert_block(new_name, predecessors, successors, SyntheticReturn)

    def insert_SyntheticFill(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        self._insert_block(new_name, predecessors, successors, SyntheticReturn)

    def insert_block_and_control_blocks(
        self, new_name: str, predecessors: Set[str], successors: Set[str]
    ):
        # TODO: needs a diagram and documentaion
        # name of the variable for this branching assignment
        branch_variable = self.name_gen.new_var_name("control")
        # initial value of the assignment
        branch_variable_value = 0
        # store for the mapping from variable value to name
        branch_value_table = {}
        # Replace any arcs from any of predecessors to any of successors with
        # an arc through the to be inserted block instead.
        for name in predecessors:
            jt = list(self.jump_targets[name])
            # Need to create synthetic assignments for each arc from a
            # predecessors to a successor and insert it between the predecessor
            # and the newly created block
            for s in set(jt).intersection(successors):
                synth_assign = self.name_gen.new_block_name(block_names.SYNTH_ASSIGN)
                variable_assignment = {}
                variable_assignment[branch_variable] = branch_variable_value
                synth_assign_block = SyntheticAssignment(
                    name=synth_assign,
                    variable_assignment=variable_assignment,
                )
                # add block
                self.add_block(synth_assign_block, [new_name], [])
                # update branching table
                branch_value_table[branch_variable_value] = s
                # update branching variable
                branch_variable_value += 1
                # replace previous successor with synth_assign
                jt[jt.index(s)] = synth_assign
            # finally, replace the jump_targets
            self._jump_targets[name] = jt
        # initialize new block, which will hold the branching table
        new_block = SyntheticHead(
            name=new_name,
            variable=branch_variable,
            branch_value_table=branch_value_table,
        )
        # add block to self
        self.add_block(new_block, successors, [])

    def join_returns(self):
        """Close the CFG.

        A closed CFG is a CFG with a unique entry and exit node that have no
        predescessors and no successors respectively.
        """
        # for all nodes that contain a return
        return_nodes = [node for node in self.blocks if self.is_exiting(node)]
        # close if more than one is found
        if len(return_nodes) > 1:
            return_solo_name = self.name_gen.new_block_name(block_names.SYNTH_RETURN)
            self.insert_SyntheticReturn(return_solo_name, return_nodes, tuple())

    def join_tails_and_exits(self, tails: Set[str], exits: Set[str]):
        if len(tails) == 1 and len(exits) == 1:
            # no-op
            solo_tail_name = next(iter(tails))
            solo_exit_name = next(iter(exits))
            return solo_tail_name, solo_exit_name

        if len(tails) == 1 and len(exits) == 2:
            # join only exits
            solo_tail_name = next(iter(tails))
            solo_exit_name = self.name_gen.new_block_name(block_names.SYNTH_EXIT)
            self.insert_SyntheticExit(solo_exit_name, tails, exits)
            return solo_tail_name, solo_exit_name

        if len(tails) >= 2 and len(exits) == 1:
            # join only tails
            solo_tail_name = self.name_gen.new_block_name(block_names.SYNTH_TAIL)
            solo_exit_name = next(iter(exits))
            self.insert_SyntheticTail(solo_tail_name, tails, exits)
            return solo_tail_name, solo_exit_name

        if len(tails) >= 2 and len(exits) >= 2:
            # join both tails and exits
            solo_tail_name = self.name_gen.new_block_name(block_names.SYNTH_TAIL)
            solo_exit_name = self.name_gen.new_block_name(block_names.SYNTH_EXIT)
            self.insert_SyntheticTail(solo_tail_name, tails, exits)
            self.insert_SyntheticExit(solo_exit_name, set((solo_tail_name,)), exits)
            return solo_tail_name, solo_exit_name

    @staticmethod
    def bcmap_from_bytecode(bc: dis.Bytecode):
        return {inst.offset: inst for inst in bc}

    @staticmethod
    def from_yaml(yaml_string):
        data = yaml.safe_load(yaml_string)
        scfg, block_dict = SCFG.from_dict(data)
        return scfg, block_dict

    @staticmethod
    def from_dict(graph_dict: dict):
        scfg = SCFG()
        name_gen = scfg.name_gen
        block_dict = {}
        for index in graph_dict.keys():
            block_dict[index] = name_gen.new_block_name(block_names.BASIC)
        for index, attributes in graph_dict.items():
            jump_targets = attributes["jt"]
            backedges = attributes.get("be", ())
            name = block_dict[index]
            block = BasicBlock(name=name)
            backedges=tuple(block_dict[idx] for idx in backedges)
            jump_targets=tuple(block_dict[idx] for idx in jump_targets)
            scfg.add_block(block, jump_targets, backedges)
        return scfg, block_dict

    def to_yaml(self):
        # Convert to yaml
        scfg_graph = self.blocks
        yaml_string = """"""

        for key, value in scfg_graph.items():
            jump_targets = [i for i in self._jump_targets[key]]
            jump_targets = str(jump_targets).replace("\'", "\"")
            back_edges = [i for i in self.back_edges[key]]
            jump_target_str = f"""
                "{key}":
                    jt: {jump_targets}"""

            if back_edges:
                back_edges = str(back_edges).replace("\'", "\"")
                jump_target_str += f"""
                    be: {back_edges}"""
            yaml_string += dedent(jump_target_str)

        return yaml_string

    def to_dict(self):
        scfg_graph = self.blocks
        graph_dict = {}
        for key, value in scfg_graph.items():
            curr_dict = {}
            curr_dict["jt"] = [i for i in self._jump_targets[key]]
            if self.back_edges[key]:
                curr_dict["be"] = [i for i in self.back_edges[key]]
            graph_dict[key] = curr_dict
        return graph_dict


class AbstractGraphView(Mapping):

    def __getitem__(self, item):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class ConcealedRegionView(AbstractGraphView):

    def __init__(self, scfg: SCFG):
        self.scfg = scfg

    def __getitem__(self, item):
        return self.scfg[item]

    def __iter__(self):
        return self.region_view_iterator()

    def region_view_iterator(self, head: str = None) -> Iterator[str]:
        """ Region View Iterator.

        This iterator is region aware, which means that regions are "concealed"
        and act as though they were a single block.

        Parameters
        ----------
        head: str, optional
            The head block (or region) from which to start iterating. If None
            is given, will discover the head automatically.

        Returns
        -------
        blocks: iter of str
            An iterator over blocks (or regions)

        """
        # Initialise housekeeping datastructures:
        # A set because we only need lookup and have unique items and a deque
        # because we need a first in, first out (FIFO) structure.
        to_visit, seen = deque([head if head else self.scfg.find_head()]), set()
        while to_visit:
            # get the next name on the list
            name = to_visit.popleft()
            # if we have visited this, we skip it
            if name in seen:
                continue
            else:
                seen.add(name)
            # get the corresponding block for the name (could also be a region)
            try:
                block = self[name]
            except KeyError:
                # If this is outside the current graph, just disregard it.
                # (might be the case if inside a region and the block being
                # looked at is outside of the region.)
                continue

            # populate the to_vist
            if type(block) == RegionBlock:
                # If this is a region, continue on to the exiting block, i.e.
                # the region is presented a single fall-through block to the
                # consumer of this iterator.
                to_visit.extend(self.scfg.jump_targets[name])
            else:
                # otherwise add any outgoing edges to the list of names to visit
                to_visit.extend(self.scfg.jump_targets[name])

            # finally, yield the name
            yield name

    def __len__(self):
        return len(self.scfg)
