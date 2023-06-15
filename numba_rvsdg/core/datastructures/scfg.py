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
    SyntheticFill,
    RegionBlock,
)
from numba_rvsdg.core.datastructures import block_names

@dataclass(frozen=True)
class NameGenerator:
    """
        The NameGenerator class is responsible for generating unique names 
        for blocks, regions, and variables within the control flow graph. 
    """
    kinds: dict[str, int] = field(default_factory=dict)
    """A dictionary that keeps track of the current index for each kind of name."""

    def new_block_name(self, kind: str) -> str:
        """
            Generates a new unique name for a block of the specified kind. It checks 
            if the kind already exists in the kinds dictionary and increments the 
            index if it does. It returns the generated name.
        """
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
        """
            Generates a new unique name for a region of the specified kind. It 
            follows the same logic as new_block_name() but uses the suffix 
            "region" in the generated name.
        """
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
        """
            Generates a new unique name for a variable of the specified kind. 
            It follows the same logic as new_block_name() but uses the suffix 
            "var" in the generated name.
        """
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
    """The SCFG class represents a map of names to blocks within the control flow graph."""

    graph: Dict[str, BasicBlock] = field(default_factory=dict)
    """A dictionary that maps names to corresponding BasicBlock objects within the control flow graph."""

    name_gen: NameGenerator = field(
        default_factory=NameGenerator, compare=False
    )
    """A NameGenerator object that provides unique names for blocks, regions, and variables."""

    # This is the top-level region that this SCFG represents.
    region: RegionBlock = field(init=False, compare=False)

    def __post_init__(self):
        name = self.name_gen.new_region_name("meta")
        new_region = RegionBlock(name=name, kind="meta", header=None,
                                 exiting=None, parent_region=None,
                                 subregion=self)
        object.__setattr__(self, "region", new_region)

    def __getitem__(self, index):
        """Allows accessing a block from the graph dictionary using the index notation."""
        return self.graph[index]

    def __contains__(self, index):
        """Checks if the given index exists in the graph dictionary."""
        return index in self.graph

    def __iter__(self):
        """Returns an iterator that yields the names and corresponding blocks in the 
        control flow graph. It follows a breadth-first search traversal starting from 
        the head block."""
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
            if name in self:
                block = self[name]
            else:
                # If this is outside the current graph, just disregard it.
                # (might be the case if inside a region and the block being
                # looked at is outside of the region.)
                continue
            # yield the name, block combo
            yield (name, block)
            # if this is a region, recursively yield everything from that region
            if type(block) == RegionBlock:
                for i in block.subregion:
                    yield i
            # finally add any jump_targets to the list of names to visit
            to_visit.extend(block.jump_targets)

    @property
    def concealed_region_view(self):
        """A property that returns a ConcealedRegionView object, representing 
        a concealed view of the control flow graph."""
        return ConcealedRegionView(self)

    def exclude_blocks(self, exclude_blocks: Set[str]) -> Iterator[str]:
        """Returns an iterator over all nodes (blocks) in the control flow graph 
        that are not present in the exclude_blocks set. It filters out the 
        excluded blocks and yields the remaining blocks."""
        for block in self.graph:
            if block not in exclude_blocks:
                yield block

    def find_head(self) -> str:
        """
            Finds the head block of the CFG.
            Assuming the CFG is closed, this will find the block
            that no other blocks are pointing to.
        """
        heads = set(self.graph.keys())
        for name in self.graph.keys():
            block = self.graph[name]
            for jt in block.jump_targets:
                heads.discard(jt)
        assert len(heads) == 1
        return next(iter(heads))

    def compute_scc(self) -> List[Set[str]]:
        """
            Computes the strongly connected components (SCC) of the control 
            flow graph using Tarjan's algorithm. The implementation is at the 
            scc function from the numba_rvsdg.networkx_vendored.scc module. 
            It returns a list of sets, where each set represents an SCC in 
            the graph. SCCs are useful for detecting loops in the graph.
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
            nodes_jump_in_loop = subgraph.intersection(self.graph[outside]._jump_targets)
            headers.update(nodes_jump_in_loop)
            if nodes_jump_in_loop:
                entries.add(outside)
        # If the loop has no headers or entries, the only header is the head of
        # the CFG.
        if not headers:
            headers = {self.find_head()}
            # If region is not meta, the current SCFG is contained in a
            # RegionBlock. The entries to the subgraph are same as entries
            # to it's parent region block's graph.
            if self.region.kind != "meta":
                parent_region = self.region.parent_region
                _, entries = parent_region.subregion.find_headers_and_entries({self.region.name})
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
            for jt in self.graph[inside].jump_targets:
                if jt not in subgraph:
                    exiting.add(inside)
                    exits.add(jt)
            # any returns
            if self.graph[inside].is_exiting:
                exiting.add(inside)
        return sorted(exiting), sorted(exits)

    def is_reachable_dfs(self, begin: str, end: str):  # -> TypeGuard:
        """
            Checks if the end block is reachable from the begin block in the 
            control flow graph. It performs a depth-first search (DFS) 
            traversal from the begin block, following the edges of the 
            graph. Returns True if the end block is reachable, and False 
            otherwise.
        """
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
        """
            Adds a BasicBlock object to the control flow graph.
            The basicblock parameter represents the block to be added.
        """
        self.graph[basicblock.name] = basicblock

    def remove_blocks(self, names: Set[str]):
        """
            Removes blocks from the control flow graph based on the 
            given set of block names (names). It deletes the corresponding 
            entries from the graph attribute.
        """
        for name in names:
            del self.graph[name]

    def _insert_block(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
        block_type: SyntheticBlock
    ):
        """
            Inserts a new synthetic block into the control flow graph. 
            This method is used internally by other methods to perform 
            block insertion operations.
        """
        # TODO: needs a diagram and documentaion
        # initialize new block
        new_block = block_type(
            name=new_name, _jump_targets=successors, backedges=set()
        )
        # add block to self
        self.add_block(new_block)
        # Replace any arcs from any of predecessors to any of successors with
        # an arc through the inserted block instead.
        for name in predecessors:
            block = self.graph.pop(name)
            jt = list(block.jump_targets)
            if successors:
                for s in successors:
                    if s in jt:
                        if new_name not in jt:
                            jt[jt.index(s)] = new_name
                        else:
                            jt.pop(jt.index(s))
            else:
                jt.append(new_name)
            self.add_block(block.replace_jump_targets(jump_targets=tuple(jt)))

    def insert_SyntheticExit(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        """
            Inserts a synthetic exit block into the control flow graph. 
            The new_name parameter specifies the name of the new block, 
            predecessors is a set of block names representing the 
            predecessors of the new block, and successors is a set of 
            block names representing the successors of the new block.
        """
        self._insert_block(new_name, predecessors, successors, SyntheticExit)

    def insert_SyntheticTail(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        """
            Inserts a synthetic tail block into the control flow graph. 
            The new_name parameter specifies the name of the new block, 
            predecessors is a set of block names representing the 
            predecessors of the new block, and successors is a set of 
            block names representing the successors of the new block.
        """
        self._insert_block(new_name, predecessors, successors, SyntheticTail)

    def insert_SyntheticReturn(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        """
            Inserts a synthetic return block into the control flow graph. 
            The new_name parameter specifies the name of the new block, 
            predecessors is a set of block names representing the predecessors 
            of the new block, and successors is a set of block names 
            representing the successors of the new block.
        """
        self._insert_block(new_name, predecessors, successors, SyntheticReturn)

    def insert_SyntheticFill(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
    ):
        """
            Inserts a synthetic fill block into the control flow graph. 
            The new_name parameter specifies the name of the new block, 
            predecessors is a set of block names representing the predecessors 
            of the new block, and successors is a set of block names 
            representing the successors of the new block.
        """
        self._insert_block(new_name, predecessors, successors, SyntheticFill)

    def insert_block_and_control_blocks(
        self, new_name: str, predecessors: Set[str], successors: Set[str]
    ):
        """
            Inserts a new block along with control blocks into the control 
            flow graph. This method is used for branching assignments.
        """
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
            block = self.graph[name]
            jt = list(block.jump_targets)
            # Need to create synthetic assignments for each arc from a
            # predecessors to a successor and insert it between the predecessor
            # and the newly created block
            for s in set(jt).intersection(successors):
                synth_assign = self.name_gen.new_block_name(block_names.SYNTH_ASSIGN)
                variable_assignment = {}
                variable_assignment[branch_variable] = branch_variable_value
                synth_assign_block = SyntheticAssignment(
                    name=synth_assign,
                    _jump_targets=(new_name,),
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
                self.graph.pop(name).replace_jump_targets(jump_targets=tuple(jt))
            )
        # initialize new block, which will hold the branching table
        new_block = SyntheticHead(
            name=new_name,
            _jump_targets=tuple(successors),
            backedges=set(),
            variable=branch_variable,
            branch_value_table=branch_value_table,
        )
        # add block to self
        self.add_block(new_block)

    def join_returns(self):
        """Close the CFG.

        A closed CFG is a CFG with a unique entry and exit node that have no
        predescessors and no successors respectively.
        """
        # for all nodes that contain a return
        return_nodes = [node for node in self.graph if self.graph[node].is_exiting]
        # close if more than one is found
        if len(return_nodes) > 1:
            return_solo_name = self.name_gen.new_block_name(block_names.SYNTH_RETURN)
            self.insert_SyntheticReturn(return_solo_name, return_nodes, tuple())

    def join_tails_and_exits(self, tails: Set[str], exits: Set[str]):
        """
            Joins the tails and exits of the CFG. The method takes a set of 
            tail node names (tails) and a set of exit node names (exits) as 
            parameters. It handles different cases based on the number of 
            tails and exits.
        """
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
        """
            Static method that creates a bytecode map from a dis.Bytecode 
            object. The method takes a dis.Bytecode object as a parameter 
            and returns a dictionary that maps bytecode offsets to 
            instruction objects.
        """
        return {inst.offset: inst for inst in bc}

    @staticmethod
    def from_yaml(yaml_string):
        """
            Static method that creates an SCFG object from a YAML string 
            representation. The method takes a YAML string as a parameter 
            and returns an SCFG object and a dictionary of block names.
        """
        data = yaml.safe_load(yaml_string)
        scfg, block_dict = SCFG.from_dict(data)
        return scfg, block_dict

    @staticmethod
    def from_dict(graph_dict: dict):
        """
            Static method that creates an SCFG object from a dictionary 
            representation. The method takes a dictionary (graph_dict) 
            representing the control flow graph and returns an SCFG 
            object and a dictionary of block names. The input dictionary 
            should have block indices as keys and dictionaries of block 
            attributes as values.
        """
        scfg_graph = {}
        name_gen = NameGenerator()
        block_dict = {}
        for index in graph_dict.keys():
            block_dict[index] = name_gen.new_block_name(block_names.BASIC)
        for index, attributes in graph_dict.items():
            jump_targets = attributes["jt"]
            backedges = attributes.get("be", ())
            name = block_dict[index]
            block = BasicBlock(
                name=name,
                backedges=tuple(block_dict[idx] for idx in backedges),
                _jump_targets=tuple(block_dict[idx] for idx in jump_targets),
            )
            scfg_graph[name] = block
        scfg = SCFG(scfg_graph, name_gen=name_gen)
        return scfg, block_dict

    def to_yaml(self):
        """
            Converts the SCFG object to a YAML string representation. 
            The method returns a YAML string representing the control 
            flow graph. It iterates over the graph dictionary and 
            generates YAML entries for each block, including jump 
            targets and backedges.
        """
        # Convert to yaml
        scfg_graph = self.graph
        yaml_string = """"""

        for key, value in scfg_graph.items():
            jump_targets = [i for i in value._jump_targets]
            jump_targets = str(jump_targets).replace("\'", "\"")
            back_edges = [i for i in value.backedges]
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
        """
            Converts the SCFG object to a dictionary representation. 
            The method returns a dictionary representing the control flow 
            graph. It iterates over the graph dictionary and generates a 
            dictionary entry for each block, including jump targets and 
            backedges if present.
        """
        scfg_graph = self.graph
        graph_dict = {}
        for key, value in scfg_graph.items():
            curr_dict = {}
            curr_dict["jt"] = [i for i in value._jump_targets]
            if value.backedges:
                curr_dict["be"] = [i for i in value.backedges]
            graph_dict[key] = curr_dict
        return graph_dict

    def view(self, name: str=None):
        from numba_rvsdg.rendering.rendering import SCFGRenderer
        SCFGRenderer(self).view(name)

class AbstractGraphView(Mapping):
    """   
        The AbstractGraphView class is a subclass of Mapping and serves 
        as an abstract base class for graph views. It defines the basic 
        interface for accessing and iterating over elements in a graph 
        view. This class cannot be instantiated directly and requires 
        subclasses to implement the abstract methods.
    """

    def __getitem__(self, item):
        """Abstract method that should be implemented in subclasses. 
        It retrieves the value associated with the given key in 
        the graph view."""
        raise NotImplementedError

    def __iter__(self):
        """Abstract method that should be implemented in subclasses. 
        It returns an iterator over the keys in the graph view."""
        raise NotImplementedError

    def __len__(self):
        """Abstract method that should be implemented in subclasses. 
        It returns the number of elements in the graph view."""
        raise NotImplementedError


class ConcealedRegionView(AbstractGraphView):
    """
        The ConcealedRegionView class is a subclass of AbstractGraphView 
        and represents a view of a control flow graph where regions are 
        "concealed" and treated as a single block. It provides methods 
        to access and iterate over blocks or regions in the concealed view.
    """

    scfg: SCFG = None
    """The control flow graph (SCFG) that the concealed region view is based on."""

    def __init__(self, scfg):
        """Initializes the ConcealedRegionView with the given control flow graph (SCFG)."""
        self.scfg = scfg

    def __getitem__(self, item):
        """Retrieves the value associated with the given key in the concealed region view.
        It delegates the operation to the underlying control flow graph (SCFG)."""
        return self.scfg[item]

    def __iter__(self):
        """Returns an iterator over blocks or regions in the concealed view. 
        It calls the region_view_iterator() method to perform the iteration"""
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
                to_visit.extend(block.subregion[block.exiting].jump_targets)
            else:
                # otherwise add any jump_targets to the list of names to visit
                to_visit.extend(block.jump_targets)

            # finally, yield the name
            yield name

    def __len__(self):
        """Returns the number of elements in the concealed region view. 
        It delegates the operation to the underlying control flow graph (SCFG)"""
        return len(self.scfg)
