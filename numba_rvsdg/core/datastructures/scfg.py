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
    """Unique Name Generator.

    The NameGenerator class is responsible for generating unique names
    for blocks, regions, and variables within the SCFG.

    Attributes
    ----------
    kinds: dict[str, int]
        A dictionary that keeps track of the current index for each kind
        of name.
    """

    kinds: dict[str, int] = field(default_factory=dict)

    def new_block_name(self, kind: str) -> str:
        """Generate a new unique name for a block of the specified kind.

        This method checks if the given string 'kind' already exists
        in the kinds dictionary attribute. If it exists, the respective
        index is incremented, if it doesn't then a new index (starting
        from zero) is asigned to the given kind. This ensures that
        the given name is unique by a combination of it's kind and it's index.
        It returns the generated name.

        Parameters
        ----------
        kind: str
            The kind of block for which name is being generated.

        Return
        ------
        name: str
            Unique name for the given kind of block.
        """
        if kind in self.kinds.keys():
            idx = self.kinds[kind]
            name = str(kind) + "_block_" + str(idx)
            self.kinds[kind] = idx + 1
        else:
            idx = 0
            name = str(kind) + "_block_" + str(idx)
            self.kinds[kind] = idx + 1
        return name

    def new_region_name(self, kind: str) -> str:
        """Generate a new unique name for a region of the specified kind.

        This method checks if the given string 'kind' already exists
        in the kinds dictionary attribute. If it exists, the respective
        index is incremented, if it doesn't then a new index (starting
        from zero) is asigned to the given kind. This ensures that
        the given name is unique by a combination of it's kind and it's index.
        It returns the generated name.

        Parameters
        ----------
        kind: str
            The kind of region for which name is being generated.

        Return
        ------
        name: str
            Unique name for the given kind of region.
        """
        if kind in self.kinds.keys():
            idx = self.kinds[kind]
            name = str(kind) + "_region_" + str(idx)
            self.kinds[kind] = idx + 1
        else:
            idx = 0
            name = str(kind) + "_region_" + str(idx)
            self.kinds[kind] = idx + 1
        return name

    def new_var_name(self, kind: str) -> str:
        """Generate a new unique name for a variable of the specified kind.

        This method checks if the given string 'kind' already exists
        in the kinds dictionary attribute. If it exists, the respective
        index is incremented, if it doesn't then a new index (starting
        from zero) is asigned to the given kind. This ensures that
        the given name is unique by a combination of it's kind and it's
        index. It returns the generated name.

        Parameters
        ----------
        kind: str
            The kind of variable for which name is being generated.

        Return
        ------
        name: str
            Unique name for the given kind of variable.
        """
        if kind in self.kinds.keys():
            idx = self.kinds[kind]
            name = str(kind) + "_var_" + str(idx)
            self.kinds[kind] = idx + 1
        else:
            idx = 0
            name = str(kind) + "_var_" + str(idx)
            self.kinds[kind] = idx + 1
        return name


@dataclass(frozen=True)
class SCFG:
    """SCFG (Structured Control Flow Graph) class.

    The SCFG class represents a map of names to blocks within the control
    flow graph.

    Attributes
    ----------
    graph: Dict[str, BasicBlock]
        A dictionary that maps names to corresponding BasicBlock objects
        within the control flow graph.

    name_gen: NameGenerator
        A NameGenerator object that provides unique names for blocks,
        regions, and variables.
    """

    graph: Dict[str, BasicBlock] = field(default_factory=dict)

    name_gen: NameGenerator = field(
        default_factory=NameGenerator, compare=False
    )

    # This is the top-level region that this SCFG represents.
    region: RegionBlock = field(init=False, compare=False)

    def __post_init__(self):
        name = self.name_gen.new_region_name("meta")
        new_region = RegionBlock(
            name=name,
            kind="meta",
            header=None,
            exiting=None,
            parent_region=None,
            subregion=self,
        )
        object.__setattr__(self, "region", new_region)

    def __getitem__(self, index):
        """Access a block from the graph dictionary using the block name.

        Parameters
        ----------
        index: str
            The name of the block to be accessed.

        Returns
        -------
        block: BasicBlock
            The requested block.
        """
        return self.graph[index]

    def __contains__(self, index):
        """Checks if the given index exists in the graph dictionary.

        Parameters
        ----------
        index: str
            The name of the block to be checked.

        Returns
        -------
        result: bool
            Returns True if a block with given name is present in the SCFG,
            and returns False if it isn't.
        """
        return index in self.graph

    def __iter__(self):
        """Returns an iterator over the blocks in the SCFG.

        Returns an iterator that yields the names and corresponding blocks
        in the SCFG. It follows a breadth-first search
        traversal starting from the head block.

        Returns
        -------
        (name, block): iter of type tuple(str, BasicBlock)
            An iterator over a tuple of name and blocks (or regions)
            over the given view.
        """
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
            # If this is a region, recursively yield everything from that
            # specific region.
            if type(block) == RegionBlock:
                yield from block.subregion
            # finally add any jump_targets to the list of names to visit
            to_visit.extend(block.jump_targets)

    @property
    def concealed_region_view(self):
        """A property that returns a ConcealedRegionView object, representing
        a concealed view of the control flow graph.

        Returns
        -------
        region_view: ConcealedRegionView
            A concealed view of the current SCFG.
        """
        return ConcealedRegionView(self)

    def exclude_blocks(self, exclude_blocks: Set[str]) -> Iterator[str]:
        """Returns an iterator over the blocks in the SCFG with exclusions.

        Returns an iterator over all nodes (blocks) in the control flow graph
        that are not present in the exclude_blocks set. It filters out the
        excluded blocks and yields the remaining blocks.

        Parameters
        ----------
        exclude_blocks: Set[str]
            Set of blocks to be excluded.

        Returns
        -------
        blocks: Iterator[str]
            An iterator over blocks (or regions) over the given SCFG with
            the specified blocks excluded.
        """
        for block in self.graph:
            if block not in exclude_blocks:
                yield block

    def find_head(self) -> str:
        """Finds the head block of the SCFG.

        Assuming the SCFG is closed, this will find the block
        that no other blocks are pointing to.

        Returns
        -------
        head: str
            Name of the head block of the graph.
        """
        heads = set(self.graph.keys())
        for name in self.graph.keys():
            block = self.graph[name]
            for jt in block.jump_targets:
                heads.discard(jt)
        assert len(heads) == 1
        return next(iter(heads))

    def compute_scc(self) -> List[Set[str]]:
        """Computes the strongly connected components (SCC) of the current
        SCFG.

        This method of SCFG computes the strongly connected components of
        the graph using Tarjan's algorithm. The implementation is at the
        scc function from the numba_rvsdg.networkx_vendored.scc module.
        It returns a list of sets, where each set represents an SCC in
        the graph. SCCs are useful for detecting loops in the graph.

        Returns
        -------
        components: List[Set[str]]
            A list of sets of strongly connected components/BasicBlocks.
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
        """Finds entries and headers in a given subgraph.

        Entries are blocks outside the subgraph that have an edge pointing to
        the subgraph headers. Headers are blocks that are part of the strongly
        connected subset and that have incoming edges from outside the
        subgraph. Entries point to headers and headers are pointed to by
        entries.

        Parameters
        ----------
        subgraph: Set[str]
            The subgraph for which headers and entries are to be computed.

        Returns
        -------
        (headers, entries): Tuple[Set[str], Set[str]]
            A tuple consisting of two entries the set of header blocks
            and set of entry blocks respectively.
        """
        outside: str
        entries: Set[str] = set()
        headers: Set[str] = set()

        for outside in self.exclude_blocks(subgraph):
            nodes_jump_in_loop = subgraph.intersection(
                self.graph[outside]._jump_targets
            )
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
                _, entries = parent_region.subregion.find_headers_and_entries(
                    {self.region.name}
                )
        return sorted(headers), sorted(entries)

    def find_exiting_and_exits(
        self, subgraph: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """Finds exiting and exit blocks in a given subgraph.

        Existing blocks are blocks inside the subgraph that have edges to
        blocks outside of the subgraph. Exit blocks are blocks outside the
        subgraph that have incoming edges from within the subgraph. Exiting
        blocks point to exits and exits and pointed to by exiting blocks.

        Parameters
        ----------
        subgraph: Set[str]
            The subgraph for which exit and exiting blocks are to be computed.

        Returns
        -------
        (exiting, exits): Tuple[Set[str], Set[str]]
            A tuple consisting of two entries the set of exiting blocks
            and set of exit blocks respectively.
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
        """Checks if the end block is reachable from the begin block in the
        SCFG.

        This method performs a depth-first search (DFS)
        traversal from the begin block, following the edges of the
        graph. Returns True if the end block is reachable, and False
        otherwise.

        Parameters
        ----------
        begin: str
            The name of starting block for traversal.
        end: str
            The name of end block for the traversal.

        Returns
        -------
        result: bool
            True if the end block is reachable from begin block
            and False otherwise.
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

    def add_block(self, basic_block: BasicBlock):
        """Adds a BasicBlock object to the control flow graph.

        Parameters
        ----------
        basic_block: BasicBlock
            The basic_block parameter represents the block to be added.
        """
        self.graph[basic_block.name] = basic_block

    def remove_blocks(self, names: Set[str]):
        """Removes a BasicBlock object from the control flow graph.

        Parameters
        ----------
        names: Set[str]
            The set of names of BasicBlocks to be removed from the graph.
        """
        for name in names:
            del self.graph[name]

    def insert_block(
        self,
        new_name: str,
        predecessors: Set[str],
        successors: Set[str],
        block_type: SyntheticBlock,
    ):
        """Inserts a new synthetic block into the SCFG
        between the given successors and predecessors.

        This method inserts a new block between the specified successor
        and predecessor blocks. Edges between all the pairs of sucessor
        and predecessor blocks are replaced by edges going through the
        newly added block. (i.e. all outgoing edges from predecessors
        pointing to successor blocks, point towards the newly aded block
        and all incoming edges towards the successors originating from a
        predecessor, now originate from the newly added block instead).

        Parameters
        ----------
        new_name: str
            The name of the newly created block.
        predecessors: Set[str]
            The set of names of BasicBlock that act as predecessors
            for the block to be inserted.
        successors: Set[str]
            The set of names of BasicBlock that act as successors
            for the block to be inserted.
        block_type: SyntheticBlock
            The type/class of the newly created block.
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
        self,
        new_name: str,
        predecessors: Set[str],
        successors: Set[str],
    ):
        """Inserts a synthetic exit block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_rvsdg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticExit)

    def insert_SyntheticTail(
        self,
        new_name: str,
        predecessors: Set[str],
        successors: Set[str],
    ):
        """Inserts a synthetic tail block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_rvsdg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticTail)

    def insert_SyntheticReturn(
        self,
        new_name: str,
        predecessors: Set[str],
        successors: Set[str],
    ):
        """Inserts a synthetic return block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_rvsdg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticReturn)

    def insert_SyntheticFill(
        self,
        new_name: str,
        predecessors: Set[str],
        successors: Set[str],
    ):
        """Inserts a synthetic fill block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_rvsdg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticFill)

    def insert_block_and_control_blocks(
        self, new_name: str, predecessors: Set[str], successors: Set[str]
    ):
        """Inserts a new block along with control blocks into the SCFG.
        This method is used for branching assignments.
        Parameters same as insert_block method.

        See also
        --------
        numba_rvsdg.core.datastructures.scfg.SCFG.insert_block
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
                synth_assign = self.name_gen.new_block_name(
                    block_names.SYNTH_ASSIGN
                )
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
                self.graph.pop(name).replace_jump_targets(
                    jump_targets=tuple(jt)
                )
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
        return_nodes = [
            node for node in self.graph if self.graph[node].is_exiting
        ]
        # close if more than one is found
        if len(return_nodes) > 1:
            return_solo_name = self.name_gen.new_block_name(
                block_names.SYNTH_RETURN
            )
            self.insert_SyntheticReturn(
                return_solo_name, return_nodes, tuple()
            )

    def join_tails_and_exits(self, tails: Set[str], exits: Set[str]):
        """Joins the tails and exits of the SCFG.

        Parameters
        ----------
        tails: Set[str]
            The set of names of BasicBlock that act as tails in the SCFG.
        exits: Set[str]
            The set of names of BasicBlock that act as exits in the SCFG.

        Return
        ------
        solo_tail_name: str
            The name of the unique tail block in the modified SCFG.
        solo_exit_name: str
            the name of the unique exit block in the modified SCFG.
        """
        if len(tails) == 1 and len(exits) == 1:
            # no-op
            solo_tail_name = next(iter(tails))
            solo_exit_name = next(iter(exits))
            return solo_tail_name, solo_exit_name

        if len(tails) == 1 and len(exits) == 2:
            # join only exits
            solo_tail_name = next(iter(tails))
            solo_exit_name = self.name_gen.new_block_name(
                block_names.SYNTH_EXIT
            )
            self.insert_SyntheticExit(solo_exit_name, tails, exits)
            return solo_tail_name, solo_exit_name

        if len(tails) >= 2 and len(exits) == 1:
            # join only tails
            solo_tail_name = self.name_gen.new_block_name(
                block_names.SYNTH_TAIL
            )
            solo_exit_name = next(iter(exits))
            self.insert_SyntheticTail(solo_tail_name, tails, exits)
            return solo_tail_name, solo_exit_name

        if len(tails) >= 2 and len(exits) >= 2:
            # join both tails and exits
            solo_tail_name = self.name_gen.new_block_name(
                block_names.SYNTH_TAIL
            )
            solo_exit_name = self.name_gen.new_block_name(
                block_names.SYNTH_EXIT
            )
            self.insert_SyntheticTail(solo_tail_name, tails, exits)
            self.insert_SyntheticExit(solo_exit_name, {solo_tail_name}, exits)
            return solo_tail_name, solo_exit_name

    @staticmethod
    def bcmap_from_bytecode(bc: dis.Bytecode):
        """Static method that creates a bytecode map from a `dis.Bytecode`
        object.

        Parameters
        ----------
        bc: dis.Bytecode
            The ByteCode object to be converted.

        Return
        ------
        bcmap: Dict
            The corresponding dictionary that maps bytecode offsets to
            instruction objects.
        """
        return {inst.offset: inst for inst in bc}

    @staticmethod
    def from_yaml(yaml_string):
        """Static method that creates an SCFG object from a YAML
        representation.

        This method takes a YAML string
        representing the control flow graph and returns an SCFG
        object and a dictionary of block names in YAML string
        corresponding to thier representation/unique name IDs in the SCFG.

        Parameters
        ----------
        yaml: str
            The input YAML string from which the SCFG is to be constructed.

        Return
        ------
        scfg: SCFG
            The corresponding SCFG created using the YAML representation.
        block_dict: Dict[str, str]
            Dictionary of block names in YAML string corresponding to their
            representation/unique name IDs in the SCFG.
        """
        data = yaml.safe_load(yaml_string)
        scfg, block_dict = SCFG.from_dict(data)
        return scfg, block_dict

    @staticmethod
    def from_dict(graph_dict: dict):
        """Static method that creates an SCFG object from a dictionary
        representation.

        This method takes a dictionary (graph_dict)
        representing the control flow graph and returns an SCFG
        object and a dictionary of block names. The input dictionary
        should have block indices as keys and dictionaries of block
        attributes as values.

        Parameters
        ----------
        graph_dict: dict
            The input dictionary from which the SCFG is to be constructed.

        Return
        ------
        scfg: SCFG
            The corresponding SCFG created using the dictionary representation.
        block_dict: Dict[str, str]
            Dictionary of block names in YAML string corresponding to their
            representation/unique name IDs in the SCFG.
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
        """Converts the SCFG object to a YAML string representation.

        The method returns a YAML string representing the control
        flow graph. It iterates over the graph dictionary and
        generates YAML entries for each block, including jump
        targets and backedges.

        Returns
        -------
        yaml: str
            A YAML string representing the SCFG.
        """
        # Convert to yaml
        scfg_graph = self.graph
        yaml_string = """"""

        for key, value in scfg_graph.items():
            jump_targets = [i for i in value._jump_targets]
            jump_targets = str(jump_targets).replace("'", '"')
            back_edges = [i for i in value.backedges]
            jump_target_str = f"""
                "{key}":
                    jt: {jump_targets}"""

            if back_edges:
                back_edges = str(back_edges).replace("'", '"')
                jump_target_str += f"""
                    be: {back_edges}"""
            yaml_string += dedent(jump_target_str)

        return yaml_string

    def to_dict(self):
        """Converts the SCFG object to a dictionary representation.

        This method returns a dictionary representing the control flow
        graph. It iterates over the graph dictionary and generates a
        dictionary entry for each block, including jump targets and
        backedges if present.

        Returns
        -------
        graph_dict: Dict[Dict[...]]
            A dictionary representing the SCFG.
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

    def view(self, name: str = None):
        """View the current SCFG as a external PDF file.

        This method internally creates a SCFGRenderer corresponding to
        the current state of SCFG and calls it's view method to view the
        graph as a graphviz generated external PDF file.

        Parameters
        ----------
        name: str
            Name to be given to the external graphviz generated PDF file.
        """
        from numba_rvsdg.rendering.rendering import SCFGRenderer

        SCFGRenderer(self).view(name)


class AbstractGraphView(Mapping):
    """Abstract Graph View class.

    The AbstractGraphView class serves as a template for graph views.
    """

    def __getitem__(self, item):
        """Retrieves the value associated with the given key or name
        in the respective graph view.

        Parameters
        ----------
        item: str
            The name for which to fetch the BasicBlock.

        Return
        ------
        block: BasicBlock
            The requested block.
        """
        raise NotImplementedError

    def __iter__(self):
        """Returns an iterator over the name of blocks in the graph view.

        Returns
        -------
        blocks: iter of str
            An iterator over blocks (or regions) over the given view.
        """
        raise NotImplementedError

    def __len__(self):
        """Returns the number of elements in the given region view.

        Return
        ------
        len: int
            Length/ of given SCFG view (number of blocks in the given
            SCFG view).
        """
        raise NotImplementedError


class ConcealedRegionView(AbstractGraphView):
    """Concealed Region View class.

    The ConcealedRegionView represents a view of a SCFG
    where regions are "concealed" and treated as a single block.

    Parameters
    ----------
    scfg: SCFG
        The SCFG for which to instantiate the view.

    Attributes
    ----------
    scfg: SCFG
        The SCFG that the concealed region view
        is based on.
    """

    scfg: SCFG = None

    def __init__(self, scfg):
        """Initializes the ConcealedRegionView with the given SCFG.

        Parameters
        ----------
        scfg: SCFG
            The SCFG for which to instantiate the view.
        """
        self.scfg = scfg

    def __getitem__(self, item):
        """Retrieves the value associated with the given key or name
        in the respective graph view.

        Parameters
        ----------
        item: str
            The name for which to fetch the BasicBlock.

        Return
        ------
        block: BasicBlock
            The requested block.
        """
        return self.scfg[item]

    def __iter__(self):
        """Returns an iterator over the name of blocks in the concealed
        graph view.

        Returns
        -------
        blocks: iter of str
            An iterator over blocks (or regions) over the given view.
        """
        return self.region_view_iterator()

    def region_view_iterator(self, head: str = None) -> Iterator[str]:
        """Region View Iterator.

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
        to_visit, seen = (
            deque([head if head else self.scfg.find_head()]),
            set(),
        )
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

        Return
        ------
        len: int
            Length/ of given SCFG view (number of blocks in the concealed
            SCFG view).
        """
        return len(self.scfg)
