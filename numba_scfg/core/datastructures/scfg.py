import dis
import yaml
from typing import (
    Any,
    Set,
    Tuple,
    Dict,
    List,
    Iterator,
    Optional,
    Generator,
    Mapping,
    MutableMapping,
    Sized,
)
from textwrap import indent
from dataclasses import dataclass, field
from collections import deque

from numba_scfg.core.datastructures.basic_block import (
    BasicBlock,
    SyntheticBlock,
    SyntheticAssignment,
    SyntheticHead,
    SyntheticExit,
    SyntheticTail,
    SyntheticReturn,
    SyntheticFill,
    PythonBytecodeBlock,
    RegionBlock,
    SyntheticBranch,
    block_type_names,
)
from numba_scfg.core.datastructures.block_names import (
    block_types,
    SYNTH_TAIL,
    SYNTH_EXIT,
    SYNTH_ASSIGN,
    SYNTH_RETURN,
)


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
            name = "__scfg_" + str(kind) + "_var_" + str(idx) + "__"
            self.kinds[kind] = idx + 1
        else:
            idx = 0
            name = "__scfg_" + str(kind) + "_var_" + str(idx) + "__"
            self.kinds[kind] = idx + 1
        return name


@dataclass(frozen=True)
class SCFG(Sized):
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

    graph: MutableMapping[str, BasicBlock] = field(default_factory=dict)

    name_gen: NameGenerator = field(
        default_factory=NameGenerator, compare=False
    )

    # This is the top-level region that this SCFG represents.
    region: RegionBlock = field(init=False, compare=False)

    def __post_init__(self) -> None:
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

    def __getitem__(self, index: str) -> BasicBlock:
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

    def __contains__(self, index: str) -> bool:
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

    def __len__(self) -> int:
        """
        Returns
        -------
        Number of nodes in the graph
        """
        return len(self.graph)

    def __iter__(self) -> Generator[Tuple[str, BasicBlock], None, None]:
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
        try:
            to_visit = [self.find_head()]
            seen: list[str] = []
        except KeyError:
            to_visit, seen = ["0"], []
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
            if type(block) == RegionBlock:  # noqa: E721
                assert block.subregion is not None
                yield from block.subregion
            # finally add any jump_targets to the list of names to visit
            to_visit.extend(block.jump_targets)

    @property
    def concealed_region_view(self) -> "ConcealedRegionView":
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
        scc function from the numba_scfg.networkx_vendored.scc module.
        It returns a list of sets, where each set represents an SCC in
        the graph. SCCs are useful for detecting loops in the graph.

        Returns
        -------
        components: List[Set[str]]
            A list of sets of strongly connected components/BasicBlocks.
        """
        from numba_scfg.networkx_vendored.scc import scc

        class GraphWrap:
            def __init__(self, graph: Mapping[str, BasicBlock]) -> None:
                self.graph = graph

            def __getitem__(self, vertex: str) -> List[str]:
                out = self.graph[vertex].jump_targets
                # Exclude node outside of the subgraph
                return [k for k in out if k in self.graph]

            def __iter__(self) -> Iterator[str]:
                return iter(self.graph.keys())

        return list(scc(GraphWrap(self.graph)))  # type: ignore

    def find_headers_and_entries(
        self, subgraph: Set[str]
    ) -> Tuple[List[str], List[str]]:
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
                assert parent_region is not None
                assert parent_region.subregion is not None
                _, entries = parent_region.subregion.find_headers_and_entries(
                    {self.region.name}
                )
        return sorted(headers), sorted(entries)

    def find_exiting_and_exits(
        self, subgraph: Set[str]
    ) -> Tuple[List[str], List[str]]:
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

    def is_reachable_dfs(self, begin: str, end: str) -> bool:
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

    def add_block(self, basic_block: BasicBlock) -> None:
        """Adds a BasicBlock object to the control flow graph.

        Parameters
        ----------
        basic_block: BasicBlock
            The basic_block parameter represents the block to be added.
        """
        self.graph[basic_block.name] = basic_block

    def remove_blocks(self, names: Set[str]) -> None:
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
        predecessors: List[str],
        successors: List[str],
        block_type: type[SyntheticBlock],
    ) -> None:
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
        predecessors: List[str]
            The list of names of BasicBlock that act as predecessors
            for the block to be inserted.
        successors: List[str]
            The list of names of BasicBlock that act as successors
            for the block to be inserted.
        block_type: SyntheticBlock
            The type/class of the newly created block.
        """
        # TODO: needs a diagram and documentaion
        # initialize new block
        new_block = block_type(
            name=new_name, _jump_targets=tuple(successors), backedges=tuple()
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
        predecessors: List[str],
        successors: List[str],
    ) -> None:
        """Inserts a synthetic exit block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticExit)

    def insert_SyntheticTail(
        self,
        new_name: str,
        predecessors: List[str],
        successors: List[str],
    ) -> None:
        """Inserts a synthetic tail block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticTail)

    def insert_SyntheticReturn(
        self,
        new_name: str,
        predecessors: List[str],
        successors: List[str],
    ) -> None:
        """Inserts a synthetic return block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticReturn)

    def insert_SyntheticFill(
        self,
        new_name: str,
        predecessors: List[str],
        successors: List[str],
    ) -> None:
        """Inserts a synthetic fill block into the SCFG.
        Parameters same as insert_block method.

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFG.insert_block
        """
        self.insert_block(new_name, predecessors, successors, SyntheticFill)

    def insert_block_and_control_blocks(
        self, new_name: str, predecessors: List[str], successors: List[str]
    ) -> None:
        """Inserts a new block along with control blocks into the SCFG.
        This method is used for branching assignments.
        Parameters same as insert_block method.

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFG.insert_block
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
            for s in sorted(set(jt).intersection(successors)):
                synth_assign = self.name_gen.new_block_name(SYNTH_ASSIGN)
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
            backedges=tuple(),
            variable=branch_variable,
            branch_value_table=branch_value_table,
        )
        # add block to self
        self.add_block(new_block)

    def join_returns(self) -> None:
        """Close the CFG.

        A closed CFG is a CFG with a unique entry and exit node that have no
        predescessors and no successors respectively. Transformation is applied
        in-place.
        """
        # for all nodes that contain a return
        return_nodes = [
            node for node in self.graph if self.graph[node].is_exiting
        ]
        # close if more than one is found
        if len(return_nodes) > 1:
            return_solo_name = self.name_gen.new_block_name(SYNTH_RETURN)
            self.insert_SyntheticReturn(return_solo_name, return_nodes, [])

    def iter_subregions(self) -> Generator[RegionBlock, "SCFG", None]:
        """Iterate over all subregions of this CFG."""
        for node in self.graph.values():
            if isinstance(node, RegionBlock):
                yield node
                assert node.subregion is not None
                yield from node.subregion.iter_subregions()

    def restructure_loop(self) -> None:
        """Apply LOOP RESTRUCTURING transform.

        Performs the operation to restructure loop constructs using the
        algorithm LOOP RESTRUCTURING from section 4.1 of Bahmann2015.  It
        applies an in-place restructuring operation to both the main SCFG and
        any subregions within it.

        """
        # Avoid cyclic imports
        from numba_scfg.core.transformations import restructure_loop

        restructure_loop(self.region)
        for region in self.iter_subregions():
            restructure_loop(region)

    def restructure_branch(self) -> None:
        """Apply BRANCH RESTRUCTURING transform.

        Performs the operation to restructure branch constructs using the
        algorithm BRANCH RESTRUCTURING from section 4.2 of Bahmann2015.  It
        applies an in-place restructuring operation to both the main SCFG and
        any subregions within it.

        """
        # Avoid cyclic imports
        from numba_scfg.core.transformations import restructure_branch

        restructure_branch(self.region)
        for region in self.iter_subregions():
            restructure_branch(region)

    def restructure(self) -> None:
        self.join_returns()
        self.restructure_loop()
        self.restructure_branch()

    def join_tails_and_exits(
        self, tails: List[str], exits: List[str]
    ) -> Tuple[str, str]:
        """Joins the tails and exits of the SCFG.

        Parameters
        ----------
        tails: List[str]
            The list of names of BasicBlock that act as tails in the SCFG.
        exits: List[str]
            The list of names of BasicBlock that act as exits in the SCFG.

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
            solo_exit_name = self.name_gen.new_block_name(SYNTH_EXIT)
            self.insert_SyntheticExit(solo_exit_name, tails, exits)
            return solo_tail_name, solo_exit_name

        if len(tails) >= 2 and len(exits) == 1:
            # join only tails
            solo_tail_name = self.name_gen.new_block_name(SYNTH_TAIL)
            solo_exit_name = next(iter(exits))
            self.insert_SyntheticTail(solo_tail_name, tails, exits)
            return solo_tail_name, solo_exit_name

        if len(tails) >= 2 and len(exits) >= 2:
            # join both tails and exits
            solo_tail_name = self.name_gen.new_block_name(SYNTH_TAIL)
            solo_exit_name = self.name_gen.new_block_name(SYNTH_EXIT)
            self.insert_SyntheticTail(solo_tail_name, tails, exits)
            self.insert_SyntheticExit(solo_exit_name, [solo_tail_name], exits)
            return solo_tail_name, solo_exit_name

        assert False, "unreachable"

    @staticmethod
    def bcmap_from_bytecode(bc: dis.Bytecode) -> Dict[int, dis.Instruction]:
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

    def view(self, name: Optional[str] = None) -> None:
        """View the current SCFG as a external PDF file.

        This method internally creates a SCFGRenderer corresponding to
        the current state of SCFG and calls it's view method to view the
        graph as a graphviz generated external PDF file.

        Parameters
        ----------
        name: str
            Name to be given to the external graphviz generated PDF file.
        """
        from numba_scfg.rendering.rendering import SCFGRenderer

        SCFGRenderer(self).view(name)

    def render(self) -> None:
        """Alias for view()."""
        self.view()

    @staticmethod
    def from_yaml(yaml_string: str) -> "Tuple[SCFG, Dict[str, str]]":
        """Static method that creates an SCFG object from a YAML
        representation.

        This method takes a YAML string
        representing the control flow graph and returns an SCFG
        object and a dictionary of block names in YAML string
        corresponding to their representation/unique name IDs in the SCFG.

        Internally forwards the `yaml_string` to `SCFGIO.from_yaml()`
        helper method.

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

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFGIO.from_yaml()
        """
        return SCFGIO.from_yaml(yaml_string)

    @staticmethod
    def from_dict(
        graph_dict: Dict[str, Dict[str, List[str]]],
    ) -> Tuple["SCFG", Dict[str, str]]:
        """Static method that creates an SCFG object from a dictionary
        representation.

        This method takes a dictionary (graph_dict)
        representing the control flow graph and returns an SCFG
        object and a dictionary of block names. The input dictionary
        should have block indices as keys and dictionaries of block
        attributes as values.

        Internally forwards the `graph_dict` to `SCFGIO.from_dict()`
        helper method.

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

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFGIO.from_dict()
        """
        return SCFGIO.from_dict(graph_dict)

    def to_yaml(self) -> str:
        """Converts the SCFG object to a YAML string representation.

        The method returns a YAML string representing the control
        flow graph. It iterates over the graph dictionary and
        generates YAML entries for each block, including jump
        targets and backedges.

        Internally calls the `SCFGIO.to_yaml()` helper method on
        current `SCFG` object.

        Returns
        -------
        yaml: str
            A YAML string representing the SCFG.

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFGIO.to_yaml()
        """
        return SCFGIO.to_yaml(self)

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """Converts the SCFG object to a dictionary representation.

        This method returns a dictionary representing the control flow
        graph. It iterates over the graph dictionary and generates a
        dictionary entry for each block, including jump targets and
        backedges if present.

        Internally calls the `SCFGIO.to_dict()` helper method on
        current `SCFG` object.

        Returns
        -------
        graph_dict: Dict[Dict[...]]
            A dictionary representing the SCFG.

        See also
        --------
        numba_scfg.core.datastructures.scfg.SCFGIO.to_dict()
        """
        return SCFGIO.to_dict(self)


class SCFGIO:
    """Helper class for `SCFG` object transformation to and from various
    other formats. Currently supports YAML and dictionary format.
    """

    @staticmethod
    def from_yaml(yaml_string: str) -> Tuple["SCFG", Dict[str, str]]:
        """Static helper method that creates an SCFG object from a YAML
        representation.

        This method takes a YAML string
        representing the control flow graph and returns an SCFG
        object and a dictionary of block names in YAML string
        corresponding to their representation/unique name IDs in the SCFG.

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
    def from_dict(
        graph_dict: Dict[str, Dict[str, Any]],
    ) -> "Tuple[SCFG, Dict[str, str]]":
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
        block_ref_dict: Dict[str, str]
            Dictionary of block names in YAML string corresponding to their
            representation/unique name IDs in the SCFG.
        """
        block_ref_dict = {}
        for key, block in graph_dict["blocks"].items():
            assert block["type"] in block_types
            block_ref_dict[key] = key

        outer_graph = SCFGIO.find_outer_graph(graph_dict)
        assert len(outer_graph) > 0

        name_gen = NameGenerator()
        scfg = SCFGIO.make_scfg(
            graph_dict, outer_graph, block_ref_dict, name_gen
        )

        return scfg, block_ref_dict

    @staticmethod
    def make_scfg(
        graph_dict: Dict[str, Dict[str, Any]],
        curr_heads: Set[str],
        block_ref_dict: Dict[str, str],
        name_gen: NameGenerator,
        exiting: Optional[str] = None,
    ) -> "SCFG":
        """Helper method for building a single 'level' of the hierarchical
        structure in an `SCFG` graph at a time. Recursively calls itself
        to build the entire graph.

        Parameters
        ----------
        graph_dict: dict
            The input dictionary from which the SCFG is to be constructed.
        curr_heads: set
            The set of blocks to start iterating from.
        block_ref_dict: Dict[str, str]
            Dictionary of block names in YAML string corresponding to their
            representation/unique name IDs in the SCFG.
        name_gen: NameGenerator
            The corresponding `NameGenerator` object for the `SCFG` object
            to be created.
        exiting: str
            The exiting node for the current region being iterated.

        Return
        ------
        scfg: SCFG
            The corresponding SCFG created using the dictionary representation.
        """
        blocks = graph_dict["blocks"]
        edges = graph_dict["edges"]
        backedges = graph_dict["backedges"]
        if backedges is None:
            backedges = {}

        scfg_graph = {}
        seen = set()
        # The queue must be a sorted FIFO to maintain reproducible insertion
        # order for the SCFG.
        queue = deque(sorted(curr_heads))

        while queue:
            current_name = queue.popleft()
            if current_name in seen:
                continue
            seen.add(current_name)

            (
                block_info,
                block_type,
                block_edges,
                block_backedges,
            ) = SCFGIO.extract_block_info(
                blocks, current_name, block_ref_dict, edges, backedges
            )

            if block_type == "region":
                block_info["subregion"] = SCFGIO.make_scfg(
                    graph_dict,
                    {block_info["header"]},
                    block_ref_dict,
                    name_gen,
                    block_info["exiting"],
                )
                block_info.pop("contains")

            block_class = block_type_names[block_type]
            block = block_class(
                name=current_name,
                backedges=block_backedges,
                _jump_targets=block_edges,
                **block_info,
            )

            scfg_graph[current_name] = block
            if current_name != exiting:
                queue.extend(edges[current_name])

        scfg = SCFG(scfg_graph, name_gen=name_gen)
        return scfg

    @staticmethod
    def to_yaml(scfg: "SCFG") -> str:
        """Helper method to convert the SCFG object to a YAML
        string representation.

        The method returns a YAML string representing the control
        flow graph. It iterates over the graph dictionary and
        generates YAML entries for each block, including jump
        targets and backedges.

        Parameters
        ----------
        scfg: SCFG
            The `SCFG` object to be transformed.

        Returns
        -------
        yaml: str
            A YAML string representing the SCFG.
        """
        # Convert to yaml
        ys = ""

        graph_dict = SCFGIO.to_dict(scfg)

        blocks = graph_dict["blocks"]
        edges = graph_dict["edges"]
        backedges = graph_dict["backedges"]

        ys += "\nblocks:\n"
        for b in sorted(blocks):
            ys += indent(f"'{b}':\n", " " * 8)
            for k, v in blocks[b].items():
                ys += indent(f"{k}: {v}\n", " " * 12)

        ys += "\nedges:\n"
        for b in sorted(blocks):
            ys += indent(f"'{b}': {edges[b]}\n", " " * 8)

        ys += "\nbackedges:\n"
        for b in sorted(blocks):
            if backedges[b]:
                ys += indent(f"'{b}': {backedges[b]}\n", " " * 8)
        return ys

    @staticmethod
    def to_dict(scfg: "SCFG") -> Dict[str, Dict[str, Any]]:
        """Helper method to convert the SCFG object to a dictionary
        representation.

        This method returns a dictionary representing the control flow
        graph. It iterates over the graph dictionary and generates a
        dictionary entry for each block, including jump targets and
        backedges if present.

        Parameters
        ----------
        scfg: SCFG
            The `SCFG` object to be transformed.

        Returns
        -------
        graph_dict: Dict[Dict[...]]
            A dictionary representing the SCFG.
        """
        blocks: Dict[str, Any] = {}
        edges, backedges = {}, {}

        def reverse_lookup(value: type) -> str:
            for k, v in block_type_names.items():
                if v == value:
                    return k
            else:
                raise TypeError("Block type not found.")

        seen = set()
        q: Set[Tuple[str, BasicBlock]] = set()
        # Order of elements doesn't matter since they're going to
        # be sorted at the end.
        q.update(scfg.graph.items())

        while q:
            key, value = q.pop()
            if key in seen:
                continue
            seen.add(key)

            block_type = reverse_lookup(type(value))
            blocks[key] = {"type": block_type}
            if isinstance(value, RegionBlock):
                assert value.subregion is not None
                assert value.parent_region is not None
                q.update(value.subregion.graph.items())
                blocks[key]["kind"] = value.kind
                blocks[key]["contains"] = sorted(
                    [idx.name for idx in value.subregion.graph.values()]
                )
                blocks[key]["header"] = value.header
                blocks[key]["exiting"] = value.exiting
                blocks[key]["parent_region"] = value.parent_region.name
            elif isinstance(value, SyntheticBranch):
                blocks[key]["branch_value_table"] = value.branch_value_table
                blocks[key]["variable"] = value.variable
            elif isinstance(value, SyntheticAssignment):
                blocks[key]["variable_assignment"] = value.variable_assignment
            elif isinstance(value, PythonBytecodeBlock):
                blocks[key]["begin"] = value.begin
                blocks[key]["end"] = value.end
            edges[key] = sorted([i for i in value._jump_targets])
            backedges[key] = sorted([i for i in value.backedges])

        graph_dict = {"blocks": blocks, "edges": edges, "backedges": backedges}

        return graph_dict

    @staticmethod
    def find_outer_graph(graph_dict: Dict[str, Dict[str, Any]]) -> Set[str]:
        """Helper method to find the outermost graph components
        of an `SCFG` object. (i.e. Components that aren't
        contained in any other region)

        Parameters
        ----------
        graph_dict: dict
            The input dictionary from which the SCFG is to be constructed.

        Return
        ------
        outer_blocks: set[str]
            Set of all the block names that lie in the outer most graph or
            aren't a part of any region.
        """
        blocks = graph_dict["blocks"]

        outer_blocks = set(blocks.keys())
        for _, block_data in blocks.items():
            if block_data.get("contains"):
                outer_blocks.difference_update(block_data["contains"])

        return outer_blocks

    @staticmethod
    def extract_block_info(
        blocks: Dict[str, Dict[str, Any]],
        current_name: str,
        block_ref_dict: Dict[str, str],
        edges: Dict[str, List[str]],
        backedges: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Any], str, Tuple[str, ...], Tuple[str, ...]]:
        """Helper method to extract information from various components of
        an `SCFG` graph.

        Parameters
        ----------
        blocks: Dict[str, dict]
            Dictionary containing all the blocks info
        current_name: str
            Name of the block whose information is to be extracted.
        block_ref_dict: Dict[str, str]
            Dictionary of block names in YAML string corresponding to their
            representation/unique name IDs in the SCFG.
        edges: Dict[str, list[str]]
            Dictionary representing the edges of the graph.
        backedges: Dict[str, list[str]]
            Dictionary representing the backedges of the graph.

        Return
        ------
        block_info: Dict[str, Any]
            Dictionary containing information about the block.
        block_type: str
            String representing the type of block.
        block_edges: List[str]
            List of edges of the requested block.
        block_backedges: List[str]
            List of backedges of the requested block.
        """
        block_info = blocks[current_name].copy()
        block_edges = tuple(block_ref_dict[idx] for idx in edges[current_name])

        if backedges.get(current_name):
            block_backedges = tuple(
                block_ref_dict[idx] for idx in backedges[current_name]
            )
        else:
            block_backedges = ()

        block_type = block_info.pop("type")

        return block_info, block_type, block_edges, block_backedges


class AbstractGraphView(
    Mapping[str, BasicBlock]
):  # todo: improve this annotation
    """Abstract Graph View class.

    The AbstractGraphView class serves as a template for graph views.
    """

    def __getitem__(self, item: str) -> BasicBlock:
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

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the name of blocks in the graph view.

        Returns
        -------
        blocks: iter of str
            An iterator over blocks (or regions) over the given view.
        """
        raise NotImplementedError

    def __len__(self) -> int:
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

    scfg: SCFG

    def __init__(self, scfg: SCFG) -> None:
        """Initializes the ConcealedRegionView with the given SCFG.

        Parameters
        ----------
        scfg: SCFG
            The SCFG for which to instantiate the view.
        """
        self.scfg = scfg

    def __getitem__(self, item: str) -> BasicBlock:
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

    def __iter__(self) -> Iterator[str]:
        """Returns an iterator over the name of blocks in the concealed
        graph view.

        Returns
        -------
        blocks: iter of str
            An iterator over blocks (or regions) over the given view.
        """
        return self.region_view_iterator()

    def region_view_iterator(
        self, head: Optional[str] = None
    ) -> Iterator[str]:
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
            if type(block) == RegionBlock:  # noqa: E721
                # If this is a region, continue on to the exiting block, i.e.
                # the region is presented a single fall-through block to the
                # consumer of this iterator.
                assert block.subregion is not None
                to_visit.extend(block.subregion[block.exiting].jump_targets)
            else:
                # otherwise add any jump_targets to the list of names to visit
                to_visit.extend(block.jump_targets)

            # finally, yield the name
            yield name

    def __len__(self) -> int:
        """Returns the number of elements in the concealed region view.

        Return
        ------
        len: int
            Length/ of given SCFG view (number of blocks in the concealed
            SCFG view).
        """
        return len(self.scfg)
