import yaml
import itertools

from textwrap import dedent
from typing import Set, Tuple, Dict, List, Iterator
from dataclasses import dataclass, field

from numba_rvsdg.core.datastructures.basic_block import BasicBlock, get_block_class, get_block_class_str
from numba_rvsdg.core.datastructures.region import Region
from numba_rvsdg.core.datastructures.labels import (
    Label,
    BlockName,
    NameGenerator,
    RegionName,
    get_label_class,
)


@dataclass(frozen=True)
class SCFG:
    """Maps of BlockNames to respective BasicBlocks.
    And stores the jump targets and back edges for
    blocks within the graph."""

    blocks: Dict[BlockName, BasicBlock] = field(default_factory=dict)

    out_edges: Dict[BlockName, List[BlockName]] = field(default_factory=dict)
    back_edges: Dict[BlockName, List[BlockName]] = field(default_factory=dict)
    regions: Dict[RegionName, Region] = field(default_factory=dict)

    name_gen: NameGenerator = field(default_factory=NameGenerator, compare=False)

    def __getitem__(self, index: BlockName) -> BasicBlock:
        return self.blocks[index]

    def __contains__(self, index: BlockName) -> bool:
        return index in self.blocks

    def __iter__(self):
        """Graph Iterator"""
        # initialise housekeeping datastructures
        to_visit, seen = [self.find_head()], []
        while to_visit:
            # get the next block_name on the list
            block_name = to_visit.pop(0)
            # if we have visited this, we skip it
            if block_name in seen:
                continue
            else:
                seen.append(block_name)
            # get the corresponding block for the block_name
            block = self[block_name]
            # yield the block_name, block combo
            yield (block_name, block)
            # finally add any out_edges to the list of block_names to visit
            to_visit.extend(self.out_edges[block_name])

    def exclude_blocks(self, exclude_blocks: Set[BlockName]) -> Iterator[BlockName]:
        """Iterator over all nodes not in exclude_blocks."""
        for block in self.blocks:
            if block not in exclude_blocks:
                yield block

    def find_head(self) -> BlockName:
        """Find the head block of the CFG.

        Assuming the CFG is closed, this will find the block
        that no other blocks are pointing to.

        """
        heads = set(self.blocks.keys())
        for name in self.blocks.keys():
            for jt in self.out_edges[name]:
                heads.discard(jt)
        assert len(heads) == 1
        return next(iter(heads))

    def compute_scc(self) -> List[Set[BlockName]]:
        """
        Strongly-connected component for detecting loops.
        """
        from numba_rvsdg.networkx_vendored.scc import scc

        out_edges = self.out_edges

        class GraphWrap:
            def __init__(self, graph):
                self.graph = graph

            def __getitem__(self, vertex):
                out = out_edges[vertex]
                # Exclude node outside of the subgraph
                return [k for k in out if k in self.graph]

            def __iter__(self):
                return iter(self.graph.keys())

        return list(scc(GraphWrap(self.blocks)))

    def compute_scc_subgraph(self, subgraph) -> List[Set[BlockName]]:
        """
        Strongly-connected component for detecting loops inside a subgraph.
        """
        from numba_rvsdg.networkx_vendored.scc import scc

        out_edges = self.out_edges

        class GraphWrap:
            def __init__(self, graph: Dict[BlockName, BasicBlock], subgraph):
                self.graph = graph
                self.subgraph = subgraph

            def __getitem__(self, vertex):
                out = out_edges[vertex]
                # Exclude node outside of the subgraph
                return [k for k in out if k in subgraph]

            def __iter__(self):
                return iter(self.graph.keys())

        return list(scc(GraphWrap(self.blocks, subgraph)))

    def find_headers_and_entries(
        self, subgraph: Set[BlockName]
    ) -> Tuple[Set[BlockName], Set[BlockName]]:
        """Find entries and headers in a given subgraph.

        Entries are blocks outside the subgraph that have an edge pointing to
        the subgraph headers. Headers are blocks that are part of the strongly
        connected subset and that have incoming edges from outside the
        subgraph. Entries point to headers and headers are pointed to by
        entries.

        """
        outside: BlockName
        entries: Set[BlockName] = set()
        headers: Set[BlockName] = set()

        for outside in self.exclude_blocks(subgraph):
            nodes_jump_in_loop = subgraph.intersection(
                self.out_edges[outside]
            )
            headers.update(nodes_jump_in_loop)
            if nodes_jump_in_loop:
                entries.add(outside)
        # If the loop has no headers or entries, the only header is the head of
        # the CFG.
        if not headers:
            headers = {self.find_head()}
        return headers, entries

    def find_exiting_and_exits(
        self, subgraph: Set[BlockName]
    ) -> Tuple[Set[BlockName], Set[BlockName]]:
        """Find exiting and exit blocks in a given subgraph.

        Existing blocks are blocks inside the subgraph that have edges to
        blocks outside of the subgraph. Exit blocks are blocks outside the
        subgraph that have incoming edges from within the subgraph. Exiting
        blocks point to exits and exits and pointed to by exiting blocks.

        """
        inside: BlockName
        exiting: Set[BlockName] = set()
        exits: Set[BlockName] = set()
        for inside in subgraph:
            # any node inside that points outside the loop
            for jt in self.out_edges[inside]:
                if jt not in subgraph:
                    exiting.add(inside)
                    exits.add(jt)
            # any returns
            if self.is_exiting(inside):
                exiting.add(inside)
        return exiting, exits

    def is_reachable_dfs(self, begin: BlockName, end: BlockName):  # -> TypeGuard:
        """Is end reachable from begin."""
        seen = set()
        to_vist = list(self.out_edges[begin])
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
                    to_vist.extend(self.out_edges[block])

    def is_exiting(self, block_name: BlockName):
        return len(self.out_edges[block_name]) == 0

    def is_fallthrough(self, block_name: BlockName):
        return len(self.out_edges[block_name]) == 1

    def check_graph(self):
        pass

    # We don't need this cause everything is 'hopefully' additive
    # def remove_blocks(self, names: Set[BlockName]):
    #     for name in names:
    #         del self.blocks[name]
    #         del self.out_edges[name]
    #         del self.back_edges[name]
    #     self.check_graph()

    def insert_block_between(
        self,
        block_name: BlockName,
        predecessors: List[BlockName],
        successors: List[BlockName]
    ):
        # Replace any arcs from any of predecessors to any of successors with
        # an arc through the inserted block instead.
        for pred_name in predecessors:
            # For every predecessor
            # Add the inserted block as out edge
            for idx, _out in enumerate(self.out_edges[pred_name]):
                if _out in successors:
                    self.out_edges[pred_name][idx] = block_name

            if block_name not in self.out_edges[pred_name]:
                self.out_edges[pred_name].append(block_name)

            self.out_edges[pred_name] = list(dict.fromkeys(self.out_edges[pred_name]))

        for success_name in successors:
            # For every sucessor
            # For inserted block, the sucessor in an out-edge
            self.out_edges[block_name].append(success_name)

        self.check_graph()

    def add_block(
        self, block_type: str = "basic", block_label: Label = Label(), **block_args
    ) -> BlockName:
        block_type = get_block_class(block_type)
        new_block: BasicBlock = block_type(**block_args, label=block_label, name_gen=self.name_gen)

        name = new_block.block_name
        self.blocks[name] = new_block

        self.back_edges[name] = []
        self.out_edges[name] = []

        return name

    def add_connections(self, block_name, out_edges=[], back_edges=[]):
        assert self.out_edges[block_name] == []
        assert self.back_edges[block_name] == []
        self.out_edges[block_name] = out_edges
        self.back_edges[block_name] = back_edges

        self.check_graph()

    def add_region(self, region_head, region_exit, kind):
        new_region = Region(self.name_gen, kind, region_head, region_exit)
        self.regions[new_region.region_name] = new_region

    @staticmethod
    def from_yaml(yaml_string):
        data = yaml.safe_load(yaml_string)
        return SCFG.from_dict(data)

    @staticmethod
    def from_dict(graph_dict: Dict[str, Dict]):
        scfg = SCFG()
        ref_dict = {}

        for block_ref, block_attrs in graph_dict.items():
            block_class = block_attrs["type"]
            block_args = block_attrs.get("block_args", {})
            label_class = get_label_class(block_attrs.get("label_type", "label"))
            label_info = block_attrs.get("label_info", None)
            block_label = label_class(label_info)
            block_name = scfg.add_block(block_class, block_label, **block_args)
            ref_dict[block_ref] = block_name

        for block_ref, block_attrs in graph_dict.items():
            out_refs = block_attrs.get("out", list())
            back_refs = block_attrs.get("back", list())

            block_name = ref_dict[block_ref]
            out_edges = list(ref_dict[out_ref] for out_ref in out_refs)
            back_edges = list(ref_dict[back_ref] for back_ref in back_refs)
            scfg.add_connections(block_name, out_edges, back_edges)

        scfg.check_graph()
        return scfg, ref_dict

    def to_yaml(self):
        # Convert to yaml
        yaml_string = """"""

        for key, value in self.blocks.items():
            out_edges = [f"{i}" for i in self.out_edges[key]]
            # out_edges = str(out_edges).replace("'", '"')
            back_edges = [f"{i}" for i in self.back_edges[key]]
            jump_target_str = f"""
                "{str(key)}":
                    type: "{get_block_class_str(value)}"
                    out: {out_edges}"""

            if back_edges:
                # back_edges = str(back_edges).replace("'", '"')
                jump_target_str += f"""
                    back: {back_edges}"""
            yaml_string += dedent(jump_target_str)

        return yaml_string

    def to_dict(self):
        graph_dict = {}
        for key, value in self.blocks.items():
            curr_dict = {}
            curr_dict["type"] = get_block_class_str(value)
            curr_dict["out"] = [f"{i}" for i in self.out_edges[key]]
            back_edges = [f"{i}" for i in self.back_edges[key]]
            if back_edges:
                curr_dict["back"] = back_edges
            graph_dict[str(key)] = curr_dict

        return graph_dict
