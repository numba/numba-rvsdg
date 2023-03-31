import yaml
import itertools

from textwrap import dedent
from typing import Set, Tuple, Dict, List, Iterator
from dataclasses import dataclass, field

from numba_rvsdg.core.datastructures.basic_block import BasicBlock, get_block_class, get_block_class_str
from numba_rvsdg.core.datastructures.region import MetaRegion, Region, LoopRegion, get_region_class
from numba_rvsdg.core.datastructures.labels import (
    Label,
    BlockName,
    NameGenerator,
    RegionName,
    MetaRegionLabel, LoopRegionLabel, RegionLabel,
    get_label_class,
)


@dataclass(frozen=True)
class SCFG:
    """Maps of BlockNames to respective BasicBlocks.
    And stores the jump targets and back edges for
    blocks within the graph."""

    blocks: Dict[BlockName, BasicBlock] = field(default_factory=dict, init=False)

    out_edges: Dict[BlockName, List[BlockName]] = field(default_factory=dict, init=False)
    back_edges: set[tuple[BlockName, BlockName]] = field(default_factory=set, init=False)

    regions: Dict[RegionName, Region] = field(default_factory=dict, init=False)
    meta_region: RegionName = field(init=False)
    region_tree: Dict[BlockName, List[RegionName]] = field(default_factory=dict, init=False)

    name_gen: NameGenerator = field(default_factory=NameGenerator, compare=False, init=False)

    def __post_init__(self):
        new_region = MetaRegion(name_gen = self.name_gen, label = MetaRegionLabel())
        region_name = new_region.region_name
        self.regions[region_name] = new_region
        self.region_tree[region_name] = []
        object.__setattr__(self, "meta_region", region_name)

    def __getitem__(self, index: BlockName) -> BasicBlock:
        return self.blocks[index]

    def __contains__(self, index: BlockName) -> bool:
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
            if isinstance(name, RegionName):
                name = self.regions[name].header
            # yield the name, block combo
            yield name
            # finally add any out_edges to the list of names to visit
            to_visit.extend(self.out_edges[name])

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
        for _, region in self.regions.items():
            if hasattr(region, "header"):
                heads.discard(region.header)
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
            self,
            subgraph: set[BlockName]
            ) -> Tuple[list[BlockName], list[BlockName]]:
        """Find entries and headers in a given subgraph.

        Entries are blocks outside the subgraph that have an edge pointing to
        the subgraph headers. Headers are blocks that are part of the strongly
        connected subset and that have incoming edges from outside the
        subgraph. Entries point to headers and headers are pointed to by
        entries.

        Parameters
        ----------
        subgraph: set of BlockName
            The subgraph for which to find the headers and entries

        Returns
        -------
        headers: list of BlockName
            The headers for this subgraph
        entries:
            The entries for this subgraph

        Notes
        -----
        The returned lists of headers and entries are sorted.
        """
        outside: BlockName
        entries: set[BlockName] = set()
        headers: set[BlockName] = set()
        # Iterate over all blocks in the graph, excluding any blocks inside the
        # subgraph.
        for outside in self.exclude_blocks(subgraph):
            # Check if the current block points to any blocks that are inside
            # the subgraph.
            targets_in_loop = subgraph.intersection(self.out_edges[outside])
            # Record both headers and entries
            if targets_in_loop:
                headers.update(targets_in_loop)
                entries.add(outside)
        # If the loop has no headers or entries, the only header is the head of
        # the CFG.
        if not headers:
            headers.add(self.find_head())
        return sorted(list(headers)), sorted(list(entries))

    def find_exiting_and_exits(
        self, subgraph: Set[BlockName]
    ) -> Tuple[list[BlockName], list[BlockName]]:
        """Find exiting and exit blocks in a given subgraph.

        Exiting blocks are blocks inside the subgraph that have edges to
        blocks outside of the subgraph. Exit blocks are blocks outside the
        subgraph that have incoming edges from within the subgraph. Exiting
        blocks point to exits and exits and pointed to by exiting blocks.

        Parameters
        ----------
        subgraph: set of BlockName
            The subgraph for which to find the exiting and exit blocks.

        Returns
        -------
        exiting: list of BlockName
            The exiting blocks for this subgraph
        exits:
            The exit block for this subgraph

        Notes
        -----
        The returned lists of exiting and exit blocks are sorted.

        """
        inside: BlockName
        # use sets internally to avoid duplicates
        exiting: set[BlockName] = set()
        exits: set[BlockName] = set()
        for inside in subgraph:
            # any node inside that points outside the loop
            for out_target in self.out_edges[inside]:
                if out_target not in subgraph:
                    exiting.add(inside)
                    exits.add(out_target)
            # any returns
            if self.is_exiting(inside):
                exiting.add(inside)
        # convert to sorted list before return
        return sorted(exiting), sorted(exits)

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
        self.out_edges[name] = []

        return name

    def add_region(self, kind: str, header: BlockName, exiting: BlockName, parent: Region = None, region_label = RegionLabel()):
        if parent is None:
            parent = self.meta_region

        region_type = get_region_class(kind)
        new_region: Region = region_type(name_gen=self.name_gen, label=region_label, header=header, exiting=exiting)
        region_name = new_region.region_name
        self.regions[region_name] = new_region
        self.region_tree[region_name] = []

        self.region_tree[parent].append(region_name)

        for block, out_edges in self.out_edges.items():
            for idx, edge in enumerate(out_edges):
                if edge == header and block is not exiting:
                    self.out_edges[block][idx] = region_name

        return region_name

    def add_connections(self, block_name, out_edges=[]):
        assert self.out_edges[block_name] == []
        self.out_edges[block_name] = out_edges

        self.check_graph()


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
            scfg.add_connections(block_name, out_edges)
            for _back in back_refs:
                scfg.back_edges.add((ref_dict[block_ref], ref_dict[_back]))


        scfg.check_graph()
        return scfg, ref_dict

    def to_yaml(self):
        # Convert to yaml
        yaml_string = """"""

        for key, value in self.blocks.items():
            out_edges = []
            back_edges = []
            for out_edge in self.out_edges[key]:
                out_edges.append(f"{out_edge}")
                if (key, out_edge) in self.back_edges:
                    back_edges.append(f"{out_edge}")
            out_edges = str(out_edges).replace("'", '"')
            jump_target_str = f"""
                "{str(key)}":
                    type: "{get_block_class_str(value)}"
                    out: {out_edges}"""

            if back_edges:
                back_edges = str(back_edges).replace("'", '"')
                jump_target_str += f"""
                    back: {back_edges}"""
            yaml_string += dedent(jump_target_str)

        return yaml_string

    def to_dict(self):
        graph_dict = {}
        for key, value in self.blocks.items():
            curr_dict = {}
            curr_dict["type"] = get_block_class_str(value)
            curr_dict["out"] = []
            back_edges = []
            for out_edge in self.out_edges[key]:
                curr_dict["out"].append(f"{out_edge}")
                if (key, out_edge) in self.back_edges:
                    back_edges.append(f"{out_edge}")
            if back_edges:
                curr_dict["back"] = back_edges
            graph_dict[str(key)] = curr_dict

        return graph_dict

    def iterate_region(self, region_name, region_view=False):
        if region_name == self.meta_region and not region_view:
            return iter(self)

        region = self.regions[region_name]
        """Region Iterator"""
        region_head = region.header if region_name is not self.meta_region else self.find_head()

        # initialise housekeeping datastructures
        to_visit, seen = [region_head], []
        while to_visit:
            # get the next block_name on the list
            block_name = to_visit.pop(0)
            # if we have visited this, we skip it
            if block_name in seen:
                continue
            else:
                seen.append(block_name)
            # yield the block_name
            yield block_name
            if region_name is not self.meta_region and block_name is region.exiting:
                continue
            # finally add any out_edges to the list of block_names to visit
            outs = self.out_edges[block_name]

            if not region_view:
                for idx, _out in enumerate(outs):
                    if isinstance(_out, RegionName):
                        to_visit.append(_out.header)
                    else:
                        to_visit.append(_out)
            else:
                to_visit.extend(outs)
