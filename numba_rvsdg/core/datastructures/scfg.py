import yaml
import itertools

from textwrap import dedent
from typing import Set, Tuple, Dict, List, Iterator
from dataclasses import dataclass, field
from collections import deque

from numba_rvsdg.core.datastructures.basic_block import BasicBlock, get_block_class, get_block_class_str
from numba_rvsdg.core.datastructures.region import Region
from numba_rvsdg.core.datastructures.labels import *

default_label = Label()


@dataclass(frozen=True)
class SCFG:
    # This maps the names of the blocks to the block objects, which may contain
    # additional information such as assembly instructions or Python byetcode
    blocks: Dict[str, BasicBlock] = field(default_factory=dict, init=False)

    # This is the actual graph, contains a mapping of block names to block
    # names they point to, i.e. the edges of the graph. Any block name in the
    # list of block names must also be a key in this mapping. I.e. this mapping
    # must be self-contained.
    out_edges: Dict[str, List[str]] = field(default_factory=dict, init=False)

    # These are any identified back-edges. This must also be self contained.
    back_edges: set[tuple[str, str]] = field(default_factory=set, init=False)

    # This is the top-level region
    meta_region: str = field(init=False)

    # This is the region storage. It maps region names to the Region objects,
    # which themselves contain the header and exiting blocks of this region
    regions: Dict[str, Region] = field(default_factory=dict, init=False)

    # This is the region tree which stores the hierarchical relationship
    # between regions. The root node of this tree is the meta-region (once the
    # graph is fully analysed).
    region_tree: Dict[str, set[str]] = field(default_factory=dict, init=False)

    name_gen: NameGenerator = field(default_factory=NameGenerator, compare=False, init=False)

    def __post_init__(self):
        new_region = Region(name_gen = self.name_gen, kind="meta", header=None, exiting=None)
        region_name = new_region.region_name
        self.regions[region_name] = new_region
        self.region_tree[region_name] = set()
        object.__setattr__(self, "meta_region", region_name)

    def __getitem__(self, index: str) -> BasicBlock:
        return self.blocks[index]

    def __contains__(self, index: str) -> bool:
        return index in self.blocks

    def __iter__(self):
        """Graph Iterator"""
        # initialise housekeeping datastructures
        to_visit, seen = [self.find_head()], set()
        while to_visit:
            # get the next name on the list
            name = to_visit.pop(0)
            # if we have visited this, we skip it
            if name in seen:
                continue
            else:
                seen.add(name)
            # yield the name, block combo
            yield name
            # finally add any out_edges to the list of names to visit
            to_visit.extend(self.out_edges[name])

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
            for jt in self.out_edges[name]:
                heads.discard(jt)
        assert len(heads) == 1
        return next(iter(heads))

    def compute_scc_subgraph(self, subgraph) -> List[Set[str]]:
        """
        Strongly-connected component for detecting loops inside a subgraph.
        """
        from numba_rvsdg.networkx_vendored.scc import scc

        scfg = self

        class GraphWrap:
            def __init__(self, graph: Dict[str, BasicBlock], subgraph):
                self.graph = graph
                self.subgraph = subgraph

            def __getitem__(self, vertex):

                out = scfg.out_edges[vertex]
                # Exclude node outside of the subgraph
                return [k for k in out if k in subgraph
                        and not (vertex, k) in scfg.back_edges]

            def __iter__(self):
                return iter(self.graph.keys())

        return list(scc(GraphWrap(self.blocks, subgraph)))

    def find_headers_and_entries(
            self,
            subgraph: set[str]
            ) -> Tuple[list[str], list[str]]:
        """Find entries and headers in a given subgraph.

        Entries are blocks outside the subgraph that have an edge pointing to
        the subgraph headers. Headers are blocks that are part of the strongly
        connected subset and that have incoming edges from outside the
        subgraph. Entries point to headers and headers are pointed to by
        entries.

        Parameters
        ----------
        subgraph: set of str
            The subgraph for which to find the headers and entries

        Returns
        -------
        headers: list of str
            The headers for this subgraph
        entries:
            The entries for this subgraph

        Notes
        -----
        The returned lists of headers and entries are sorted.
        """
        outside: str
        entries: set[str] = set()
        headers: set[str] = set()
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
        return sorted(headers), sorted(entries)

    def find_exiting_and_exits(
        self, subgraph: Set[str]
    ) -> Tuple[list[str], list[str]]:
        """Find exiting and exit blocks in a given subgraph.

        Exiting blocks are blocks inside the subgraph that have edges to
        blocks outside of the subgraph. Exit blocks are blocks outside the
        subgraph that have incoming edges from within the subgraph. Exiting
        blocks point to exits and exits and pointed to by exiting blocks.

        Parameters
        ----------
        subgraph: set of str
            The subgraph for which to find the exiting and exit blocks.

        Returns
        -------
        exiting: list of str
            The exiting blocks for this subgraph
        exits:
            The exit block for this subgraph

        Notes
        -----
        The returned lists of exiting and exit blocks are sorted.

        """
        inside: str
        # use sets internally to avoid duplicates
        exiting: set[str] = set()
        exits: set[str] = set()
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

    def is_reachable_dfs(self, begin: str, end: str):  # -> TypeGuard:
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

    def is_exiting(self, block_name: str):
        return len(self.out_edges[block_name]) == 0

    def is_fallthrough(self, block_name: str):
        return len(self.out_edges[block_name]) == 1

    def check_graph(self):
        pass

    def insert_block_between(
        self,
        block_name: str,
        predecessors: List[str],
        successors: List[str]
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
        self, block_type: str = "basic", block_label: Label = default_label, **block_args
    ) -> str:
        block_type = get_block_class(block_type)
        new_block: BasicBlock = block_type(**block_args, label=block_label, name_gen=self.name_gen)

        name = new_block.block_name
        self.blocks[name] = new_block
        self.out_edges[name] = []
        return name

    def add_region(self, parent_region: str, kind: str, header: str, exiting: str):
        assert kind in ["loop", "head", "tail", "branch"]
        assert parent_region in self.regions.keys()
        new_region = Region(name_gen=self.name_gen, kind=kind, header=header, exiting=exiting)
        region_name = new_region.region_name
        self.regions[region_name] = new_region
        self.region_tree[region_name] = set()
        self.region_tree[parent_region].add(region_name)

        return region_name

    def add_connections(self, block_name, out_edges):
        assert self.out_edges[block_name] == []
        self.out_edges[block_name] = out_edges
        self.check_graph()


    @staticmethod
    def from_yaml(yaml_string):
        data = yaml.safe_load(yaml_string)
        return SCFG.from_dict(data)

    @staticmethod
    def from_dict(graph_dict: Dict[str, Dict]):
        label_types = {
            "label": Label,
            "python_bytecode": PythonBytecodeLabel,
            "control": ControlLabel,
            "synth_branch": SyntheticBranch,
            "synth_tail": SyntheticTail,
            "synth_exit": SyntheticExit,
            "synth_head": SyntheticHead,
            "synth_return": SyntheticReturn,
            "synth_latch": SyntheticLatch,
            "synth_exit_latch": SyntheticExitingLatch,
            "synth_assign": SynthenticAssignment,
        }

        def get_label_class(label_type_string):
            if label_type_string in label_types:
                return label_types[label_type_string]
            else:
                raise TypeError(f"Block Type {label_type_string} not recognized.")

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

    def block_view(self, region_name: str):
        """All blocks in a given region.
        Out edges within the region.
        """
        if region_name == self.meta_region:
            header = self.find_head()
            exiting  = None
        else:
            header = self.regions[region_name].header
            exiting = self.regions[region_name].exiting

        # initialise housekeeping datastructures
        to_visit, all_blocks, all_outs = [header], set(), dict()

        while to_visit:
            # get the next name on the list
            name = to_visit.pop(0)
            # if we have visited this, we skip it
            if name in all_blocks:
                continue
            
            # Otherwise we process the valid block
            all_blocks.add(name)
            if name is exiting:
                # This ignores back-edge case?
                all_outs[name] = []
                continue
            to_visit.extend(self.out_edges[name])
            all_outs[name] = self.out_edges[name]
        return all_blocks, all_outs

    def region_view(self, region_name: str):
        """ All subregions + non-subregion blocks.
        Out edges within the region.
        Backedges of subregions shown as an edge pointing to the subregion itself.
        """
        all_blocks, all_outs = self.block_view(region_name)
        all_headers = {}

        for subregion in self.region_tree[region_name]:
            subregion_blocks, _ = self.block_view(subregion)
            all_blocks.difference_update(subregion_blocks)
            all_blocks.add(subregion)
            subregion_header = self.regions[subregion].header
            subregion_exiting = self.regions[subregion].exiting
            all_outs[subregion] = self.out_edges[subregion_exiting]
            all_headers[subregion_header] = subregion
 
        for _out in list(all_outs.keys()):
            if _out not in all_blocks:
                all_outs.pop(_out)
            else:
                for idx, _out_item in enumerate(all_outs[_out]):
                    if _out_item in all_headers.keys():
                        all_outs[_out][idx] = all_headers[_out_item]
        return all_blocks, all_outs
