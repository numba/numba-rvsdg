import dis
import yaml
from textwrap import dedent
from typing import Set, Tuple, Dict, List, Iterator
from dataclasses import dataclass, field
from collections import deque

from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    SyntheticBlock,
    SyntheticAssignment,
    SyntheticHead,
    SyntheticExit,
    SyntheticTail,
    SyntheticReturn,
    RegionBlock,
    Region
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

    graph: Dict[str, BasicBlock] = field(default_factory=dict)
    name_gen: NameGenerator = field(
        default_factory=NameGenerator, compare=False
    )

    # This is the top-level region
    meta_region: str = field(init=False)

    # This is the region storage. It maps region names to the Region objects,
    # which themselves contain the header and exiting blocks of this region
    regions: Dict[str, Region] = field(default_factory=dict, init=False)

    def __post_init__(self):
        name = self.name_gen.new_region_name("meta")
        new_region = Region(name = name, kind="meta", header=None, exiting=None, parent=None)
        region_name = new_region.name
        self.regions[region_name] = new_region
        object.__setattr__(self, "meta_region", region_name)

    def __getitem__(self, index):
        return self.graph[index]

    def __contains__(self, index):
        return index in self.graph

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
            # finally add any jump_targets to the list of names to visit
            to_visit.extend(block.jump_targets)

    def region_view(self, region=None):
        return RegionView(self, region)

    def block_view(self, region=None):
        return BlockView(self, region)

    def exclude_blocks(self, exclude_blocks: Set[str]) -> Iterator[str]:
        """Iterator over all nodes not in exclude_blocks."""
        for block in self.graph:
            if block not in exclude_blocks:
                yield block

    def add_block(self, basicblock: BasicBlock):
        self.graph[basicblock.name] = basicblock
    
    def make_region(self, header, exiting, region_kind, parent_region=None):
        assert len(header) == 1
        assert len(exiting) == 1
        region_header = next(iter(header))
        region_exiting = next(iter(exiting))
        region_name = self.name_gen.new_region_name(region_kind)

        if parent_region is None:
            parent_region = self.meta_region
        
        subregion = Region(
            name=region_name,
            kind=region_kind,
            headers=region_header,
            exiting=region_exiting,
            parent=parent_region
        )
        self.regions[region_name] = subregion
        return region_name

    def remove_blocks(self, names: Set[str]):
        for name in names:
            del self.graph[name]

    def _insert_block(
        self, new_name: str, predecessors: Set[str], successors: Set[str],
        block_type: SyntheticBlock
    ):
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

    def to_yaml(self):
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
        scfg_graph = self.graph
        graph_dict = {}
        for key, value in scfg_graph.items():
            curr_dict = {}
            curr_dict["jt"] = [i for i in value._jump_targets]
            if value.backedges:
                curr_dict["be"] = [i for i in value.backedges]
            graph_dict[key] = curr_dict
        return graph_dict


class AbstractGraphView:
    scfg: SCFG
    region_name: str

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def find_head(self) -> str:
        """Find the head block of the CFG.

        Assuming the CFG is closed, this will find the block
        that no other blocks are pointing to.
        """
        heads = self.blocks
        for name in self._jump_targets.keys():
            for jt in self._jump_targets[name]:
                heads.discard(jt)
        assert len(heads) == 1
        return next(iter(heads))

    def compute_scc(self) -> List[Set[str]]:
        """
        Strongly-connected component for detecting loops.
        """
        from numba_rvsdg.networkx_vendored.scc import scc
        jump_targets = self.jump_targets
        blocks = self.blocks

        class GraphWrap:
            def __getitem__(self, vertex):
                out = jump_targets[vertex]
                # Exclude node outside of the subgraph
                return [k for k in out if k in blocks]

            def __iter__(self):
                return iter(blocks)

        return list(scc(GraphWrap()))

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
            nodes_jump_in_loop = subgraph.intersection(self.graph[outside].jump_targets)
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
            for jt in self.graph[inside].jump_targets:
                if jt not in subgraph:
                    exiting.add(inside)
                    exits.add(jt)
            # any returns
            if self.graph[inside].is_exiting:
                exiting.add(inside)
        return sorted(exiting), sorted(exits)

    def is_reachable_dfs(self, begin: str, end: str):  # -> TypeGuard:
        """Is end reachable from begin."""
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

    def get_region_blocks(self, region_name):
        if region_name == self.scfg.meta_region:
            return set(block for _, block in self.scfg.graph.items())

        region = self.scfg.regions[region_name]
        header = region.header
        exiting = region.exiting
        region_blocks = set()

        # initialise housekeeping datastructures
        to_visit, seen = [header], []
        while to_visit:
            # get the next name on the list
            name = to_visit.pop(0)
            # if we have visited this, we skip it
            if name in seen or name in exiting:
                continue
            else:
                seen.append(name)
            # get the corresponding block for the name
            block = self.scfg[name]
            region_blocks.add(block)
            # finally add any jump_targets to the list of names to visit
            to_visit.extend(block.jump_targets)


class RegionView(AbstractGraphView):
    def __init__(self, scfg, region_name):
        self.scfg = scfg
        if region_name is None:
            region_name = self.scfg.meta_region
            self.region_name = region_name
        assert region_name in self.scfg.regions.keys()
    
    def _jump_targets(self, block):
        pass

    def jump_targets(self, block):
        pass

    def back_edges(self, block):
        pass

    def __iter__(self):
        pass

    def __len__(self):
        pass


class BlockView(AbstractGraphView):

    def __init__(self, scfg, region_name):
        self.scfg = scfg
        if region_name is None:
            region_name = self.scfg.meta_region
            self.region_name = region_name
        assert region_name in self.scfg.regions.keys()
    
    def _jump_targets(self, block):
        pass

    def jump_targets(self, block):
        pass

    def back_edges(self, block):
        pass
    
    def __iter__(self):
        pass

    def __len__(self):
        pass
