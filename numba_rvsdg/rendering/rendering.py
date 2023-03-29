import logging
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    PythonBytecodeBlock,
    ControlVariableBlock,
    BranchBlock,
)
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.labels import (
    Label,
    PythonBytecodeLabel,
    ControlLabel,
    BlockName,
)
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
import dis
from typing import Dict


class ByteFlowRenderer(object):
    def __init__(self, byte_flow: ByteFlow):
        from graphviz import Digraph

        self.g = Digraph()
        self.byte_flow = byte_flow
        self.scfg = byte_flow.scfg
        self.bcmap_from_bytecode(byte_flow.bc)

        self.rendered_blocks = set()
        self.rendered_regions = set()
        self.render_region(self.g, None)
        self.render_edges()

    def render_basic_block(self, graph, block_name: BlockName):
        block = self.scfg[block_name]

        if isinstance(block.label, PythonBytecodeLabel):
            instlist = block.get_instructions(self.bcmap)
            body = str(block_name) + "\l"
            body += "\l".join(
                [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
            )
        elif isinstance(block, ControlLabel):
            body = str(block_name)
        else:
            raise Exception("Unknown label type: " + block.label)
        graph.node(str(block_name), shape="rect", label=body)

    def render_control_variable_block(self, graph, block_name: BlockName):
        block = self.scfg[block_name]

        if isinstance(block.label, ControlLabel):
            body = str(block_name) + "\l"
            # body += "\l".join(
            #     (f"{k} = {v}" for k, v in block.variable_assignment.items())
            # )
        else:
            raise Exception("Unknown label type: " + block.label)
        graph.node(str(block_name), shape="rect", label=body)

    def render_branching_block(self, graph, block_name: BlockName):
        block = self.scfg[block_name]

        if isinstance(block.label, ControlLabel):

            def find_index(v):
                if hasattr(v, "offset"):
                    return v.offset
                if hasattr(v, "index"):
                    return v.index

            body = str(block_name) + "\l"
            # body += f"variable: {block.variable}\l"
            # body += "\l".join(
            #     (f" {k} => {find_index(v)}" for k, v in block.branch_value_table.items())
            # )
        else:
            raise Exception("Unknown label type: " + block.label)
        graph.node(str(block_name), shape="rect", label=body)

    def render_region(self, graph, region_name):
        # If region name is none, we're in the 'root' region
        # that is the graph itself.
        if region_name is None:
            region_headers = self.scfg.region_headers
            all_blocks_iterator = self.scfg
        else:
            region = self.scfg.regions[region_name]
            region_headers = region.sub_region_headers
            all_blocks_iterator = self.scfg.iterate_region(region_name)

        # If subregions exist within this region we render them first
        if region_headers:
            for _, regions in region_headers.items():
                for _region_name in regions:
                    _region = self.scfg.regions[_region_name]
                    with graph.subgraph(name=f"cluster_{_region_name}") as subg:
                        color = "blue"
                        if _region.kind == "branch":
                            color = "green"
                        if _region.kind == "tail":
                            color = "purple"
                        if _region.kind == "head":
                            color = "red"
                        subg.attr(color=color, label=_region.kind)
                        self.render_region(subg, _region_name)

        # If there are no further subregions then we render the blocks
        for block_name in all_blocks_iterator:
            self.render_block(graph, block_name, self.scfg[block_name])

    def render_block(self, graph, block_name, block):
        if block_name in self.rendered_blocks:
            return

        if block_name in self.scfg.region_headers.keys():
            for region in self.scfg.region_headers[block_name]:
                if region not in self.rendered_regions:
                    self.rendered_regions.add(region)
                    self.render_region(graph, region)

        if type(block) == BasicBlock:
            self.render_basic_block(graph, block_name)
        elif type(block) == PythonBytecodeBlock:
            self.render_basic_block(graph, block_name)
        elif type(block) == ControlVariableBlock:
            self.render_control_variable_block(graph, block_name)
        elif type(block) == BranchBlock:
            self.render_branching_block(graph, block_name)
        else:
            raise Exception("unreachable")
        self.rendered_blocks.add(block_name)

    def render_edges(self):

        for block_name, out_edges in self.scfg.out_edges.items():
            for out_edge in out_edges:
                self.g.edge(str(block_name), str(out_edge))

        for block_name, back_edges in self.scfg.back_edges.items():
            for back_edge in back_edges:
                self.g.edge(
                    str(block_name),
                    str(back_edge),
                    style="dashed",
                    color="grey",
                    constraint="0",
                )

    def bcmap_from_bytecode(self, bc: dis.Bytecode):
        self.bcmap: Dict[int, dis.Instruction] = ByteFlow.bcmap_from_bytecode(bc)

    def view(self, *args):
        self.g.view(*args)


logging.basicConfig(level=logging.DEBUG)


def render_func(func):
    flow = ByteFlow.from_bytecode(func)
    ByteFlowRenderer(flow).view("before")

    flow._join_returns()
    ByteFlowRenderer(flow).view("closed")

    flow._restructure_loop()
    ByteFlowRenderer(flow).view("loop restructured")

    flow._restructure_branch()
    ByteFlowRenderer(flow).view("branch restructured")
