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
    RegionName
)
from numba_rvsdg.core.datastructures.region import MetaRegion, LoopRegion
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
            region_name = self.scfg.meta_region
            region = self.scfg.regions[region_name]
        else:
            region = self.scfg.regions[region_name]

        with graph.subgraph(name=f"cluster_{region_name}") as subg:
            if isinstance(region, LoopRegion):
                color = "blue"
            else:
                color = "black"
            subg.attr(color=color, label=str(region.label))

            for sub_region in self.scfg.region_tree[region_name]:
                self.render_region(subg, sub_region)

            # If there are no further subregions then we render the blocks
            for block_name in self.scfg.regional_components[region_name]:
                self.render_block(subg, block_name)

    def render_block(self, graph, block_name):
        if block_name in self.rendered_blocks:
            return

        block = self.scfg[block_name]
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
                if isinstance(out_edge, RegionName):
                    out_edge = self.scfg.regions[out_edge].header
                if (block_name, out_edge) in self.scfg.back_edges:
                    self.g.edge(
                        str(block_name),
                        str(out_edge),
                        style="dashed",
                        color="grey",
                        constraint="0",
                    )
                else:
                    self.g.edge(str(block_name), str(out_edge))

    def bcmap_from_bytecode(self, bc: dis.Bytecode):
        self.bcmap: Dict[int, dis.Instruction] = ByteFlow.bcmap_from_bytecode(bc)

    def view(self, *args):
        self.g.view(*args)


logging.basicConfig(level=logging.DEBUG)


def render_func(func):
    flow = ByteFlow.from_bytecode(func)
    render_flow(flow)


def render_flow(byte_flow):
    ByteFlowRenderer(byte_flow).view("before")

    byte_flow._join_returns()
    ByteFlowRenderer(byte_flow).view("closed")

    byte_flow._restructure_loop()
    ByteFlowRenderer(byte_flow).view("loop restructured")

    byte_flow._restructure_branch()
    ByteFlowRenderer(byte_flow).view("branch restructured")
