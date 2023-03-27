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

        self.render_blocks()
        self.render_edges()
        self.render_regions()

    def render_regions(self):
        # render subgraph
        # graph = regionblock.get_full_graph()
        # with digraph.subgraph(name=f"cluster_{label}") as subg:
        #     color = "blue"
        #     if regionblock.kind == "branch":
        #         color = "green"
        #     if regionblock.kind == "tail":
        #         color = "purple"
        #     if regionblock.kind == "head":
        #         color = "red"
        #     subg.attr(color=color, label=regionblock.kind)
        #     for label, block in graph.items():
        #         self.render_block(subg, label, block)
        # # render edges within this region
        # self.render_edges(graph)
        pass

    def render_basic_block(self, block_name: BlockName):
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
        self.g.node(str(block_name), shape="rect", label=body)

    def render_control_variable_block(self, block_name: BlockName):
        block = self.scfg[block_name]

        if isinstance(block.label, ControlLabel):
            body = str(block_name) + "\l"
            body += "\l".join(
                (f"{k} = {v}" for k, v in block.variable_assignment.items())
            )
        else:
            raise Exception("Unknown label type: " + block.label)
        self.g.node(str(block_name), shape="rect", label=body)

    def render_branching_block(self, block_name: BlockName):
        block = self.scfg[block_name]

        if isinstance(block.label, ControlLabel):

            def find_index(v):
                if hasattr(v, "offset"):
                    return v.offset
                if hasattr(v, "index"):
                    return v.index

            body = str(block_name) + "\l"
            body += f"variable: {block.variable}\l"
            body += "\l".join(
                (f"{k}=>{find_index(v)}" for k, v in block.branch_value_table.items())
            )
        else:
            raise Exception("Unknown label type: " + block.label)
        self.g.node(str(block_name), shape="rect", label=body)

    def render_blocks(self):
        for block_name, block in self.scfg.blocks.items():
            if type(block) == BasicBlock:
                self.render_basic_block(block_name)
            elif type(block) == PythonBytecodeBlock:
                self.render_basic_block(block_name)
            elif type(block) == ControlVariableBlock:
                self.render_control_variable_block(block_name)
            elif type(block) == BranchBlock:
                self.render_branching_block(block_name)
            else:
                raise Exception("unreachable")

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
