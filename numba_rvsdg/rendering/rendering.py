import logging
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    PythonBytecodeBlock,
    SyntheticAssignment,
    SyntheticExitingLatch,
    SyntheticExitBranch,
    SyntheticBranch,
    SyntheticBlock,
    SyntheticHead,
    SyntheticExit,
)
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
import dis
from typing import Dict


class ByteFlowRenderer(object):
    def __init__(self):
        from graphviz import Digraph

        self.g = Digraph()

    def render_region_block(
        self, digraph: "Digraph", name: str, regionblock: RegionBlock
    ):
        # render subgraph
        graph = regionblock.get_full_graph()
        with digraph.subgraph(name=f"cluster_{name}") as subg:
            color = "blue"
            if regionblock.kind == "branch":
                color = "green"
            if regionblock.kind == "tail":
                color = "purple"
            if regionblock.kind == "head":
                color = "red"
            subg.attr(color=color, label=regionblock.kind)
            for name, block in graph.items():
                self.render_block(subg, name, block)
        # render edges within this region
        self.render_edges(graph)

    def render_basic_block(self, digraph: "Digraph", name: str, block: BasicBlock):
        if name.startswith('python_bytecode'):
            instlist = block.get_instructions(self.bcmap)
            body = name + "\l"
            body += "\l".join(
                [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
            )
        else:
            body = name + "\l"

        digraph.node(str(name), shape="rect", label=body)

    def render_control_variable_block(
        self, digraph: "Digraph", name: str, block: BasicBlock
    ):
        if isinstance(name, str):
            body = name + "\l"
            body += "\l".join(
                (f"{k} = {v}" for k, v in block.variable_assignment.items())
            )
        else:
            raise Exception("Unknown name type: " + name)
        digraph.node(str(name), shape="rect", label=body)

    def render_branching_block(
        self, digraph: "Digraph", name: str, block: BasicBlock
    ):
        if isinstance(name, str):
            body = name + "\l"
            body += f"variable: {block.variable}\l"
            body += "\l".join(
                (f"{k}=>{v}" for k, v in block.branch_value_table.items())
            )
        else:
            raise Exception("Unknown name type: " + name)
        digraph.node(str(name), shape="rect", label=body)

    def render_block(self, digraph: "Digraph", name: str, block: BasicBlock):
        if type(block) == BasicBlock:
            self.render_basic_block(digraph, name, block)
        elif type(block) == PythonBytecodeBlock:
            self.render_basic_block(digraph, name, block)
        elif type(block) == SyntheticAssignment:
            self.render_control_variable_block(digraph, name, block)
        elif isinstance(block, SyntheticBranch):
            self.render_branching_block(digraph, name, block)
        elif isinstance(block, SyntheticBlock):
            self.render_basic_block(digraph, name, block)
        elif type(block) == RegionBlock:
            self.render_region_block(digraph, name, block)
        else:
            raise Exception("unreachable")

    def render_edges(self, blocks: Dict[str, BasicBlock]):
        for name, block in blocks.items():
            for dst in block.jump_targets:
                if dst in blocks:
                    if type(block) in (
                        PythonBytecodeBlock,
                        BasicBlock,
                        SyntheticBlock,
                        SyntheticAssignment,
                        SyntheticExitingLatch,
                        SyntheticExitBranch,
                        SyntheticHead,
                        SyntheticExit,
                        SyntheticBranch,
                    ):
                        self.g.edge(str(name), str(dst))
                    elif type(block) == RegionBlock:
                        if block.exiting is not None:
                            self.g.edge(str(block.exiting), str(dst))
                        else:
                            self.g.edge(str(name), str(dst))
                    else:
                        raise Exception("unreachable " + str(block))
            for dst in block.backedges:
                # assert dst in blocks
                self.g.edge(
                    str(name), str(dst), style="dashed", color="grey", constraint="0"
                )

    def render_byteflow(self, byteflow: ByteFlow):
        self.bcmap_from_bytecode(byteflow.bc)

        # render nodes
        for name, block in byteflow.scfg.graph.items():
            self.render_block(self.g, name, block)
        self.render_edges(byteflow.scfg.graph)
        return self.g

    def render_scfg(self, scfg):
        # render nodes
        for name, block in scfg.graph.items():
            self.render_block(self.g, name, block)
        self.render_edges(scfg.graph)
        return self.g

    def bcmap_from_bytecode(self, bc: dis.Bytecode):
        self.bcmap: Dict[int, dis.Instruction] = SCFG.bcmap_from_bytecode(bc)


logging.basicConfig(level=logging.DEBUG)


def render_func(func):
    render_flow(ByteFlow.from_bytecode(func))


def render_flow(flow):
    ByteFlowRenderer().render_byteflow(flow).view("before")

    cflow = flow._join_returns()
    ByteFlowRenderer().render_byteflow(cflow).view("closed")

    lflow = cflow._restructure_loop()
    ByteFlowRenderer().render_byteflow(lflow).view("loop restructured")

    bflow = lflow._restructure_branch()
    ByteFlowRenderer().render_byteflow(bflow).view("branch restructured")


def render_scfg(scfg):
    ByteFlowRenderer().render_scfg(scfg).view("scfg")
