import logging
from abc import abstractmethod
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    PythonBytecodeBlock,
    SyntheticAssignment,
    SyntheticBranch,
    SyntheticBlock,
)
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
import dis
from typing import Dict, Optional
from graphviz import Digraph


class BaseRenderer:
    """Base Renderer class.

    This is the base class for all types of graph renderers. It defines two
    methods `render_block` and `render_edges` that define how the blocks and
    edges of the graph are rendered respectively.
    """

    g: "Digraph"

    @abstractmethod
    def render_basic_block(
        self, digraph: "Digraph", name: str, block: BasicBlock
    ) -> None:
        """ """

    @abstractmethod
    def render_control_variable_block(
        self, digraph: "Digraph", name: str, block: SyntheticAssignment
    ) -> None:
        """ """

    @abstractmethod
    def render_branching_block(
        self, digraph: "Digraph", name: str, block: SyntheticBranch
    ) -> None:
        """ """

    @abstractmethod
    def render_region_block(
        self, digraph: "Digraph", name: str, regionblock: RegionBlock
    ) -> None:
        """ """

    def render_block(
        self, digraph: "Digraph", name: str, block: BasicBlock
    ) -> None:
        """Function that defines how the BasicBlocks in a graph should be
        rendered.

        Parameters
        ----------
        digraph: Digraph
            The graphviz Digraph object that represents the graph/subgraph upon
            which the current blocks are to be rendered.
        name: str
            Name of the block to be rendered.
        block: BasicBlock
            The BasicBlock to be rendered.

        """
        if type(block) == BasicBlock:
            self.render_basic_block(digraph, name, block)
        if type(block) == PythonBytecodeBlock:
            self.render_basic_block(digraph, name, block)
        elif type(block) == SyntheticAssignment:
            self.render_control_variable_block(digraph, name, block)
        elif isinstance(block, SyntheticBranch):
            self.render_branching_block(digraph, name, block)
        elif type(block) == RegionBlock:
            self.render_region_block(digraph, name, block)
        elif isinstance(block, SyntheticBlock):
            self.render_basic_block(digraph, name, block)
        else:
            raise Exception("unreachable")

    def render_edges(self, scfg: SCFG) -> None:
        """Function that renders the edges in an SCFG.

        Parameters
        ----------
        scfg: SCFG
            The graph whose edges are to be rendered.

        """
        blocks = dict(scfg)

        def find_base_header(block: BasicBlock) -> BasicBlock:
            if isinstance(block, RegionBlock):
                block = blocks[block.header]  # type: ignore
                block = find_base_header(block)
            return block

        for _, src_block in blocks.items():
            if isinstance(src_block, RegionBlock):
                continue
            src_block = find_base_header(src_block)
            for dst_name in src_block.jump_targets:
                dst_name = find_base_header(blocks[dst_name]).name
                if dst_name in blocks.keys():
                    self.g.edge(str(src_block.name), str(dst_name))
                else:
                    raise Exception("unreachable " + str(src_block))
            for dst_name in src_block.backedges:
                dst_name = find_base_header(blocks[dst_name]).name
                if dst_name in blocks.keys():
                    self.g.edge(
                        str(src_block.name),
                        str(dst_name),
                        style="dashed",
                        color="grey",
                        constraint="0",
                    )
                else:
                    raise Exception("unreachable " + str(src_block))


class ByteFlowRenderer(BaseRenderer):
    """The `ByteFlowRenderer` class is used to render the visual
    representation of a `ByteFlow` object.

    Attributes
    ----------
    g: Digraph
        The graphviz Digraph object that represents the entire graph upon
        which the current ByteFlow is to be rendered.
    bcmap: Dict[int, dis.Instruction]
        Mapping of bytecode offset to instruction.

    """

    def __init__(self) -> None:
        from graphviz import Digraph

        self.g = Digraph()

    def render_region_block(
        self, digraph: "Digraph", name: str, regionblock: RegionBlock
    ) -> None:
        # render subgraph
        with digraph.subgraph(name=f"cluster_{name}") as subg:
            color = "blue"
            if regionblock.kind == "branch":
                color = "green"
            if regionblock.kind == "tail":
                color = "purple"
            if regionblock.kind == "head":
                color = "red"
            subg.attr(color=color, label=regionblock.name)
            assert regionblock.subregion is not None
            for name, block in regionblock.subregion.graph.items():
                self.render_block(subg, name, block)

    def render_basic_block(
        self, digraph: "Digraph", name: str, block: BasicBlock
    ) -> None:
        if name.startswith("python_bytecode") and isinstance(
            block, PythonBytecodeBlock
        ):
            instlist = block.get_instructions(self.bcmap)
            body = name + r"\l"
            body += r"\l".join(
                [f"{inst.offset:3}: {inst.opname}" for inst in instlist] + [""]
            )
        else:
            body = name + r"\l"

        digraph.node(str(name), shape="rect", label=body)

    def render_control_variable_block(
        self, digraph: "Digraph", name: str, block: SyntheticAssignment
    ) -> None:
        if isinstance(name, str):
            body = name + r"\l"
            body += r"\l".join(
                (f"{k} = {v}" for k, v in block.variable_assignment.items())
            )
        else:
            raise Exception("Unknown name type: " + name)
        digraph.node(str(name), shape="rect", label=body)

    def render_branching_block(
        self, digraph: "Digraph", name: str, block: SyntheticBranch
    ) -> None:
        if isinstance(name, str):
            body = name + r"\l"
            body += rf"variable: {block.variable}\l"
            body += r"\l".join(
                (f"{k}=>{v}" for k, v in block.branch_value_table.items())
            )
        else:
            raise Exception("Unknown name type: " + name)
        digraph.node(str(name), shape="rect", label=body)

    def render_byteflow(self, byteflow: ByteFlow) -> "Digraph":
        """Renders the provided `ByteFlow` object."""
        self.bcmap_from_bytecode(byteflow.bc)
        # render nodes
        for name, block in byteflow.scfg.graph.items():
            self.render_block(self.g, name, block)
        self.render_edges(byteflow.scfg)
        return self.g

    def bcmap_from_bytecode(self, bc: dis.Bytecode) -> None:
        self.bcmap: Dict[int, dis.Instruction] = SCFG.bcmap_from_bytecode(bc)


class SCFGRenderer(BaseRenderer):
    """The `SCFGRenderer` class is used to render the visual
    representation of a `SCFG` object.

    Attributes
    ----------
    g: Digraph
        The graphviz Digraph object that represents the entire graph upon
        which the current SCFG is to be rendered.

    """

    def __init__(self, scfg: SCFG):
        from graphviz import Digraph

        self.g = Digraph()
        # render nodes
        for name, block in scfg.graph.items():
            self.render_block(self.g, name, block)
        self.render_edges(scfg)

    def render_region_block(
        self, digraph: "Digraph", name: str, regionblock: RegionBlock
    ) -> None:
        # render subgraph
        with digraph.subgraph(name=f"cluster_{name}") as subg:
            color = "blue"
            if regionblock.kind == "branch":
                color = "green"
            if regionblock.kind == "tail":
                color = "purple"
            if regionblock.kind == "head":
                color = "red"
            label = (
                regionblock.name
                + "\njump targets: "
                + str(regionblock.jump_targets)
                + "\nback edges: "
                + str(regionblock.backedges)
            )

            subg.attr(color=color, label=label)
            assert regionblock.subregion is not None
            for name, block in regionblock.subregion.graph.items():
                self.render_block(subg, name, block)

    def render_basic_block(
        self, digraph: "Digraph", name: str, block: BasicBlock
    ) -> None:
        body = (
            name
            + r"\l"
            + "\njump targets: "
            + str(block.jump_targets)
            + "\nback edges: "
            + str(block.backedges)
        )

        digraph.node(str(name), shape="rect", label=body)

    def render_control_variable_block(
        self, digraph: "Digraph", name: str, block: SyntheticAssignment
    ) -> None:
        if isinstance(name, str):
            body = name + r"\l"
            body += r"\l".join(
                (f"{k} = {v}" for k, v in block.variable_assignment.items())
            )
            body += (
                "\njump targets: "
                + str(block.jump_targets)
                + "\nback edges: "
                + str(block.backedges)
            )

        else:
            raise Exception("Unknown name type: " + name)
        digraph.node(str(name), shape="rect", label=body)

    def render_branching_block(
        self, digraph: "Digraph", name: str, block: SyntheticBranch
    ) -> None:
        if isinstance(name, str):
            body = name + r"\l"
            body += rf"variable: {block.variable}\l"
            body += r"\l".join(
                (f"{k}=>{v}" for k, v in block.branch_value_table.items())
            )
            body += (
                "\njump targets: "
                + str(block.jump_targets)
                + "\nback edges: "
                + str(block.backedges)
            )

        else:
            raise Exception("Unknown name type: " + name)
        digraph.node(str(name), shape="rect", label=body)

    def view(self, name: Optional[str] = None) -> None:
        """Method used to view the current SCFG as an external graphviz
        generated PDF file.

        Parameters
        ----------
        name: str
            Name to be given to the external graphviz generated PDF file.
        """
        self.g.view(name)


logging.basicConfig(level=logging.DEBUG)


def render_func(func) -> None:  # type: ignore
    """The `render_func`` function takes a `func` parameter as the Python
    function to be transformed and rendered and renders the byte flow
    representation of the bytecode of the function.

    Parameters
    ----------
    func: Python function
        The Python function for which bytecode is to be rendered.
    """
    render_flow(ByteFlow.from_bytecode(func))


def render_flow(flow: ByteFlow) -> None:
    """Renders multiple ByteFlow representations across various SCFG
    transformations.

    The `render_flow`` function takes a `flow` parameter as the `ByteFlow`
    to be transformed and rendered and performs the following operations:

        - Renders the pure `ByteFlow` representation of the function using
          `ByteFlowRenderer` and displays it as a document named "before".

        - Joins the return blocks in the `ByteFlow` object graph and renders
          the graph, displaying it as a document named "closed".

        - Restructures the loops recursively in the `ByteFlow` object graph
          and renders the graph, displaying it as named "loop restructured".

        - Restructures the branch recursively in the `ByteFlow` object graph
          and renders the graph, displaying it as named "branch restructured".

    Parameters
    ----------
    flow: ByteFlow
        The ByteFlow object to be trnasformed and rendered.
    """
    ByteFlowRenderer().render_byteflow(flow).view("before")

    flow._join_returns()
    ByteFlowRenderer().render_byteflow(flow).view("closed")

    flow._restructure_loop()
    ByteFlowRenderer().render_byteflow(flow).view("loop restructured")

    flow._restructure_branch()
    ByteFlowRenderer().render_byteflow(flow).view("branch restructured")


def render_scfg(scfg: SCFG) -> None:
    """The `render_scfg` function takes a `scfg` parameter as the SCFG
    object to be transformed and rendered and renders the graphviz
    representation of the SCFG.

    Parameters
    ----------
    scfg: SCFG
        The structured control flow graph (SCFG) to be rendered.
    """
    # is this function used??
    ByteFlowRenderer().render_scfg(scfg).view("scfg")  # type: ignore
