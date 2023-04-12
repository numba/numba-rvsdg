from dataclasses import dataclass, field
from enum import IntEnum
from pprint import pprint
from io import StringIO
from typing import IO, Dict
import random
import textwrap

from mock_asm import ProgramGen, parse, VM, Inst, GotoOperands, BrCtrOperands


def test_mock_asm():
    asm = textwrap.dedent("""
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr A B
        label B
            print B
    """)

    instlist = parse(asm)
    assert instlist[0].operands.text == "Start"
    assert instlist[1].operands.jump_target == 2
    assert instlist[2].operands.text == "A"
    assert instlist[3].operands.counter == 10
    assert instlist[4].operands.true_target == 2
    assert instlist[4].operands.false_target == 5
    assert instlist[5].operands.text == "B"

    with StringIO() as buf:
        VM(buf).run(instlist)
        got = buf.getvalue().split()

    expected = ["Start", *(["A"] * 10), "B"]
    assert got == expected


def test_double_exchange_loop():
    asm = textwrap.dedent("""
            print Start
       label A
            print A
            ctr 4
            brctr B Exit
        label B
            print B
            ctr 5
            brctr A Exit
        label Exit
            print Exit
    """)
    instlist = parse(asm)
    with StringIO() as buf:
        VM(buf).run(instlist)
        got = buf.getvalue().split()

    expected = ["Start", *(["A", "B"] * 3), "A", "Exit"]
    assert got == expected


def test_program_gen():
    rng = random.Random(123)
    pg = ProgramGen(rng)
    ct_term = 0
    total = 10000
    for i in range(total):
        print(str(i).center(80, "="))
        asm = pg.generate_program()


        instlist = parse(asm)
        with StringIO() as buf:
            terminated = VM(buf).run(instlist,
                                     max_step=1000)
            got = buf.getvalue().split()
            if terminated:
                print(asm)
                print(got)
                ct_term += 1
    print("terminated", ct_term, "total", total)

from numba_rvsdg.core.datastructures.block_map import BlockMap
from numba_rvsdg.core.datastructures.labels import Label
from numba_rvsdg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
)
from numba_rvsdg.core.transformations import (
    restructure_loop,
    restructure_branch,
)


@dataclass(frozen=True, order=True)
class MockAsmLabel(Label):
    pass

@dataclass(frozen=True)
class MockAsmBasicBlock(BasicBlock):
    bbinstlist: list[Inst] = field(default_factory=list)
    bboffset: int = 0



from numba_rvsdg.core.datastructures.labels import (
    ControlLabel, ControlLabelGenerator
)
from numba_rvsdg.core.datastructures.basic_block import (
    ControlVariableBlock,
    BranchBlock,
)
# NOTE: modified Renderer to be more general
class Renderer(object):
    def __init__(self, bbmap: BlockMap):
        from graphviz import Digraph

        self.g = Digraph()

        self.rendered_blocks = set()

        # render nodes
        for label, block in bbmap.graph.items():
            self.render_block(self.g, label, block)
        self.render_edges(bbmap.graph)

    def render_basic_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        body = str(label)
        digraph.node(str(label), shape="rect", label=body)

    def render_region_block(
        self, digraph: "Digraph", label: Label, regionblock: RegionBlock
    ):
        # render subgraph
        graph = regionblock.get_full_graph()
        with digraph.subgraph(name=f"cluster_{label}") as subg:
            color = "blue"
            if regionblock.kind == "branch":
                color = "green"
            if regionblock.kind == "tail":
                color = "purple"
            if regionblock.kind == "head":
                color = "red"
            subg.attr(color=color, label=regionblock.kind)
            for label, block in graph.items():
                self.render_block(subg, label, block)
        # render edges within this region
        self.render_edges(graph)

    def render_branching_block(
        self, digraph: "Digraph", label: Label, block: BasicBlock
    ):
        if isinstance(label, ControlLabel):

            def find_index(v):
                if hasattr(v, "offset"):
                    return v.offset
                if hasattr(v, "index"):
                    return v.index

            body = label.__class__.__name__ + ": " + str(label.index) + "\l"
            body += f"variable: {block.variable}\l"
            body += "\l".join(
                (f"{k}=>{find_index(v)}" for k, v in block.branch_value_table.items())
            )
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

    def render_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        # elif type(block) == PythonBytecodeBlock:
        #     self.render_basic_block(digraph, label, block)
        if type(block) == ControlVariableBlock:
            self.render_control_variable_block(digraph, label, block)
        elif type(block) == BranchBlock:
            self.render_branching_block(digraph, label, block)
        elif type(block) == RegionBlock:
            self.render_region_block(digraph, label, block)
        elif isinstance(block, BasicBlock):
            self.render_basic_block(digraph, label, block)
        else:
            raise Exception("unreachable")


    def render_edges(self, blocks: Dict[Label, BasicBlock]):
        for label, block in blocks.items():
            for dst in block.jump_targets:
                if dst in blocks:
                    if type(block) in (
                        # PythonBytecodeBlock,
                        MockAsmBasicBlock,
                        BasicBlock,
                        ControlVariableBlock,
                        BranchBlock,
                    ):
                        self.g.edge(str(label), str(dst))
                    elif type(block) == RegionBlock:
                        if block.exit is not None:
                            self.g.edge(str(block.exit), str(dst))
                        else:
                            self.g.edge(str(label), str(dst))
                    else:
                        raise Exception("unreachable")
            for dst in block.backedges:
                # assert dst in blocks
                self.g.edge(
                    str(label), str(dst), style="dashed", color="grey", constraint="0"
                )
    def view(self, *args):
        self.g.view(*args)



class MockAsmRenderer(Renderer):

    def render_basic_block(self, digraph: "Digraph", label: Label, block: BasicBlock):
        block_name = str(label)

        if isinstance(block, MockAsmBasicBlock):
            end = r"\l"
            lines = [
                f"offset: {block.bboffset} | {block_name} ",
                *[str(inst) for inst in block.bbinstlist],
            ]
            body = ''.join([ln + end for ln in lines])
            digraph.node(str(block_name), shape="rect", label=body)
        else:
            super().render_basic_block(digraph, block_name)


def to_scfg(instlist: list[Inst]) -> BlockMap:
    labels = set([0, len(instlist)])
    for inst in instlist:
        if isinstance(inst.operands, GotoOperands):
            labels.add(inst.operands.jump_target)
        elif isinstance(inst.operands, BrCtrOperands):
            labels.add(inst.operands.true_target)
            labels.add(inst.operands.false_target)

    block_map_graph = {}
    clg = ControlLabelGenerator()
    scfg = BlockMap(block_map_graph, clg)
    bb_offsets =sorted(labels)
    labelmap = {}


    for begin, end in zip(bb_offsets, bb_offsets[1:]):
        labelmap[begin] = label = MockAsmLabel(str(clg.new_index()))

    for begin, end in zip(bb_offsets, bb_offsets[1:]):
        bb = instlist[begin:end]
        inst = bb[-1]  # terminator
        if isinstance(inst.operands, GotoOperands):
            targets = [inst.operands.jump_target]
        elif isinstance(inst.operands, BrCtrOperands):
            targets = [inst.operands.true_target,
                       inst.operands.false_target]
        else:
            targets = []

        label = labelmap[begin]
        block = MockAsmBasicBlock(
            label=label,
            bbinstlist=bb,
            bboffset=begin,
            _jump_targets=tuple(labelmap[tgt] for tgt in targets),
        )
        scfg.add_block(block)

    # for name, bb in bbmap.items():
    #     if targets:
    #         scfg.add_connections(name, [edgemap[tgt] for tgt in targets])
    # scfg.check_graph()

    scfg.join_returns()
    restructure_loop(scfg)
    restructure_branch(scfg)
    MockAsmRenderer(scfg).view()
    return scfg


def test_mock_scfg_loop():
    asm = textwrap.dedent("""
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr A B
        label B
            print B
    """)

    instlist = parse(asm)
    scfg = to_scfg(instlist)


def test_mock_scfg_basic():
    asm = textwrap.dedent("""
        label S
            print Start
            goto A
        label A
            print A
            ctr 10
            brctr S B
        label B
            print B
    """)

    instlist = parse(asm)
    scfg = to_scfg(instlist)

def test_mock_scfg_diamond():
    asm = textwrap.dedent("""
            print Start
            ctr 1
            brctr A B
        label A
            print A
            goto C
        label B
            print B
            goto C
        label C
            print C
    """)

    instlist = parse(asm)
    scfg = to_scfg(instlist)