from dataclasses import dataclass, field
from enum import IntEnum
from pprint import pprint
from io import StringIO
from typing import IO, Dict
import random
import textwrap
import os

from mock_asm import ProgramGen, parse, VM, Inst, GotoOperands, BrCtrOperands


DEBUGGRAPH = int(os.environ.get("DEBUGGRAPH", 0))

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
    bbtargets: tuple[int, ...] = ()


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

    def render_control_variable_block(
        self, digraph: "Digraph", label: Label, block: BasicBlock
    ):
        if isinstance(label, ControlLabel):
            body = label.__class__.__name__ + ": " + str(label.index) + "\l"
            body += "\l".join(
                (f"{k} = {v}" for k, v in block.variable_assignment.items())
            )
        else:
            raise Exception("Unknown label type: " + label)
        digraph.node(str(label), shape="rect", label=body)

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
            super().render_basic_block(digraph, label, block)


def to_scfg(instlist: list[Inst]) -> BlockMap:
    labels = set([0, len(instlist)])
    for pc, inst in enumerate(instlist):
        if isinstance(inst.operands, GotoOperands):
            labels.add(inst.operands.jump_target)
            if pc + 1 < len(instlist):
                labels.add(pc + 1)
        elif isinstance(inst.operands, BrCtrOperands):
            labels.add(inst.operands.true_target)
            labels.add(inst.operands.false_target)
            if pc + 1 < len(instlist):
                labels.add(pc + 1)
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
        elif end < len(instlist):
            targets = [end]
        else:
            targets = []

        label = labelmap[begin]
        block = MockAsmBasicBlock(
            label=label,
            bbinstlist=bb,
            bboffset=begin,
            bbtargets=tuple(targets),
            _jump_targets=tuple(labelmap[tgt] for tgt in targets),
        )
        scfg.add_block(block)

    # remove dead code from reachabiliy of entry block
    reachable = set([labelmap[0]])
    stack = [labelmap[0]]
    while stack:
        blk: BasicBlock = scfg.graph[stack.pop()]
        for k in blk._jump_targets:
            if k not in reachable:
                stack.append(k)
                reachable.add(k)
    scfg.remove_blocks(set(scfg.graph.keys()) - reachable)

    # for name, bb in bbmap.items():
    #     if targets:
    #         scfg.add_connections(name, [edgemap[tgt] for tgt in targets])
    # scfg.check_graph()

    scfg.join_returns()
    if DEBUGGRAPH:
        MockAsmRenderer(scfg).view('jointed')
    restructure_loop(scfg)
    if DEBUGGRAPH:
        MockAsmRenderer(scfg).view('loop')
    restructure_branch(scfg)
    if DEBUGGRAPH:
        MockAsmRenderer(scfg).view('branch')
    return scfg

class Simulator:
    def __init__(self, scfg: BlockMap, buf: StringIO, max_step):
        self.vm = VM(buf)
        self.scfg = scfg
        self.region_stack = []
        self.ctrl_varmap = dict()
        self.max_step = max_step
        self.step = 0

    def run(self):
        scfg = self.scfg
        label = scfg.find_head()
        while True:
            action = self.run_block(self.scfg.graph[label])
            # If we need to return, break and do so
            if "return" in action:
                break  # break and return action
            elif "jumpto" in action:
                label = action["jumpto"]
                # Otherwise check if we stay in the region and break otherwise
                if label in self.scfg.graph:
                    continue  # stay in the region
                else:
                    break  # break and return action
            else:
                assert False, "unreachable" # in case of coding errors

    def run_block(self, block):
        print("run block", block.label)
        if isinstance(block, RegionBlock):
            return self.run_RegionBlock(block)
        elif isinstance(block, MockAsmBasicBlock):
            return self.run_MockAsmBasicBlock(block)
        elif isinstance(block.label, ControlLabel):
            print("    ", block)
            label = block.label
            handler = getattr(self, f"synth_{type(label).__name__}")
            out = handler(label, block)
            print("    ctrl_varmap dump:", self.ctrl_varmap)
            return out
        else:
            assert False

    def run_RegionBlock(self, block: RegionBlock):
        self.region_stack.append(block)

        label = block.subregion.find_head()
        while True:
            action = self.run_block(block.subregion.graph[label])
            # If we need to return, break and do so
            if "return" in action:
                break  # break and return action
            elif "jumpto" in action:
                label = action["jumpto"]
                # Otherwise check if we stay in the region and break otherwise
                if label in block.subregion.graph:
                    continue  # stay in the region
                else:
                    break  # break and return action
            else:
                assert False, "unreachable" # in case of coding errors

        self.region_stack.pop()
        return action

    def run_MockAsmBasicBlock(self, block: MockAsmBasicBlock):
        vm = self.vm
        pc = block.bboffset

        if self.step > self.max_step:
            raise AssertionError("step > max_step")

        for inst in block.bbinstlist:
            print("inst", pc, inst)
            pc = vm.eval_inst(pc, inst)
            self.step += 1
        if block.bbtargets:
            pos = block.bbtargets.index(pc)
            label = block._jump_targets[pos]
            return {"jumpto": label}
        else:
            return {"return": None}

   ### Synthetic Instructions ###
    def synth_SynthenticAssignment(self, control_label, block):
        self.ctrl_varmap.update(block.variable_assignment)
        [label] = block.jump_targets
        return {"jumpto": label}

    def _synth_branch(self, control_label, block):
        jump_target = block.branch_value_table[self.ctrl_varmap[block.variable]]
        return {"jumpto": jump_target}

    def synth_SyntheticExitingLatch(self, control_label, block):
        return self._synth_branch(control_label, block)

    def synth_SyntheticHead(self, control_label, block):
        return self._synth_branch(control_label, block)

    def synth_SyntheticExit(self, control_label, block):
        return self._synth_branch(control_label, block)

    def synth_SyntheticReturn(self, control_label, block):
        [label] = block.jump_targets
        return {"jumpto": label}

    def synth_SyntheticTail(self, control_label, block):
        [label] = block.jump_targets
        return {"jumpto": label}

    def synth_SyntheticBranch(self, control_label, block):
        [label] = block.jump_targets
        return {"jumpto": label}


def simulate_scfg(scfg: BlockMap):
    with StringIO() as buf:
        Simulator(scfg, buf, max_step=1000).run()
        return buf.getvalue()

def compare_simulated_scfg(asm):
    instlist = parse(asm)
    scfg = to_scfg(instlist)

    with StringIO() as buf:
        terminated = VM(buf).run(instlist, max_step=1000)
        assert terminated
        expect = buf.getvalue()
    print("EXPECT".center(80, '='))
    print(expect)

    got = simulate_scfg(scfg)
    assert got == expect

    return scfg

def ensure_contains_region(scfg: BlockMap, loop: int, branch: int):
    def recurse_find_regions(bbmap: BlockMap):
        for blk in bbmap.graph.values():
            if isinstance(blk, RegionBlock):
                yield blk
                yield from recurse_find_regions(blk.subregion)

    regions = list(recurse_find_regions(scfg))
    count_loop = 0
    count_branch = 0
    for reg in regions:
        if reg.kind == 'loop':
            count_loop += 1
        elif reg.kind == 'branch':
            count_branch += 1

    assert loop == count_loop
    assert branch == count_branch

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
    scfg = compare_simulated_scfg(asm)
    ensure_contains_region(scfg, loop=1, branch=0)


def test_mock_scfg_head_cycle():
    # Must manually enforce the the entry block only has no predecessor
    asm = textwrap.dedent("""
            print Start
            goto S
        label S
            print S
            goto A
        label A
            print A
            ctr 10
            brctr S B
        label B
            print B
    """)
    scfg = compare_simulated_scfg(asm)
    ensure_contains_region(scfg, loop=1, branch=0)

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
    scfg = compare_simulated_scfg(asm)
    ensure_contains_region(scfg, loop=0, branch=2)


def test_mock_scfg_double_exchange_loop():
    asm = textwrap.dedent("""
            print Start
            goto A
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
    scfg = compare_simulated_scfg(asm)
    # branch count may be more once branch restructuring is fixed
    ensure_contains_region(scfg, loop=1, branch=4)



def run_fuzzer(seed):
    rng = random.Random(seed)
    pg = ProgramGen(rng)
    asm = pg.generate_program()

    print(seed)
    instlist = parse(asm)
    with StringIO() as buf:
        terminated = VM(buf).run(instlist, max_step=1000)
        got = buf.getvalue().split()
        if terminated:
            print(f"seed={seed}".center(80, "="))
            print(asm)
            print(got)

            compare_simulated_scfg(asm)
            return True

failing_case = {
    0,
    7,
    9,
    36,
}

def test_mock_scfg_fuzzer():
    ct_term = 0
    total = 10000
    for i in range(total):
        if i in failing_case:
            continue
        run_fuzzer(i)
    print("terminated", ct_term, "total", total)


def test_mock_scfg_fuzzer_case0():
    run_fuzzer(seed=0)

def test_mock_scfg_fuzzer_case7():
    run_fuzzer(seed=7)

def test_mock_scfg_fuzzer_case9():
    # invalid control variable causes infinite loop
    run_fuzzer(seed=9)

def test_mock_scfg_fuzzer_case36():
    run_fuzzer(seed=36)
