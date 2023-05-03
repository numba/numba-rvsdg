# Figure 3 of the paper
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.rendering.rendering import render_flow

# import logging
# logging.basicConfig(level=logging.DEBUG)


def make_flow():
    # flowinfo = FlowInfo()
    import dis

    # fake bytecode just good enough for FlowInfo
    bc = [
        dis.Instruction("OP", 1, None, None, "", 0, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 12, "", 2, None, False),
        # label 4
        dis.Instruction("OP", 1, None, None, "", 4, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 12, "", 6, None, False),
        dis.Instruction("OP", 1, None, None, "", 8, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 20, "", 10, None, False),
        # label 12
        dis.Instruction("OP", 1, None, None, "", 12, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 4, "", 14, None, False),
        dis.Instruction("OP", 1, None, None, "", 16, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 20, "", 18, None, False),
        # label 20
        dis.Instruction("RETURN_VALUE", 1, None, None, "", 20, None, False),
    ]
    flow = FlowInfo.from_bytecode(bc)
    scfg = flow.build_basicblocks()
    return ByteFlow(bc=bc, scfg=scfg)


def test_fig3():
    f = make_flow()
    f.restructure()


if __name__ == "__main__":
    render_flow(make_flow())
