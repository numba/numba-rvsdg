# Figure 3 of the paper
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.rendering.rendering import ByteFlowRenderer

# import logging
# logging.basicConfig(level=logging.DEBUG)


def make_flow():
    # flowinfo = FlowInfo()
    import dis

    # fake bytecode just good enough for FlowInfo
    bc = [
        dis.Instruction("OP", 1, None, None, "", 0, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 14, "", 2, None, False),
        # label 4
        dis.Instruction("OP", 1, None, None, "", 4, None, False),
        dis.Instruction("POP_JUMP_IF_TRUE", 2, None, 12, "", 6, None, False),
        dis.Instruction("OP", 1, None, None, "", 8, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 18, "", 10, None, False),
        # label 12
        dis.Instruction("OP", 1, None, None, "", 12, None, False),
        dis.Instruction("OP", 2, None, 4, "", 14, None, False),
        dis.Instruction("JUMP_ABSOLUTE", 2, None, 18, "", 16, None, False),
        # label 18
        dis.Instruction("RETURN_VALUE", 1, None, None, "", 18, None, False),
    ]
    flow = FlowInfo.from_bytecode(bc)
    bbmap = flow.build_basicblocks()
    return ByteFlow(bc=bc, bbmap=bbmap)


flow = make_flow()
# ByteFlowRenderer().render_byteflow(flow).view("before")
#
# flow = flow.restructure()
# ByteFlowRenderer().render_byteflow(flow).view("after")
#
#
# flow = ByteFlow.from_bytecode(foo)
ByteFlowRenderer().render_byteflow(flow).view("before")

cflow = flow._join_returns()
ByteFlowRenderer().render_byteflow(cflow).view("closed")

lflow = cflow._restructure_loop()
ByteFlowRenderer().render_byteflow(lflow).view("loop restructured")

bflow = lflow._restructure_branch()
ByteFlowRenderer().render_byteflow(bflow).view("branch restructured")
