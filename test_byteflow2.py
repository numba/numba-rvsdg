import dis

import unittest
from byteflow2 import FlowInfo, BCLabel, Block, BlockMap


def fun():
    x = 1
    return x


bytecode = dis.Bytecode(fun)


class TestFlowInfo(unittest.TestCase):

    def test_constructor(self):
        flowinfo = FlowInfo()
        self.assertEqual(len(flowinfo.block_offsets), 0)
        self.assertEqual(len(flowinfo.jump_insts), 0)

    def test_from_bytecode(self):

        print("\n".join([str(b) for b in bytecode]))
        expected = FlowInfo(block_offsets={BCLabel(offset=0)},
                            jump_insts={BCLabel(offset=6): ()},
                            last_offset=6
                            )

        received = FlowInfo.from_bytecode(bytecode)
        self.assertEqual(expected, received)

    def test_build_basic_blocks(self):
        expected = BlockMap(graph={
            BCLabel(offset=0): Block(begin=BCLabel(offset=0),
                                     end=BCLabel(offset=6),
                                     fallthrough=True,
                                     jump_targets=(BCLabel(offset=6),),
                                     backedges=()
                                     )
                                  }
                           )
        received = FlowInfo.from_bytecode(bytecode).build_basicblocks()
        self.assertEqual(expected, received)


if __name__ == '__main__':
    unittest.main()
