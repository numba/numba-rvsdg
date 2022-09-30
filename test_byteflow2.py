import dis

import unittest
from byteflow2 import (FlowInfo, BCLabel, BasicBlock, BlockMap, ByteFlow,
                       bcmap_from_bytecode)


def fun():
    x = 1
    return x


bytecode = dis.Bytecode(fun)


class TestBCMapFromBytecode(unittest.TestCase):

    def test(self):
        expected = {BCLabel(offset=0): dis.Instruction(
                        opname='LOAD_CONST',
                        opcode=100,
                        arg=1,
                        argval=1,
                        argrepr='1',
                        offset=0,
                        starts_line=9,
                        is_jump_target=False),
                    BCLabel(offset=2): dis.Instruction(
                        opname='STORE_FAST',
                        opcode=125,
                        arg=0, argval='x',
                        argrepr='x',
                        offset=2,
                        starts_line=None,
                        is_jump_target=False),
                    BCLabel(offset=4): dis.Instruction(
                        opname='LOAD_FAST',
                        opcode=124,
                        arg=0,
                        argval='x',
                        argrepr='x',
                        offset=4,
                        starts_line=10,
                        is_jump_target=False),
                    BCLabel(offset=6): dis.Instruction(
                        opname='RETURN_VALUE',
                        opcode=83,
                        arg=None,
                        argval=None,
                        argrepr='',
                        offset=6,
                        starts_line=None,
                        is_jump_target=False)}
        received = bcmap_from_bytecode(bytecode)
        self.assertEqual(expected, received)


class TestBlock(unittest.TestCase):

    def test_constructor(self):
        block = BasicBlock(
            begin=BCLabel(offset=0),
            end=BCLabel(offset=8),
            fallthrough=False,
            jump_targets=(),
            backedges=()
        )
        self.assertEqual(block.begin, BCLabel(offset=0))
        self.assertEqual(block.end, BCLabel(offset=8))
        self.assertEqual(block.fallthrough, False)
        self.assertEqual(block.jump_targets, ())
        self.assertEqual(block.backedges, ())
        self.assertTrue(block.is_exiting())

    def test_is_jump_target(self):
        block = BasicBlock(
            begin=BCLabel(offset=0),
            end=BCLabel(offset=8),
            fallthrough=False,
            jump_targets=(BCLabel(10)),
            backedges=()
        )
        self.assertEqual(block.jump_targets, (BCLabel(10)))
        self.assertFalse(block.is_exiting())

    def test_get_instructions(self):
        block = BasicBlock(
            begin=BCLabel(offset=0),
            end=BCLabel(offset=8),
            fallthrough=False,
            jump_targets=(),
            backedges=()
        )
        expected = [dis.Instruction(
            opname='LOAD_CONST', opcode=100, arg=1, argval=1,
            argrepr='1', offset=0, starts_line=9, is_jump_target=False),
                    dis.Instruction(
            opname='STORE_FAST', opcode=125, arg=0, argval='x',
            argrepr='x', offset=2, starts_line=None, is_jump_target=False),
                    dis.Instruction(
            opname='LOAD_FAST', opcode=124, arg=0, argval='x',
            argrepr='x', offset=4, starts_line=10, is_jump_target=False),
                    dis.Instruction(
            opname='RETURN_VALUE', opcode=83, arg=None, argval=None,
            argrepr='', offset=6, starts_line=None, is_jump_target=False)]
        received = block.get_instructions(bcmap_from_bytecode(bytecode))
        self.assertEqual(expected, received)


class TestFlowInfo(unittest.TestCase):

    def test_constructor(self):
        flowinfo = FlowInfo()
        self.assertEqual(len(flowinfo.block_offsets), 0)
        self.assertEqual(len(flowinfo.jump_insts), 0)

    def test_from_bytecode(self):

        expected = FlowInfo(block_offsets={BCLabel(offset=0)},
                            jump_insts={BCLabel(offset=6): ()},
                            last_offset=6
                            )

        received = FlowInfo.from_bytecode(bytecode)
        self.assertEqual(expected, received)

    def test_build_basic_blocks(self):
        expected = BlockMap(graph={
            BCLabel(offset=0): BasicBlock(
                begin=BCLabel(offset=0),
                end=BCLabel(offset=8),
                fallthrough=False,
                jump_targets=(),
                backedges=()
                )
                                  }
                           )
        received = FlowInfo.from_bytecode(bytecode).build_basicblocks()
        self.assertEqual(expected, received)


class TestByteFlow(unittest.TestCase):

    def test_constructor(self):
        byteflow = ByteFlow([], [])
        self.assertEqual(len(byteflow.bc), 0)
        self.assertEqual(len(byteflow.bbmap), 0)

    def test_from_bytecode(self):
        bbmap = BlockMap(graph={
            BCLabel(offset=0): BasicBlock(
                begin=BCLabel(offset=0),
                end=BCLabel(offset=8),
                fallthrough=False,
                jump_targets=(),
                backedges=()
                )
                                  }
                           )
        expected = ByteFlow(bc=bytecode, bbmap=bbmap)
        received = ByteFlow.from_bytecode(fun)
        self.assertEqual(expected.bbmap, received.bbmap)


if __name__ == '__main__':
    unittest.main()
