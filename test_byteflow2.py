from dis import Bytecode, Instruction, Positions

import unittest
from byteflow2 import (FlowInfo, PythonBytecodeBlock, PythonBytecodeLabel, BasicBlock, BlockMap, ByteFlow,
                       bcmap_from_bytecode)


def fun():
    x = 1
    return x


bytecode = Bytecode(fun)


class TestBCMapFromBytecode(unittest.TestCase):

    def test(self):

        expected = {0: Instruction(
                        opname='RESUME', opcode=151, arg=0, argval=0,
                        argrepr='', offset=0, starts_line=8,
                        is_jump_target=False,
                        positions=Positions(
                            lineno=8, end_lineno=8,
                            col_offset=0, end_col_offset=0)),
                    2: Instruction(
                        opname='LOAD_CONST', opcode=100, arg=1, argval=1,
                        argrepr='1', offset=2, starts_line=9,
                        is_jump_target=False,
                        positions=Positions(
                            lineno=9, end_lineno=9,
                            col_offset=8, end_col_offset=9)),
                    4: Instruction(
                        opname='STORE_FAST', opcode=125, arg=0, argval='x',
                        argrepr='x', offset=4, starts_line=None,
                        is_jump_target=False,
                        positions=Positions(
                            lineno=9, end_lineno=9,
                            col_offset=4, end_col_offset=5)),
                    6: Instruction(
                        opname='LOAD_FAST', opcode=124, arg=0, argval='x',
                        argrepr='x', offset=6, starts_line=10,
                        is_jump_target=False,
                        positions=Positions(
                            lineno=10, end_lineno=10,
                            col_offset=11, end_col_offset=12)),
                    8: Instruction(
                        opname='RETURN_VALUE', opcode=83, arg=None,
                        argval=None, argrepr='', offset=8, starts_line=None,
                        is_jump_target=False,
                        positions=Positions(
                            lineno=10, end_lineno=10,
                            col_offset=4, end_col_offset=12))}
        received = bcmap_from_bytecode(bytecode)
        self.assertEqual(expected, received)


class TestPythonBytecodeBlock(unittest.TestCase):

    def test_constructor(self):
        block = PythonBytecodeBlock(
            label=PythonBytecodeLabel(index=0),
            begin=0,
            end=8,
            _jump_targets=(),
            backedges=()
        )
        self.assertEqual(block.label, PythonBytecodeLabel(index=0))
        self.assertEqual(block.begin, 0)
        self.assertEqual(block.end, 8)
        self.assertFalse(block.fallthrough)
        self.assertTrue(block.is_exiting)
        self.assertEqual(block.jump_targets, ())
        self.assertEqual(block.backedges, ())

    def test_is_jump_target(self):
        block = PythonBytecodeBlock(
            label=PythonBytecodeLabel(index=0),
            begin=0,
            end=8,
            _jump_targets=(PythonBytecodeLabel(index=1), ),
            backedges=()
        )
        self.assertEqual(block.jump_targets, (PythonBytecodeLabel(index=1), ))
        self.assertFalse(block.is_exiting)

    def test_get_instructions(self):
        block = PythonBytecodeBlock(
            label=PythonBytecodeLabel(index=0),
            begin=0,
            end=8,
            _jump_targets=(),
            backedges=()
        )
        expected = [Instruction(
            opname='RESUME', opcode=151, arg=0, argval=0, argrepr='',
            offset=0, starts_line=8, is_jump_target=False,
            positions=Positions(lineno=8, end_lineno=8,
                                col_offset=0, end_col_offset=0)),
                       Instruction(
            opname='LOAD_CONST', opcode=100, arg=1, argval=1, argrepr='1',
            offset=2, starts_line=9, is_jump_target=False,
            positions=Positions(lineno=9, end_lineno=9,
                                col_offset=8, end_col_offset=9)),
                       Instruction(
            opname='STORE_FAST', opcode=125, arg=0, argval='x', argrepr='x',
            offset=4, starts_line=None, is_jump_target=False,
            positions=Positions(lineno=9, end_lineno=9,
                                col_offset=4, end_col_offset=5)),
                        Instruction(opname='LOAD_FAST', opcode=124, arg=0, argval='x',
            argrepr='x', offset=6, starts_line=10, is_jump_target=False,
            positions=Positions(lineno=10, end_lineno=10,
                                col_offset=11, end_col_offset=12))]

        received = block.get_instructions(bcmap_from_bytecode(bytecode))
        self.assertEqual(expected, received)


class TestFlowInfo(unittest.TestCase):

    def test_constructor(self):
        flowinfo = FlowInfo()
        self.assertEqual(len(flowinfo.block_offsets), 0)
        self.assertEqual(len(flowinfo.jump_insts), 0)

    def test_from_bytecode(self):

        expected = FlowInfo(block_offsets={0},
                            jump_insts={8: ()},
                            last_offset=8
                            )

        received = FlowInfo.from_bytecode(bytecode)
        self.assertEqual(expected, received)

    def test_build_basic_blocks(self):
        expected = BlockMap(graph={
            PythonBytecodeLabel(index=0):
            PythonBytecodeBlock(
                label=PythonBytecodeLabel(index=0),
                begin=0,
                end=10,
                _jump_targets=(),
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
            PythonBytecodeLabel(index=0):
            PythonBytecodeBlock(
                label=PythonBytecodeLabel(index=0),
                begin=0,
                end=10,
                _jump_targets=(),
                backedges=()
                )
            }
        )
        expected = ByteFlow(bc=bytecode, bbmap=bbmap)
        received = ByteFlow.from_bytecode(fun)
        self.assertEqual(expected.bbmap, received.bbmap)


if __name__ == '__main__':
    unittest.main()
