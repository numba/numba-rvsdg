import unittest
from byteflow2 import FlowInfo


class TestFlowInfo(unittest.TestCase):

    def test_constructor(self):
        flowinfo = FlowInfo()
        self.assertEqual(len(flowinfo.block_offsets), 0)
        self.assertEqual(len(flowinfo.jump_insts), 0)


if __name__ == '__main__':
    unittest.main()
