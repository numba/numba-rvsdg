
from numba_rvsdg.tests.test_transforms import MapComparator
from numba_rvsdg.core.datastructures.basic_block import BasicBlock
from numba_rvsdg.core.datastructures.block_map import BlockMap
from numba_rvsdg.core.datastructures.labels import ControlLabel

class TestBlockMapIterator(MapComparator):

    def test_block_map_iter(self):
        expected = [
            (ControlLabel("0"), BasicBlock(label=ControlLabel("0"),
                                           _jump_targets=(ControlLabel("1"),))),
            (ControlLabel("1"), BasicBlock(label=ControlLabel("1"))),
        ]
        block_map = BlockMap.from_yaml("""
        "0":
            jt: ["1"]
        "1":
            jt: []
        """)
        received = list(block_map)
        self.assertEqual(expected, received)
