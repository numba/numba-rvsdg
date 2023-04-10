
from numba_rvsdg.tests.test_transforms import SCFGComparator
from numba_rvsdg.core.datastructures.basic_block import BasicBlock
from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.labels import ControlLabel

class TestSCFGIterator(SCFGComparator):

    def test_scfg_iter(self):
        expected = [
            (ControlLabel("0"), BasicBlock(label=ControlLabel("0"),
                                           _jump_targets=(ControlLabel("1"),))),
            (ControlLabel("1"), BasicBlock(label=ControlLabel("1"))),
        ]
        scfg = SCFG.from_yaml("""
        "0":
            jt: ["1"]
        "1":
            jt: []
        """)
        received = list(scfg)
        self.assertEqual(expected, received)
