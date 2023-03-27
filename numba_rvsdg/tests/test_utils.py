from unittest import TestCase

from numba_rvsdg.core.datastructures.scfg import SCFG


class SCFGComparator(TestCase):
    def assertSCFGEqual(self, first_scfg: SCFG, second_scfg: SCFG):

        for key1, key2 in zip(
            sorted(first_scfg.blocks.keys(), key=lambda x: x.name),
            sorted(second_scfg.blocks.keys(), key=lambda x: x.name),
        ):
            block_1 = first_scfg[key1]
            block_2 = second_scfg[key2]

            # compare labels
            self.assertEqual(type(block_1.label), type(block_2.label))
            # compare edges
            self.assertEqual(first_scfg.out_edges[key1], second_scfg.out_edges[key2])
            self.assertEqual(first_scfg.in_edges[key1], second_scfg.in_edges[key2])
            self.assertEqual(first_scfg.back_edges[key1], second_scfg.back_edges[key2])
