from unittest import TestCase

from typing import Dict, List
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
            self.assertEqual(first_scfg.back_edges[key1], second_scfg.back_edges[key2])

    def assertYAMLEquals(self, first_yaml: str, second_yaml: str, ref_dict: Dict):
        for key, value in ref_dict.items():
            second_yaml = second_yaml.replace(repr(value), key)

        self.assertEqual(first_yaml, second_yaml)
    
    def assertDictEquals(self, first_dict: str, second_dict: str, ref_dict: Dict):
        self.assertEqual(first_dict, second_dict)
