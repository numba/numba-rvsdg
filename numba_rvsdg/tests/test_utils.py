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

    def assertDictEquals(self, first_dict: dict, second_dict: dict, ref_dict: dict):

        def replace_with_refs(scfg_dict: dict):
            new_dict = {}
            for key, value in scfg_dict.items():
                key = str(ref_dict[key])
                _new_dict = {}
                for _key, _value in value.items():
                    if isinstance(_value, list):
                        for i in range(len(_value)):
                            _value[i] = str(ref_dict[_value[i]])
                    _new_dict[_key] = _value
                new_dict[key] = _new_dict

            return new_dict

        first_dict = replace_with_refs(first_dict)
        self.assertEqual(first_dict, second_dict)
