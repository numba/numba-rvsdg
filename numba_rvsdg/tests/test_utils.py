from unittest import TestCase

class MapComparator(TestCase):
    def assertMapEqual(self, first_map, second_map):

        for key1, key2 in zip(
            sorted(first_map.graph.keys(), key=lambda x: x.index),
            sorted(second_map.graph.keys(), key=lambda x: x.index),
        ):
            # compare indices of labels
            self.assertEqual(key1.index, key2.index)
            # compare indices of jump_targets
            self.assertEqual(
                sorted([j.index for j in first_map[key1]._jump_targets]),
                sorted([j.index for j in second_map[key2]._jump_targets]),
            )
            # compare indices of backedges
            self.assertEqual(
                sorted([j.index for j in first_map[key1].backedges]),
                sorted([j.index for j in second_map[key2].backedges]),
            )
