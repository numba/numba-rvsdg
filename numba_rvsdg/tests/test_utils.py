from unittest import TestCase

class SCFGComparator(TestCase):
    def assertSCFGEqual(self, first_scfg, second_scfg):

        for key1, key2 in zip(
            sorted(first_scfg.graph.keys(), key=lambda x: x.index),
            sorted(second_scfg.graph.keys(), key=lambda x: x.index),
        ):
            # compare indices of labels
            self.assertEqual(key1.index, key2.index)
            # compare indices of jump_targets
            self.assertEqual(
                sorted([j.index for j in first_scfg[key1]._jump_targets]),
                sorted([j.index for j in second_scfg[key2]._jump_targets]),
            )
            # compare indices of backedges
            self.assertEqual(
                sorted([j.index for j in first_scfg[key1].backedges]),
                sorted([j.index for j in second_scfg[key2].backedges]),
            )
