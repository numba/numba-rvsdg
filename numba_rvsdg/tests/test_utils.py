from unittest import TestCase
import yaml

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import BasicBlock


class SCFGComparator(TestCase):
    def assertSCFGEqual(
        self, first_scfg: SCFG, second_scfg: SCFG, head_map=None
    ):
        if head_map:
            # If more than one head the corresponding map needs to be provided
            block_mapping = head_map
            stack = list(block_mapping.keys())
        else:
            first_head = first_scfg.find_head()
            second_head = second_scfg.find_head()
            block_mapping = {first_head: second_head}
            stack = [first_head]

        # Assert number of blocks are equal in both SCFGs
        assert len(first_scfg.graph) == len(
            second_scfg.graph
        ), "Number of blocks in both graphs are not equal"
        seen = set()

        while stack:
            node_name: BasicBlock = stack.pop()
            if node_name in seen:
                continue
            seen.add(node_name)
            node: BasicBlock = first_scfg[node_name]
            # Assert that there's a corresponding mapping of current node
            # in second scfg
            assert node_name in block_mapping.keys()
            # Get the corresponding node in second graph
            second_node_name = block_mapping[node_name]
            second_node: BasicBlock = second_scfg[second_node_name]
            # Both nodes should have equal number of jump targets and backedges
            assert len(node.jump_targets) == len(second_node.jump_targets)
            assert len(node.backedges) == len(second_node.backedges)

            # Add the jump targets as corresponding nodes in block mapping
            # dictionary. Since order must be same we can simply add zip
            # functionality as the correspondence function for nodes
            for jt1, jt2 in zip(node.jump_targets, second_node.jump_targets):
                block_mapping[jt1] = jt2
                stack.append(jt1)

            for be1, be2 in zip(node.backedges, second_node.backedges):
                block_mapping[be1] = be2
                stack.append(be1)

    def assertYAMLEqual(
        self, first_yaml: SCFG, second_yaml: SCFG, head_map: dict
    ):
        self.assertDictEqual(
            yaml.safe_load(first_yaml), yaml.safe_load(second_yaml), head_map
        )

    def assertDictEqual(
        self, first_yaml: str, second_yaml: str, head_map: dict
    ):
        block_mapping = head_map
        stack = list(block_mapping.keys())
        # Assert number of blocks are equal in both SCFGs
        assert len(first_yaml) == len(
            second_yaml
        ), "Number of blocks in both graphs are not equal"
        seen = set()

        while stack:
            node_name: BasicBlock = stack.pop()
            if node_name in seen:
                continue
            seen.add(node_name)
            node: BasicBlock = first_yaml[node_name]
            # Assert that there's a corresponding mapping of current node
            # in second scfg
            assert node_name in block_mapping.keys()
            # Get the corresponding node in second graph
            second_node_name = block_mapping[node_name]
            second_node: BasicBlock = second_yaml[second_node_name]
            # Both nodes should have equal number of jump targets and backedges
            assert len(node["jt"]) == len(second_node["jt"])
            if "be" in node.keys():
                assert len(node["be"]) == len(second_node["be"])

            # Add the jump targets as corresponding nodes in block mapping
            # dictionary. Since order must be same we can simply add zip
            # functionality as the correspondence function for nodes
            for jt1, jt2 in zip(node["jt"], second_node["jt"]):
                block_mapping[jt1] = jt2
                stack.append(jt1)
