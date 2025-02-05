# mypy: ignore-errors

from unittest import TestCase
import yaml

from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.core.datastructures.basic_block import (
    BasicBlock,
    RegionBlock,
    SyntheticBranch,
    SyntheticAssignment,
)


class SCFGComparator(TestCase):
    def assertSCFGEqual(
        self, first_scfg: SCFG, second_scfg: SCFG, head_map=None, exiting=None
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

            # If the given block is a RegionBlock, then the underlying SCFGs
            # for both regions must be equal.
            if isinstance(node, RegionBlock):
                self.assertSCFGEqual(
                    node.subregion, second_node.subregion, exiting=node.exiting
                )
            elif isinstance(node, SyntheticAssignment):
                assert (
                    node.variable_assignment == second_node.variable_assignment
                )
            elif isinstance(node, SyntheticBranch):
                assert (
                    node.branch_value_table == second_node.branch_value_table
                )
                assert node.variable == second_node.variable

            # Add the jump targets as corresponding nodes in block mapping
            # dictionary. Since order must be same we can simply add zip
            # functionality as the correspondence function for nodes
            for jt1, jt2 in zip(node.jump_targets, second_node.jump_targets):
                if node.name == exiting:
                    continue
                block_mapping[jt1] = jt2
                stack.append(jt1)

            for be1, be2 in zip(node.backedges, second_node.backedges):
                if node.name == exiting:
                    continue
                block_mapping[be1] = be2
                stack.append(be1)

    def assertYAMLEqual(
        self, first_yaml: str, second_yaml: str, head_map: dict
    ):
        self.assertDictEqual(
            yaml.safe_load(first_yaml), yaml.safe_load(second_yaml), head_map
        )

    def assertDictEqual(  # type: ignore
        self, first_yaml: dict, second_yaml: dict, head_map: dict
    ):
        block_mapping = head_map
        stack = list(block_mapping.keys())
        # Assert number of blocks are equal in both SCFGs
        assert len(first_yaml) == len(
            second_yaml
        ), "Number of blocks in both graphs are not equal"
        seen = set()

        while stack:
            node_name = stack.pop()
            if node_name in seen:
                continue
            seen.add(node_name)
            # Assert that there's a corresponding mapping of current node
            # in second scfg
            assert node_name in block_mapping.keys()
            co_node_name = block_mapping[node_name]

            node_properties = first_yaml["blocks"][node_name]
            co_node_properties = second_yaml["blocks"][co_node_name]
            assert node_properties == co_node_properties

            # Both nodes should have equal number of jump targets and backedges
            assert len(first_yaml["edges"][node_name]) == len(
                second_yaml["edges"][co_node_name]
            )
            if first_yaml["backedges"] and first_yaml["backedges"].get(
                node_name
            ):
                assert len(first_yaml["backedges"][node_name]) == len(
                    second_yaml["backedges"][co_node_name]
                )

            # Add the jump targets as corresponding nodes in block mapping
            # dictionary. Since order must be same we can simply add zip
            # functionality as the correspondence function for nodes
            for jt1, jt2 in zip(
                first_yaml["edges"][node_name],
                second_yaml["edges"][co_node_name],
            ):
                block_mapping[jt1] = jt2
                stack.append(jt1)
