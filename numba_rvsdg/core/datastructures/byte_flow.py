import dis
from copy import deepcopy
from dataclasses import dataclass

from numba_rvsdg.core.datastructures.block_map import BlockMap
from numba_rvsdg.core.datastructures.basic_block import RegionBlock
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.core.utils import _logger, _LogWrap

from numba_rvsdg.core.transformations import restructure_loop, restructure_branch


@dataclass(frozen=True)
class ByteFlow:
    bc: dis.Bytecode
    bbmap: "BlockMap"

    @staticmethod
    def from_bytecode(code) -> "ByteFlow":
        bc = dis.Bytecode(code)
        _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

        flowinfo = FlowInfo.from_bytecode(bc)
        bbmap = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, bbmap=bbmap)

    def _join_returns(self):
        bbmap = deepcopy(self.bbmap)
        bbmap.join_returns()
        return ByteFlow(bc=self.bc, bbmap=bbmap)

    def _restructure_loop(self):
        bbmap = deepcopy(self.bbmap)
        restructure_loop(bbmap)
        for region in _iter_subregions(bbmap):
            restructure_loop(region.subregion)
        return ByteFlow(bc=self.bc, bbmap=bbmap)

    def _restructure_branch(self):
        bbmap = deepcopy(self.bbmap)
        restructure_branch(bbmap)
        for region in _iter_subregions(bbmap):
            restructure_branch(region.subregion)
        return ByteFlow(bc=self.bc, bbmap=bbmap)

    def restructure(self):
        bbmap = deepcopy(self.bbmap)
        # close
        bbmap.join_returns()
        # handle loop
        restructure_loop(bbmap)
        for region in _iter_subregions(bbmap):
            restructure_loop(region.subregion)
        # handle branch
        restructure_branch(bbmap)
        for region in _iter_subregions(bbmap):
            restructure_branch(region.subregion)
        return ByteFlow(bc=self.bc, bbmap=bbmap)


def _iter_subregions(bbmap: "BlockMap"):
    for node in bbmap.graph.values():
        if isinstance(node, RegionBlock):
            yield node
            yield from _iter_subregions(node.subregion)
