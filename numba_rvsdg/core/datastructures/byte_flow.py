import dis
from copy import deepcopy
from dataclasses import dataclass

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import RegionBlock
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.core.utils import _logger, _LogWrap

from numba_rvsdg.core.transformations import restructure_loop, restructure_branch


@dataclass(frozen=True)
class ByteFlow:
    bc: dis.Bytecode
    scfg: "SCFG"

    @staticmethod
    def from_bytecode(code) -> "ByteFlow":
        bc = dis.Bytecode(code)
        _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

        flowinfo = FlowInfo.from_bytecode(bc)
        scfg = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, scfg=scfg)

    def _join_returns(self):
        scfg = deepcopy(self.scfg)
        scfg.join_returns()
        return ByteFlow(bc=self.bc, scfg=scfg)

    def _restructure_loop(self):
        scfg = deepcopy(self.scfg)
        restructure_loop(scfg.parent_region)
        for region in _iter_subregions(scfg):
            restructure_loop(region)
        return ByteFlow(bc=self.bc, scfg=scfg)

    def _restructure_branch(self):
        scfg = deepcopy(self.scfg)
        restructure_branch(scfg.parent_region)
        for region in _iter_subregions(scfg):
            restructure_branch(region)
        return ByteFlow(bc=self.bc, scfg=scfg)

    def restructure(self):
        scfg = deepcopy(self.scfg)
        # close
        scfg.join_returns()
        # handle loop
        restructure_loop(scfg.parent_region)
        for region in _iter_subregions(scfg):
            restructure_loop(region)
        # handle branch
        restructure_branch(scfg.parent_region)
        for region in _iter_subregions(scfg):
            restructure_branch(region)
        return ByteFlow(bc=self.bc, scfg=scfg)


def _iter_subregions(scfg: "SCFG"):
    for node in scfg.graph.values():
        if isinstance(node, RegionBlock):
            yield node
            yield from _iter_subregions(node.subregion)
