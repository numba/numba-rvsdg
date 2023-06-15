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
    """
        The ByteFlow class represents the flow of  bytecode and its 
        corresponding structured control flow graph (SCFG).
   """
    bc: dis.Bytecode
    """The dis.Bytecode object representing the bytecode."""
    scfg: "SCFG"
    """The structured control flow graph (SCFG) representing the control flow of 
    the bytecode."""

    @staticmethod
    def from_bytecode(code) -> "ByteFlow":
        """
            Creates a ByteFlow object from the given code, which is the bytecode. 
            This method uses dis.Bytecode to parse the bytecode, builds the basic blocks 
            and flow information from it, and returns a ByteFlow object with the 
            bytecode and the SCFG.
        """
        bc = dis.Bytecode(code)
        _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

        flowinfo = FlowInfo.from_bytecode(bc)
        scfg = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, scfg=scfg)

    def _join_returns(self):
        """
            Creates a deep copy of the SCFG and performs the operation to join
            return blocks within the control flow. It returns a new ByteFlow 
            object with the updated SCFG.
        """
        scfg = deepcopy(self.scfg)
        scfg.join_returns()
        return ByteFlow(bc=self.bc, scfg=scfg)

    def _restructure_loop(self):
        """
            Creates a deep copy of the SCFG and performs the operation to 
            restructure loop constructs within the control flow. It applies 
            the restructuring operation to both the main SCFG and any 
            subregions within it. It returns a new ByteFlow object with 
            the updated SCFG.
        """
        scfg = deepcopy(self.scfg)
        restructure_loop(scfg.region)
        for region in _iter_subregions(scfg):
            restructure_loop(region)
        return ByteFlow(bc=self.bc, scfg=scfg)

    def _restructure_branch(self):
        """
            Creates a deep copy of the SCFG and performs the operation to 
            restructure branch constructs within the control flow. It applies 
            the restructuring operation to both the main SCFG and any 
            subregions within it. It returns a new ByteFlow object with 
            the updated SCFG.
        """
        scfg = deepcopy(self.scfg)
        restructure_branch(scfg.region)
        for region in _iter_subregions(scfg):
            restructure_branch(region)
        return ByteFlow(bc=self.bc, scfg=scfg)

    def restructure(self):
        """
            Creates a deep copy of the SCFG and applies a series of 
            restructuring operations to it. The operations include 
            joining return points, restructuring loop constructs, and 
            restructuring branch constructs. It returns a new ByteFlow 
            object with the updated SCFG.
        """
        scfg = deepcopy(self.scfg)
        # close
        scfg.join_returns()
        # handle loop
        restructure_loop(scfg.region)
        for region in _iter_subregions(scfg):
            restructure_loop(region)
        # handle branch
        restructure_branch(scfg.region)
        for region in _iter_subregions(scfg):
            restructure_branch(region)
        return ByteFlow(bc=self.bc, scfg=scfg)


def _iter_subregions(scfg: "SCFG"):
    for node in scfg.graph.values():
        if isinstance(node, RegionBlock):
            yield node
            yield from _iter_subregions(node.subregion)
