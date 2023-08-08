import dis
from copy import deepcopy
from dataclasses import dataclass

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import RegionBlock
from numba_rvsdg.core.datastructures.flow_info import FlowInfo

# from numba_rvsdg.core.utils import _logger, _LogWrap

from numba_rvsdg.core.transformations import (
    restructure_loop,
    restructure_branch,
)


@dataclass(frozen=True)
class ByteFlow:
    """ByteFlow class.

    The ByteFlow class represents the bytecode and its relation with
    corresponding structured control flow graph (SCFG).

    Attributes
    ----------
    bc: dis.Bytecode
        The dis.Bytecode object representing the bytecode.
    scfg: SCFG
        The SCFG object representing the control flow of
        the bytecode.
    """

    bc: dis.Bytecode
    scfg: "SCFG"

    @staticmethod
    def from_bytecode(code) -> "ByteFlow":
        """Creates a ByteFlow object from the given python
        function.

        This method uses dis.Bytecode to parse the bytecode
        generated from the given Python function.
        It returns a ByteFlow object with the corresponding
        bytecode and SCFG.

        Parameters
        ----------
        code: Python Function
            The Python Function from which ByteFlow is to
            be generated.

        Returns
        -------
        byteflow: ByteFlow
            The resulting ByteFlow object.
        """
        bc = dis.Bytecode(code)
        # _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

        flowinfo = FlowInfo.from_bytecode(bc)
        scfg = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, scfg=scfg)

    def _join_returns(self):
        """Joins the return blocks within the corresponding SCFG.

        This method creates a deep copy of the SCFG and performs
        operation to join return blocks within the control flow.
        It returns a new ByteFlow object with the updated SCFG.

        Returns
        -------
        byteflow: ByteFlow
            The new ByteFlow object with updated SCFG.
        """
        scfg = deepcopy(self.scfg)
        scfg.join_returns()
        return ByteFlow(bc=self.bc, scfg=scfg)

    def _restructure_loop(self):
        """Restructures the loops within the corresponding SCFG.

        Creates a deep copy of the SCFG and performs the operation to
        restructure loop constructs within the control flow using
        the algorithm LOOP RESTRUCTURING from section 4.1 of Bahmann2015.
        It applies the restructuring operation to both the main SCFG
        and any subregions within it. It returns a new ByteFlow object
        with the updated SCFG.

        Returns
        -------
        byteflow: ByteFlow
            The new ByteFlow object with updated SCFG.
        """
        scfg = deepcopy(self.scfg)
        restructure_loop(scfg.region)
        for region in _iter_subregions(scfg):
            restructure_loop(region)
        return ByteFlow(bc=self.bc, scfg=scfg)

    def _restructure_branch(self):
        """Restructures the branches within the corresponding SCFG.

        Creates a deep copy of the SCFG and performs the operation to
        restructure branch constructs within the control flow. It applies
        the restructuring operation to both the main SCFG and any
        subregions within it. It returns a new ByteFlow object with
        the updated SCFG.

        Returns
        -------
        byteflow: ByteFlow
            The new ByteFlow object with updated SCFG.
        """
        scfg = deepcopy(self.scfg)
        restructure_branch(scfg.region)
        for region in _iter_subregions(scfg):
            restructure_branch(region)
        return ByteFlow(bc=self.bc, scfg=scfg)

    def restructure(self):
        """Applies join_returns, restructure_loop and restructure_branch
        in the respective order on the SCFG.

        Creates a deep copy of the SCFG and applies a series of
        restructuring operations to it. The operations include
        joining return blocks, restructuring loop constructs, and
        restructuring branch constructs. It returns a new ByteFlow
        object with the updated SCFG.

        Returns
        -------
        byteflow: ByteFlow
            The new ByteFlow object with updated SCFG.
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
