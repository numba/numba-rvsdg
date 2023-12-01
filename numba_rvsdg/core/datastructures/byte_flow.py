import dis
from dataclasses import dataclass
from typing import Generator, Callable

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import RegionBlock
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.core.utils import _logger, _LogWrap

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
    scfg: SCFG

    @staticmethod
    def from_bytecode(code: Callable) -> "ByteFlow":  # type: ignore
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
        _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))  # type: ignore  # noqa E501

        flowinfo = FlowInfo.from_bytecode(bc)
        scfg = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, scfg=scfg)

    def _join_returns(self) -> None:
        """Joins the return blocks within the corresponding SCFG.

        This method performs operation to join return blocks within
        the control flow.
        """
        self.scfg.join_returns()

    def _restructure_loop(self) -> None:
        """Restructures the loops within the corresponding SCFG.

        Performs the operation to restructure loop constructs within
        the control flow using the algorithm LOOP RESTRUCTURING from
        section 4.1 of Bahmann2015.
        It applies the restructuring operation to both the main SCFG
        and any subregions within it.
        """
        restructure_loop(self.scfg.region)
        for region in _iter_subregions(self.scfg):
            restructure_loop(region)

    def _restructure_branch(self) -> None:
        """Restructures the branches within the corresponding SCFG.

        This method applies restructuring branch operation to both
        the main SCFG and any subregions within it.
        """
        restructure_branch(self.scfg.region)
        for region in _iter_subregions(self.scfg):
            restructure_branch(region)

    def restructure(self) -> None:
        """Applies join_returns, restructure_loop and restructure_branch
        in the respective order on the SCFG.

        Applies a series of restructuring operations to given SCFG.
        The operations include joining return blocks, restructuring
        loop constructs, and restructuring branch constructs.
        """
        # close
        self._join_returns()
        # handle loop
        self._restructure_loop()
        # handle branch
        self._restructure_branch()


def _iter_subregions(scfg: SCFG) -> Generator[RegionBlock, SCFG, None]:
    for node in scfg.graph.values():
        if isinstance(node, RegionBlock):
            yield node
            assert node.subregion is not None
            yield from _iter_subregions(node.subregion)
