import dis
from dataclasses import dataclass
from typing import Callable

from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.core.datastructures.flow_info import FlowInfo
from numba_scfg.core.utils import _logger, _LogWrap


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
