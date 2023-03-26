import dis
from dataclasses import dataclass

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.flow_info import FlowInfo
from numba_rvsdg.core.utils import _logger, _LogWrap

from numba_rvsdg.core.transformations import restructure_loop, restructure_branch, join_returns


@dataclass(frozen=True)
class ByteFlow:
    bc: dis.Bytecode
    bbmap: SCFG

    @staticmethod
    def from_bytecode(code) -> "ByteFlow":
        bc = dis.Bytecode(code)
        _logger.debug("Bytecode\n%s", _LogWrap(lambda: bc.dis()))

        flowinfo = FlowInfo.from_bytecode(bc)
        bbmap = flowinfo.build_basicblocks()
        return ByteFlow(bc=bc, bbmap=bbmap)

    def _join_returns(self):
        join_returns(self.bbmap)

    def _restructure_loop(self):
        restructure_loop(self.bbmap)

    def _restructure_branch(self):
        restructure_branch(self.bbmap)

    def restructure(self):
        # close
        join_returns(self.bbmap)
        # handle loop
        restructure_loop(self.bbmap)
        # handle branch
        restructure_branch(self.bbmap)

    @staticmethod
    def bcmap_from_bytecode(bc: dis.Bytecode):
        return {inst.offset: inst for inst in bc}

