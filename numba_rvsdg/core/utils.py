import logging
import sys

_logger = logging.getLogger(__name__)

PYVERSION = sys.version_info[:2]


class _LogWrap:
    def __init__(self, fn):  # type: ignore
        self._fn = fn

    def __str__(self):  # type: ignore
        return self._fn()


_cond_jump = {
    "FOR_ITER",
    "POP_JUMP_IF_FALSE",
    "JUMP_IF_FALSE_OR_POP",
    "POP_JUMP_IF_TRUE",
    "JUMP_IF_TRUE_OR_POP",
    "POP_JUMP_FORWARD_IF_TRUE",
    "POP_JUMP_BACKWARD_IF_TRUE",
    "POP_JUMP_FORWARD_IF_FALSE",
    "POP_JUMP_BACKWARD_IF_FALSE",
    "POP_JUMP_FORWARD_IF_NOT_NONE",
    "POP_JUMP_BACKWARD_IF_NOT_NONE",
    "POP_JUMP_FORWARD_IF_NONE",
    "POP_JUMP_BACKWARD_IF_NONE",
}
_uncond_jump = {"JUMP_ABSOLUTE", "JUMP_FORWARD", "JUMP_BACKWARD"}
_terminating = {"RETURN_VALUE"}


def is_conditional_jump(opname: str) -> bool:
    return opname in _cond_jump


def is_unconditional_jump(opname: str) -> bool:
    return opname in _uncond_jump


def is_exiting(opname: str) -> bool:
    return opname in _terminating


def _next_inst_offset(offset: int) -> int:
    # Fix offset
    assert isinstance(offset, int)
    return offset + 2


def _prev_inst_offset(offset: int) -> int:
    # Fix offset
    assert isinstance(offset, int)
    return offset - 2
