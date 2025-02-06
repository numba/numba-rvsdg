import dis
import ast
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, replace, field

from numba_scfg.core.utils import _next_inst_offset
from numba_scfg.core.datastructures import block_names


@dataclass(frozen=True)
class BasicBlock:
    """Basic building block of an SCFG graph.

    The BasicBlock class represents an atomic basic block in an SCFG.

    Attributes
    ----------
    name: str
        The corresponding name for this block.

    _jump_targets: Tuple[str]
        Jump targets (branch destinations) for this block.

    backedges: Tuple[str]
        Backedges for this block.
    """

    name: str

    _jump_targets: Tuple[str, ...] = tuple()

    backedges: Tuple[str, ...] = tuple()

    @property
    def is_exiting(self) -> bool:
        """Indicates whether this block is an exiting block, i.e.,
        it does not have any jump targets.

        Returns
        -------
        is_exiting: bool
            True if the current block is an exiting block, False if it
            isn't.

        """
        return not self.jump_targets

    @property
    def fallthrough(self) -> bool:
        """Indicates whether this block has a single fallthrough jump target.

        Returns
        -------
        fallthrough: bool
            True if the current block is a fallthorough block, False if it
            isn't.

        """
        return len(self._jump_targets) == 1

    @property
    def jump_targets(self) -> Tuple[str, ...]:
        """Retrieves the jump targets for this block,
        excluding any jump targets that are also backedges.

        Returns
        -------
        jump_targets: Tuple[str]
            Tuple of jump targets of this block, (exludes backedges).
            Ordered according to their position.

        """
        acc = []
        for j in self._jump_targets:
            if j not in self.backedges:
                acc.append(j)
        return tuple(acc)

    def declare_backedge(self, target: str) -> "BasicBlock":
        """Declare one of the jump targets as a backedge of this block.

        Parameters
        ----------
        target: str
            The jump target that is to be declared as a backedge.

        Returns
        -------
        basic_block: BasicBlock
            The resulting block.

        """
        if target in self.jump_targets:
            assert not self.backedges
            return replace(self, backedges=(target,))
        return self

    def replace_jump_targets(
        self, jump_targets: Tuple[str, ...]
    ) -> "BasicBlock":
        """Replaces jump targets of this block by the given tuple.

        This method replaces the jump targets of the current BasicBlock.
        The provided jump targets must be in the same order as their
        intended original replacements.

        Note that replacing jump targets will not replace the backedge
        tuple, so replacement for any jump targets that is declared as
        a backedge needs to be updated separately using replace_backedges

        Parameters
        ----------
        jump_targets: Tuple
            The new jump target tuple. Must be ordered.

        Returns
        -------
        basic_block: BasicBlock
            The resulting BasicBlock.

        """
        return replace(self, _jump_targets=jump_targets)

    def replace_backedges(self, backedges: Tuple[str, ...]) -> "BasicBlock":
        """Replaces back edges of this block by the given tuple.

        This method replaces the back edges of the current BasicBlock.
        The provided back edges must be in the same order as their
        intended original replacements.

        Parameters
        ----------
        backedges: Tuple
            The new back edges tuple. Must be ordered.

        Returns
        -------
        basic_block: BasicBlock
            The resulting BasicBlock.

        """
        return replace(self, backedges=backedges)


@dataclass(frozen=True)
class PythonBytecodeBlock(BasicBlock):
    """The PythonBytecodeBlock class is a subclass of the BasicBlock that
    represents basic blocks with Python bytecode.

    Attributes
    ----------
    begin: int
        The starting bytecode offset.

    end: int
        The bytecode offset immediately after the last bytecode of the block.
    """

    begin: int = -1

    end: int = -1

    def get_instructions(
        self, bcmap: Dict[int, dis.Instruction]
    ) -> List[dis.Instruction]:
        """Retrieves a list of `dis.Instruction` objects corresponding to
        the instructions within the bytecode block.

        In this method, The bcmap parameter is a dictionary mapping bytecode
        offsets to `dis.Instruction` objects. This method iterates over the
        bytecode offsets within the begin and end range, retrieves the
        corresponding `dis.Instruction` objects from bcmap, and returns a list
        of these instructions.

        Parameters
        ----------
        bcmap: Dict[int, dis.Instruction]
            Dictionary mapping bytecode offsets to dis.Instruction objects.

        Return
        ------
        out: List[dis.Instruction]
            The requested instructions according to bcmap between begin and
            end offsets.

        """
        begin = self.begin
        end = self.end
        it = begin
        out = []
        while it < end:
            # Python 3.11 hack: account for gaps in the bytecode sequence
            try:
                out.append(bcmap[it])
            except KeyError:
                pass
            finally:
                it = _next_inst_offset(it)

        return out


@dataclass(frozen=True)
class PythonASTBlock(BasicBlock):
    """The PythonASTBlock class is a subclass of the BasicBlock that
    represents basic blocks with Python AST.

    Attributes
    ----------
    begin: int
        The starting line.

    end: int
        The ending line.
    """

    begin: int = -1

    end: int = -1

    tree: List[ast.AST] = field(default_factory=lambda: [])

    def get_tree(self) -> List[ast.AST]:
        return self.tree


@dataclass(frozen=True)
class SyntheticBlock(BasicBlock):
    """The SyntheticBlock represents a artificially added block in a
    structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class SyntheticExit(SyntheticBlock):
    """The SyntheticExit class represents a artificially added exit block
    in a structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class SyntheticReturn(SyntheticBlock):
    """The SyntheticReturn class represents a artificially added return block
    in a structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class SyntheticTail(SyntheticBlock):
    """The SyntheticTail class represents a artificially added tail block
    in a structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class SyntheticFill(SyntheticBlock):
    """The SyntheticFill class represents a artificially added fill block
    in a structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class SyntheticAssignment(SyntheticBlock):
    """The SyntheticAssignment class represents a artificially added
    assignment block in a structured control flow graph (SCFG).

    This block is responsible for giving variables their values,
    once the respective block is executed.

    Attributes
    ----------
    variable_assignment: dict
        A dictionary representing the variable assignments. It maps
        the variable name to the value that is is assigned when
        the block is executed.
    """

    variable_assignment: Dict[str, int] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class SyntheticBranch(SyntheticBlock):
    """The SyntheticBranch class represents a artificially added branch block
    in a structured control flow graph (SCFG).

    Attributes
    ----------
    variable: str
        The variable on the basis of which branching will happen when the
        current block is executed.
    branch_value_table: dict
        The value table maps variable values to the repective jump target
        to be executed on the basis of that value.
    """

    variable: str = ""
    branch_value_table: Dict[int, str] = field(default_factory=lambda: {})

    def replace_jump_targets(
        self, jump_targets: Tuple[str, ...]
    ) -> "BasicBlock":
        """Replaces jump targets of this block by the given tuple.

        This method replaces the jump targets of  the current BasicBlock.
        The provided jump targets must be in the same order as their
        intended original replacements. Additionally also updates the
        branch value table of the branch block.

        Note that replacing jump targets will not replace the backedge
        tuple, so replacement for any jump targets that is declared as
        a backedge needs to be updated separately using replace_backedges.

        Parameters
        ----------
        jump_targets: Tuple
            The new jump target tuple. Must be ordered.

        Returns
        -------
        basic_block: BasicBlock
            The resulting BasicBlock.

        """

        old_branch_value_table = self.branch_value_table
        new_branch_value_table = {}
        for target in self._jump_targets:
            if target not in jump_targets:
                # ASSUMPTION: only one jump_target is being updated
                diff = set(jump_targets).difference(self._jump_targets)
                assert len(diff) == 1
                new_target = next(iter(diff))
                for k, v in old_branch_value_table.items():
                    if v == target:
                        new_branch_value_table[k] = new_target
            else:
                # copy all old values
                for k, v in old_branch_value_table.items():
                    if v == target:
                        new_branch_value_table[k] = v

        return replace(
            self,
            _jump_targets=jump_targets,
            branch_value_table=new_branch_value_table,
        )


@dataclass(frozen=True)
class SyntheticHead(SyntheticBranch):
    """The SyntheticHead class represents a artificially added head block
    in a structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class SyntheticExitingLatch(SyntheticBranch):
    """The SyntheticExitingLatch class represents a artificially added
    exiting latch block in a structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class SyntheticExitBranch(SyntheticBranch):
    """The SyntheticExitBranch class represents a artificially added
    exit branch block in a structured control flow graph (SCFG).
    """


@dataclass(frozen=True)
class RegionBlock(BasicBlock):
    """The RegionBlock is a BasicBlock that represents a region in a
    structured control flow graph (SCFG) object.

    Attributes
    ----------
    kind: str
        The kind of region. Can be 'head', 'tail', 'branch',
        'loop' or 'meta' strings.
    parent_region: "RegionBlock"
        The parent region of this region as per the SCFG.
    header: str
        The header node of the region.
    subregion: "SCFG"
        The subgraph as an independent SCFG. Note that in case
        of subregions the exiting node may point to blocks outside
        of the current SCFG object.
    exiting: str
        The exiting node of the region.
    """

    kind: Optional[str] = None
    parent_region: Optional["RegionBlock"] = None
    header: Optional[str] = None
    subregion: Optional["SCFG"] = None  # type: ignore  # noqa
    exiting: Optional[str] = None

    def replace_header(self, new_header: str) -> None:
        """This method performs a inplace replacement of the header block.

        Parameters
        ----------
        new_header: str
            The new header block of the region represented by the RegionBlock.
        """
        object.__setattr__(self, "header", new_header)

    def replace_exiting(self, new_exiting: str) -> None:
        """This method performs a inplace replacement of the header block.

        Parameters
        ----------
        new_exiting: str
            The new exiting block of the region represented by the RegionBlock.
        """
        object.__setattr__(self, "exiting", new_exiting)


block_type_names = {
    block_names.BASIC: BasicBlock,
    block_names.PYTHON_BYTECODE: PythonBytecodeBlock,
    block_names.SYNTH_HEAD: SyntheticHead,
    block_names.SYNTH_BRANCH: SyntheticBranch,
    block_names.SYNTH_TAIL: SyntheticTail,
    block_names.SYNTH_EXIT: SyntheticExit,
    block_names.SYNTH_ASSIGN: SyntheticAssignment,
    block_names.SYNTH_RETURN: SyntheticReturn,
    block_names.SYNTH_EXIT_LATCH: SyntheticExitingLatch,
    block_names.SYNTH_EXIT_BRANCH: SyntheticExitBranch,
    block_names.SYNTH_FILL: SyntheticFill,
    block_names.REGION: RegionBlock,
}
