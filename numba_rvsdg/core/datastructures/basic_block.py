import dis
from collections import ChainMap
from typing import Tuple, Dict, List
from dataclasses import dataclass, field, replace

from numba_rvsdg.core.utils import _next_inst_offset


@dataclass(frozen=True)
class BasicBlock:
    """
        The BasicBlock class represents an atomic basic block in a data flow graph.
        Note that the BasicBlock class is defined with the `frozen=True`` parameter,
        making instances of this class immutable.
    """

    name: str
    """The corresponding name for this block.  """

    _jump_targets: Tuple[str] = tuple()
    """Jump targets (branch destinations) for this block"""

    backedges: Tuple[str] = tuple()
    """Backedges for this block."""

    @property
    def is_exiting(self) -> bool:
        """Indicates whether this block is an exiting block, i.e., 
        it does not have any jump targets."""
        return not self.jump_targets

    @property
    def fallthrough(self) -> bool:
        """Indicates whether this block has a single fallthrough jump target."""
        return len(self._jump_targets) == 1

    @property
    def jump_targets(self) -> Tuple[str]:
        """Retrieves the jump targets for this block, 
        excluding any jump targets that are also backedges."""
        acc = []
        for j in self._jump_targets:
            if j not in self.backedges:
                acc.append(j)
        return tuple(acc)

    def declare_backedge(self, target: str) -> "BasicBlock":
        """Declare one of the jump targets as a backedge of this block.
        """
        if target in self.jump_targets:
            assert not self.backedges
            return replace(self, backedges=(target,))
        return self

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        """Replaces jump targets of this block by the given tuple.
        The provided jump targets must be in the same order as their
        intended original replacements.

        Note that replacing jump targets will not replace the backedge
        tuple, so replacement for any jump targets that is declared as
        a backedge needs to be updated separately using replace_backedges"""
        return replace(self, _jump_targets=jump_targets)

    def replace_backedges(self, backedges: Tuple) -> "BasicBlock":
        """Replace back edges of this block by the given tuple.
        The provided back edges must be in the same order as their
        intended original replacements."""
        return replace(self, backedges=backedges)


@dataclass(frozen=True)
class PythonBytecodeBlock(BasicBlock):
    """
        The PythonBytecodeBlock class is a subclass of the BasicBlock class
        and represents a basic block in Python bytecode. It inherits all
        attributes and methods from the BasicBlock class as well as it's
        immutability properties.
   """

    begin: int = None
    """The starting bytecode offset."""

    end: int = None
    """The bytecode offset immediate after the last bytecode of the block."""

    def get_instructions(
        self, bcmap: Dict[int, dis.Instruction]
    ) -> List[dis.Instruction]:
        """
            Retrieves a list of dis.Instruction objects corresponding to
            the instructions within the bytecode block. The bcmap parameter
            is a dictionary mapping bytecode offsets to dis.Instruction
            objects. This method iterates over the bytecode offsets within
            the begin and end range, retrieves the corresponding dis.Instruction
            objects from bcmap, and returns a list of these instructions.
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
class SyntheticBlock(BasicBlock):
    """
        The SyntheticBlock class is a subclass of the BasicBlock class and
        represents a artificially added block in a data flow graph. It serves as
        a base class for other artificially added block types. This class inherits
        all attributes and methods from the BasicBlock class.
    """
    pass

@dataclass(frozen=True)
class SyntheticExit(SyntheticBlock):
    """
        The SyntheticExit class is a subclass of the SyntheticBlock class
        and represents a artificially added exit block in a data flow graph. 
        It is used to denote an exit point in the data flow. This class
        inherits all attributes and methods from the SyntheticBlock and 
        BasicBlock classes.
    """
    pass

@dataclass(frozen=True)
class SyntheticReturn(SyntheticBlock):
    """
        The SyntheticReturn class is a subclass of the SyntheticBlock class 
        and represents a artificially added return block in a data flow graph. It 
        is used to denote a return point in the data flow. This class 
        inherits all attributes and methods from the SyntheticBlock and 
        BasicBlock classes.
    """
    pass

@dataclass(frozen=True)
class SyntheticTail(SyntheticBlock):
    """
        The SyntheticTail class is a subclass of the SyntheticBlock class and 
        represents a artificially added tail block in a data flow graph. It is used 
        to denote a tail call point in the data flow. This class inherits 
        all attributes and methods from the SyntheticBlock and BasicBlock 
        classes.
    """
    pass

@dataclass(frozen=True)
class SyntheticFill(SyntheticBlock):
    """
        The SyntheticFill class is a subclass of the SyntheticBlock class 
        and represents a artificially added fill block in a data flow graph. It is 
        used to denote a fill point in the data flow. This class inherits 
        all attributes and methods from the SyntheticBlock and BasicBlock 
        classes.
   """
    pass

@dataclass(frozen=True)
class SyntheticAssignment(SyntheticBlock):
    """
        The SyntheticAssignment class is a subclass of the SyntheticBlock
        class and represents a artificially added assignment block in a data
        flow graph. It is used to denote a block where variable assignments
        occur. This class inherits all attributes and methods from the
        SyntheticBlock and BasicBlock classes.
    """
    variable_assignment: dict = None
    """
        A dictionary representing the variable assignments
        that occur within the artificially added assignment block. It maps
        the variable name to the value that is is assigned when
        the block is executed.
    """


@dataclass(frozen=True)
class SyntheticBranch(SyntheticBlock):
    variable: str = None
    branch_value_table: dict = None

    def replace_jump_targets(self, jump_targets: Tuple) -> "BasicBlock":
        """Replaces jump targets of this block by the given tuple.
        The provided jump targets must be in the same order as their
        intended original replacements. Additionally also updates the
        branch value table of the branch block.

        Note that replacing jump targets will not replace the backedge
        tuple, so replacement for any jump targets that is declared as
        a backedge needs to be updated separately using replace_backedges"""

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
    """
        The SyntheticHead class is a subclass of the SyntheticBranch 
        class and represents a artificially added head block in a data flow 
        graph. It is used to denote the head of a artificially added branch. 
        This class inherits all attributes and methods from the 
        SyntheticBranch and BasicBlock classes.
    """
    pass


@dataclass(frozen=True)
class SyntheticExitingLatch(SyntheticBranch):
    """
        The SyntheticExitingLatch class is a subclass of the SyntheticBranch 
        class and represents a artificially added exiting latch block in a data 
        flow graph. It is used to denote a artificially added latch block that is 
        also an exit point in the data flow. This class inherits all 
        attributes and methods from the SyntheticBranch and BasicBlock 
        classes.
    """
    pass


@dataclass(frozen=True)
class SyntheticExitBranch(SyntheticBranch):
    """
        The SyntheticExitBranch class is a subclass of the SyntheticBranch 
        class and represents a synthetic exit branch block in a control flow 
        graph. It is used to denote a synthetic branch block that leads to 
        an exit point in the control flow. This class inherits all attributes 
        and methods from the SyntheticBranch and BasicBlock classes.
    """
    pass


@dataclass(frozen=True)
class RegionBlock(BasicBlock):
    """
        The RegionBlock class is a subclass of the BasicBlock class and 
        represents a block within a region in a control flow graph. It 
        extends the BasicBlock class with additional attributes and methods. 
   """
    kind: str = None
    """The kind of region. Can be 'head', 'tail', 'branch',
    'loop' or 'meta' strings"""
    parent_region: "RegionBlock" = None
    """The parent region of this region as per the SCFG."""
    header: str = None
    """The header node of the region"""
    subregion: "SCFG" = None
    """The subgraph as an independent SCFG. Note that in case
    of subregions the exiting node may point to blocks outside
    of the current SCFG object."""
    exiting: str = None
    """The exiting node of the region."""

    def replace_header(self, new_header):
        """Does inplace replacement of the header block."""
        object.__setattr__(self, "header", new_header)

    def replace_exiting(self, new_exiting):
        """Does, inplace replacement of the exiting block."""
        object.__setattr__(self, "exiting", new_exiting)
