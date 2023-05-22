
============================
Datastructures API Reference
============================

BasicBlocks
===========

.. currentmodule:: numba-rvsdg.core.datastructures

.. contents::
   :local:
   :depth: 1


.. class:: BasicBlock

   The BasicBlock class represents an atomic basic block in a data flow graph. 
   Note that the BasicBlock class is defined with the `frozen=True`` parameter,
   making instances of this class immutable. It is a dataclass with the
   following attributes and methods:


   .. attribute:: name: str

      The corresponding name for this block.

   .. attribute:: _jump_targets: Tuple[str] (default: tuple())

      Jump targets (branch destinations) for this block.

   .. attribute:: backedges: Tuple[str] (default: tuple())

      Backedges for this block.

   .. property:: is_exiting: bool

      Indicates whether this block is an exiting block, i.e., it does not have any jump targets.

   .. property:: fallthrough: bool

      Indicates whether this block has a single fallthrough jump target.

   .. property:: jump_targets: Tuple[str]

      Retrieves the jump targets for this block, excluding any jump targets that are also backedges.

   .. method:: replace_backedge(target: str) -> BasicBlock

        Replaces the backedge of this block with the specified target, returning a new BasicBlock instance.

   .. method:: replace_jump_targets(jump_targets: Tuple) -> BasicBlock

        Replaces the jump targets of this block with the specified jump targets, returning a new BasicBlock instance.


.. class:: PythonBytecodeBlock

   The PythonBytecodeBlock class is a subclass of the BasicBlock class
   and represents a basic block in Python bytecode. It inherits all
   attributes and methods from the BasicBlock class as well as it's
   immutability properties. This class has the following attributes
   and methods:

   .. attribute:: begin: int = None

        The starting bytecode offset.

   .. attribute:: end: int = None

        The bytecode offset immediately after the last bytecode of the block.

   .. method:: get_instructions(bcmap: Dict[int, dis.Instruction]) -> List[dis.Instruction]

        Retrieves a list of dis.Instruction objects corresponding to
        the instructions within the bytecode block. The bcmap parameter
        is a dictionary mapping bytecode offsets to dis.Instruction
        objects. This method iterates over the bytecode offsets within
        the begin and end range, retrieves the corresponding dis.Instruction
        objects from bcmap, and returns a list of these instructions.

.. class:: SyntheticBlock()

   The SyntheticBlock class is a subclass of the BasicBlock class and
   represents a artificially added block in a data flow graph. It serves as
   a base class for other artificially added block types. This class inherits
   all attributes and methods from the BasicBlock class.


.. class:: SyntheticExit()

   The SyntheticExit class is a subclass of the SyntheticBlock class
   and represents a artificially added exit block in a data flow graph. 
   It is used to denote an exit point in the data flow. This class
   inherits all attributes and methods from the SyntheticBlock and 
   BasicBlock classes.

.. class:: SyntheticReturn()

   The SyntheticReturn class is a subclass of the SyntheticBlock class 
   and represents a artificially added return block in a data flow graph. It 
   is used to denote a return point in the data flow. This class 
   inherits all attributes and methods from the SyntheticBlock and 
   BasicBlock classes.

.. class:: SyntheticTail()

   The SyntheticTail class is a subclass of the SyntheticBlock class and 
   represents a artificially added tail block in a data flow graph. It is used 
   to denote a tail call point in the data flow. This class inherits 
   all attributes and methods from the SyntheticBlock and BasicBlock 
   classes.

.. class:: SyntheticFill()

   The SyntheticFill class is a subclass of the SyntheticBlock class 
   and represents a artificially added fill block in a data flow graph. It is 
   used to denote a fill point in the data flow. This class inherits 
   all attributes and methods from the SyntheticBlock and BasicBlock 
   classes.

.. class:: SyntheticAssignment()

   The SyntheticAssignment class is a subclass of the SyntheticBlock
   class and represents a artificially added assignment block in a data
   flow graph. It is used to denote a block where variable assignments
   occur. This class inherits all attributes and methods from the
   SyntheticBlock and BasicBlock classes. Additionally, it defines
   the following attribute:

   .. attribute:: variable_assignment: dict = None

      A dictionary representing the variable assignments
      that occur within the artificially added assignment block. It maps
      the variable name to the value that is is assigned when
      the block is executed.

.. class:: SyntheticHead()

   The SyntheticHead class is a subclass of the SyntheticBranch 
   class and represents a artificially added head block in a data flow 
   graph. It is used to denote the head of a artificially added branch. 
   This class inherits all attributes and methods from the 
   SyntheticBranch and BasicBlock classes.

.. class:: SyntheticExitingLatch()

   The SyntheticExitingLatch class is a subclass of the SyntheticBranch 
   class and represents a artificially added exiting latch block in a data 
   flow graph. It is used to denote a artificially added latch block that is 
   also an exit point in the data flow. This class inherits all 
   attributes and methods from the SyntheticBranch and BasicBlock 
   classes.

.. class:: SyntheticExitBranch()

   The SyntheticExitBranch class is a subclass of the SyntheticBranch 
   class and represents a synthetic exit branch block in a control flow 
   graph. It is used to denote a synthetic branch block that leads to 
   an exit point in the control flow. This class inherits all attributes 
   and methods from the SyntheticBranch and BasicBlock classes.

.. class:: RegionBlock()

   The RegionBlock class is a subclass of the BasicBlock class and 
   represents a block within a region in a control flow graph. It 
   extends the BasicBlock class with additional attributes and methods. 
   This class is defined as a dataclass with the following attributes:

   .. attribute:: kind: str = None

      The kind of the region block.

   .. attribute:: headers: Dict[str, BasicBlock] = None

      A dictionary representing the headers of the region.
      The keys are the names of the headers, and the values 
      are the corresponding BasicBlock objects.

   .. attribute:: subregion: "SCFG" = None

      The subgraph representing the region, excluding the headers.

   .. attribute:: exiting: str = None

      The name of the exiting node in the region.

   .. method:: get_full_graph()

      Retrieves the full graph of the region, including the subregion 
      and the headers. It returns a ChainMap object that combines the 
      subregion's graph and the headers.

ByteFlow
========

.. currentmodule:: numba-rvsdg.core.datastructures

.. contents::
   :local:
   :depth: 1

.. class:: ByteFlow

   The ByteFlow class represents the flow of bytes in a bytecode and its 
   corresponding structured control flow graph (SCFG). It is defined as 
   a dataclass with the following attributes:

   .. attribute:: bc: dis.Bytecode

      The dis.Bytecode object representing the bytecode.

   .. attribute:: scfg: "SCFG"

      The structured control flow graph (SCFG) representing the control flow of 
      the bytecode.

   .. method:: from_bytecode(code) -> ByteFlow

      Creates a ByteFlow object from the given code, which is the bytecode. 
      This method uses dis.Bytecode to parse the bytecode, builds the basic blocks 
      and flow information from it, and returns a ByteFlow object with the 
      bytecode and the SCFG.

   .. method:: _join_returns()

      Creates a deep copy of the SCFG and performs the operation to join
       return points within the control flow. It returns a new ByteFlow 
       object with the updated SCFG.

   .. method:: _restructure_loop()

      Creates a deep copy of the SCFG and performs the operation to 
      restructure loop constructs within the control flow. It applies 
      the restructuring operation to both the main SCFG and any 
      subregions within it. It returns a new ByteFlow object with 
      the updated SCFG.

   .. method:: _restructure_branch()

      Creates a deep copy of the SCFG and performs the operation to 
      restructure branch constructs within the control flow. It applies 
      the restructuring operation to both the main SCFG and any 
      subregions within it. It returns a new ByteFlow object with 
      the updated SCFG.

   .. method:: restructure()

      Creates a deep copy of the SCFG and applies a series of 
      restructuring operations to it. The operations include 
      joining return points, restructuring loop constructs, and 
      restructuring branch constructs. It returns a new ByteFlow 
      object with the updated SCFG.


FlowInfo
========

.. currentmodule:: numba-rvsdg.core.datastructures

.. contents::
   :local:
   :depth: 1

.. class:: FlowInfo

   The FlowInfo class is responsible for converting bytecode into a 
   ByteFlow object, which represents the control flow graph (CFG). 
   It is defined as a dataclass with the following attributes:

   .. attribute:: block_offsets: Set[int] = field(default_factory=set)

      A set that marks the starting offsets of basic blocks in the bytecode.

   .. attribute:: jump_insts: Dict[int, Tuple[int, ...]] = field(default_factory=dict)

      A dictionary that contains jump instructions and their target offsets.

   .. attribute:: last_offset: int = field(default=0)

      The offset of the last bytecode instruction.

   .. method:: _add_jump_inst(self, offset: int, targets: Sequence[int])

      Internal method to add a jump instruction to the FlowInfo. It adds the 
      target offsets of the jump instruction to the block_offsets set and 
      updates the jump_insts dictionary.

   .. method:: from_bytecode(bc: dis.Bytecode) -> "FlowInfo"

      Static method that builds the control-flow information from the given 
      dis.Bytecode object bc. It analyzes the bytecode instructions, marks 
      the start of basic blocks, and records jump instructions and their 
      target offsets. It returns a FlowInfo object.

   .. method:: build_basicblocks(self: "FlowInfo", end_offset=None) -> "SCFG"

      Builds a graph of basic blocks (SCFG) based on the flow information. 
      It creates a structured control flow graph (SCFG) object, assigns 
      names to the blocks, and defines the block boundaries, jump targets, 
      and backedges. It returns an SCFG object representing the control 
      flow graph.


SCFG classes
============

.. currentmodule:: numba-rvsdg.core.datastructures

.. contents::
   :local:
   :depth: 1

.. class:: NameGenerator

   The NameGenerator class is responsible for generating unique names 
   for blocks, regions, and variables within the control flow graph. 
   It is defined as a dataclass with the following attributes:

   .. attribute:: kinds: dict[str, int] = field(default_factory=dict)

      A dictionary that keeps track of the current index for each kind of name.

   .. method:: new_block_name(self, kind: str) -> str

      Generates a new unique name for a block of the specified kind. It checks 
      if the kind already exists in the kinds dictionary and increments the 
      index if it does. It returns the generated name.

   .. method:: new_region_name(self, kind: str) -> str

      Generates a new unique name for a region of the specified kind. It 
      follows the same logic as new_block_name() but uses the suffix 
      "region" in the generated name.

   .. method:: new_var_name(self, kind: str) -> str

      Generates a new unique name for a variable of the specified kind. 
      It follows the same logic as new_block_name() but uses the suffix 
      "var" in the generated name.


.. class:: SCFG

   The SCFG class represents a map of names to blocks within the control flow graph. It is defined as a dataclass with the following attributes:

   .. attribute:: graph: Dict[str, BasicBlock] = field(default_factory=dict)

      A dictionary that maps names to corresponding BasicBlock objects within the control flow graph.

   .. attribute:: name_gen: NameGenerator = field(default_factory=NameGenerator, compare=False)

      A NameGenerator object that provides unique names for blocks, regions, and variables.

   .. method:: __getitem__(self, index)

      Allows accessing a block from the graph dictionary using the index notation.

   .. method:: __contains__(self, index)

      Checks if the given index exists in the graph dictionary.

   .. method:: __iter__(self)

      Returns an iterator that yields the names and corresponding blocks in the control flow graph. It follows a breadth-first search traversal starting from the head block.

   .. method:: concealed_region_view: ConcealedRegionView

      A property that returns a ConcealedRegionView object, representing a concealed view of the control flow graph.

   .. method:: exclude_blocks(self, exclude_blocks: Set[str]) -> Iterator[str]

      Returns an iterator over all nodes (blocks) in the control flow graph that are not present in the exclude_blocks set. It filters out the excluded blocks and yields the remaining blocks.

   .. method:: find_head(self) -> str

      Finds the head block of the control flow graph. Assuming the CFG is closed, this method identifies the block that no other blocks are pointing to. It returns the name of the head block.

   .. method:: compute_scc(self) -> List[Set[str]]

      Computes the strongly connected components (SCC) of the control flow graph using the scc function from the numba_rvsdg.networkx_vendored.scc module. It returns a list of sets, where each set represents an SCC in the graph. SCCs are useful for detecting loops in the graph.

   .. method:: compute_scc_subgraph(self, subgraph) -> List[Set[str]]

      Computes the strongly connected components (SCC) within a given subgraph of the control flow graph. The subgraph parameter is a set of block names that define the subgraph. This method excludes nodes outside of the subgraph when computing the SCCs. It returns a list of sets, where each set represents an SCC within the subgraph.

   .. method:: find_headers_and_entries(self, subgraph: Set[str]) -> Tuple[Set[str], Set[str]]

      Finds the headers and entries within a given subgraph of the control flow graph. Entries are blocks outside the subgraph that have an edge pointing to the subgraph headers. Headers are blocks that are part of the strongly connected subset (SCC) within the subgraph and have incoming edges from outside the subgraph. Entries point to headers, and headers are pointed to by entries. This method returns a tuple containing two sets: the headers and the entries.

   .. method:: find_exiting_and_exits(self, subgraph: Set[str]) -> Tuple[Set[str], Set[str]]

      Finds the exiting and exit blocks in a given subgraph of the control flow graph. Exiting blocks are blocks inside the subgraph that have edges to blocks outside of the subgraph. Exit blocks are blocks outside the subgraph that have incoming edges from within the subgraph. Exiting blocks point to exit blocks, and exit blocks are pointed to by exiting blocks. This method returns a tuple containing two sets: the exiting blocks and the exit blocks.

   .. method:: is_reachable_dfs(self, begin: str, end: str) -> bool

      Checks if the end block is reachable from the begin block in the control flow graph. It performs a depth-first search (DFS) traversal from the begin block, following the edges of the graph. Returns True if the end block is reachable, and False otherwise.

   .. method:: add_block(self, basicblock: BasicBlock)

      Adds a BasicBlock object to the control flow graph. The basicblock parameter represents the block to be added.

   .. method:: remove_blocks(self, names: Set[str])

      Removes blocks from the control flow graph based on the given set of block names (names). It deletes the corresponding entries from the graph attribute.

   .. method:: _insert_block(self, new_name: str, predecessors: Set[str], successors: Set[str], block_type: SyntheticBlock)

      Inserts a new synthetic block into the control flow graph. This method is used internally by other methods to perform block insertion operations.

   .. method:: insert_SyntheticExit(self, new_name: str, predecessors: Set[str], successors: Set[str])

      Inserts a synthetic exit block into the control flow graph. The new_name parameter specifies the name of the new block, predecessors is a set of block names representing the predecessors of the new block, and successors is a set of block names representing the successors of the new block.

   .. method:: insert_SyntheticTail(self, new_name: str, predecessors: Set[str], successors: Set[str])

      Inserts a synthetic tail block into the control flow graph. The new_name parameter specifies the name of the new block, predecessors is a set of block names representing the predecessors of the new block, and successors is a set of block names representing the successors of the new block.

   .. method:: insert_SyntheticReturn(self, new_name: str, predecessors: Set[str], successors: Set[str])

      Inserts a synthetic return block into the control flow graph. The new_name parameter specifies the name of the new block, predecessors is a set of block names representing the predecessors of the new block, and successors is a set of block names representing the successors of the new block.

   .. method:: insert_SyntheticFill(self, new_name: str, predecessors: Set[str], successors: Set[str])

      Inserts a synthetic fill block into the control flow graph. The new_name parameter specifies the name of the new block, predecessors is a set of block names representing the predecessors of the new block, and successors is a set of block names representing the successors of the new block.

   .. method:: insert_block_and_control_blocks(self, new_name: str, predecessors: Set[str], successors: Set[str])

      Inserts a new block along with control blocks into the control flow graph. This method is used for branching assignments.

   .. method:: join_returns(self)

      Closes the CFG by ensuring that it has a unique entry and exit node. A closed CFG is defined as a CFG with an entry node that has no predecessors and an exit node that has no successors. This method identifies nodes containing return statements and closes the CFG if there are multiple return nodes.

   .. method:: join_tails_and_exits(self, tails: Set[str], exits: Set[str])

       Joins the tails and exits of the CFG. The method takes a set of tail node names (tails) and a set of exit node names (exits) as parameters. It handles different cases based on the number of tails and exits:

   .. method:: bcmap_from_bytecode(bc: dis.Bytecode)

      Static method that creates a bytecode map from a dis.Bytecode object. The method takes a dis.Bytecode object as a parameter and returns a dictionary that maps bytecode offsets to instruction objects.

   .. method:: from_yaml(yaml_string)

      Static method that creates an SCFG object from a YAML string representation. The method takes a YAML string as a parameter and returns an SCFG object and a dictionary of block names.

   .. method:: from_dict(graph_dict: dict) 

       Static method that creates an SCFG object from a dictionary representation. The method takes a dictionary (graph_dict) representing the control flow graph and returns an SCFG object and a dictionary of block names. The input dictionary should have block indices as keys and dictionaries of block attributes as values.

   .. method:: to_yaml(self) 

      Converts the SCFG object to a YAML string representation. The method returns a YAML string representing the control flow graph. It iterates over the graph dictionary and generates YAML entries for each block, including jump targets and backedges.

   .. method:: to_dict(self) 

      Converts the SCFG object to a dictionary representation. The method returns a dictionary representing the control flow graph. It iterates over the graph dictionary and generates a dictionary entry for each block, including jump targets and backedges if present.


.. class:: AbstractGraphView

   The AbstractGraphView class is a subclass of Mapping and serves 
   as an abstract base class for graph views. It defines the basic 
   interface for accessing and iterating over elements in a graph 
   view. This class cannot be instantiated directly and requires 
   subclasses to implement the abstract methods. The class is 
   defined as follows:

   .. method:: __getitem__(self, item)

      Abstract method that should be implemented in subclasses. 
      It retrieves the value associated with the given key in 
      the graph view.

   .. method:: __iter__(self)

      Abstract method that should be implemented in subclasses. 
      It returns an iterator over the keys in the graph view.

   .. method:: __len__(self)

      Abstract method that should be implemented in subclasses. 
      It returns the number of elements in the graph view.


.. class:: ConcealedRegionView

   The ConcealedRegionView class is a subclass of AbstractGraphView and represents a view of a control flow graph where regions are "concealed" and treated as a single block. It provides methods to access and iterate over blocks or regions in the concealed view. The class is defined as follows:

   .. attribute:: scfg: SCFG

      The control flow graph (SCFG) that the concealed region view is based on.

   .. method:: __init__(self, scfg)

      Initializes the ConcealedRegionView with the given control flow graph (SCFG).

   .. method:: __getitem__(self, item)

      Retrieves the value associated with the given key in the concealed region view. It delegates the operation to the underlying control flow graph (SCFG).

   .. method:: __iter__(self)

      Returns an iterator over blocks or regions in the concealed view. It calls the region_view_iterator() method to perform the iteration.

   .. method:: region_view_iterator(self, head: str = None) -> Iterator[str]

      An iterator that traverses the concealed view of the control flow graph. It starts from the specified head block (or region) and yields the names of blocks or regions in the concealed view. If head is not provided, it automatically discovers the head block. This iterator takes into account the concealed nature of regions and presents them as a single fall-through block.

   .. method:: __len__(self)

      Returns the number of elements in the concealed region view. It delegates the operation to the underlying control flow graph (SCFG).

