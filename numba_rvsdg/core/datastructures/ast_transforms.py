import ast
import inspect
from typing import Callable, Any, MutableMapping
import textwrap
from dataclasses import dataclass

from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import PythonASTBlock


class WritableASTBlock:
    """A basic block containing Python AST that can be written to.

    The recursive AST -> CFG algorithm requires a basic block that can be
    written to.

    """

    def __init__(
        self,
        name: str,
        instructions: list[ast.AST] | None = None,
        jump_targets: list[str] | None = None,
    ) -> None:
        self.name = name
        self.instructions: list[ast.AST] = (
            [] if instructions is None else instructions
        )
        self.jump_targets: list[str] = (
            [] if jump_targets is None else jump_targets
        )

    def set_jump_targets(self, *indices: int) -> None:
        """Set jump targets for the block."""
        self.jump_targets = [str(a) for a in indices]

    def is_instruction(self, instruction: type[ast.AST]) -> bool:
        """Check if the last instruction is of a certain type."""
        return len(self.instructions) > 0 and isinstance(
            self.instructions[-1], instruction
        )

    def is_return(self) -> bool:
        """Check if the last instruction is a return statement."""
        return self.is_instruction(ast.Return)

    def is_break(self) -> bool:
        """Check if the last instruction is a break statement."""
        return self.is_instruction(ast.Break)

    def is_continue(self) -> bool:
        """Check if the last instruction is a continue statement."""
        return self.is_instruction(ast.Continue)

    def seal_outside_loop(self, index: int) -> None:
        """Seal the block by setting the jump targets based on the last
        instruction.
        """
        if self.is_return():
            pass
        else:
            self.set_jump_targets(index)

    def seal_inside_loop(
        self, head_index: int, exit_index: int, default_index: int
    ) -> None:
        """Seal the block by setting the jump targets based on the last
        instruction and taking into account that this block is nested in a
        loop.
        """
        if self.is_continue():
            self.set_jump_targets(head_index)
            self.instructions.pop()
        elif self.is_break():
            self.set_jump_targets(exit_index)
            self.instructions.pop()
        elif self.is_return():
            pass
        else:
            self.set_jump_targets(default_index)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "instructions": [ast.unparse(n) for n in self.instructions],
            "jump_targets": self.jump_targets,
        }

    def __repr__(self) -> str:
        return (
            f"WritableASTBlock({self.name}, "
            "{self.instructions}, {self.jump_targets})"
        )


class ASTCFG(dict[str, WritableASTBlock]):
    """A CFG consisting of WritableASTBlocks."""

    def convert_blocks(self) -> MutableMapping[str, Any]:
        """Convert WritableASTBlocks to PythonASTBlocks."""
        return {
            v.name: PythonASTBlock(
                v.name,
                tree=v.instructions,
                _jump_targets=tuple(v.jump_targets),
            )
            for v in self.values()
        }

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Convert ASTCFG to simple dict based data structure."""
        return {k: v.to_dict() for (k, v) in self.items()}

    def to_SCFG(self) -> SCFG:
        """Convert ASTCFG to SCFG"""
        return SCFG(graph=self.convert_blocks())

    def prune_unreachable(self) -> set[WritableASTBlock]:
        """Prune unreachable blocks from the CFG."""
        # Assume that the entry block is named zero (0).
        to_visit, reachable, unreachable = set("0"), set(), set()
        # Visit all reachable blocks.
        while to_visit:
            block = to_visit.pop()
            if block not in reachable:
                # Add block to reachable set.
                reachable.add(block)
                # Update to_visit with jump targets of the block.
                to_visit.update(self[block].jump_targets)
        # Remove unreachable blocks.
        for block in list(self.keys()):
            if block not in reachable:
                unreachable.add(self.pop(block))
        self.unreachable = unreachable
        return unreachable

    def prune_empty(self) -> set[WritableASTBlock]:
        """Prune empty blocks from the CFG."""
        empty = set()
        for name, block in list(self.items()):
            if not block.instructions:
                empty.add(self.pop(name))
                # Empty blocks can only have a single jump target.
                it = block.jump_targets[0]
                # Iterate over the blocks looking for blocks that point to the
                # removed block. Then rewire the jump_targets accordingly.
                for b in list(self.values()):
                    if len(b.jump_targets) == 0:
                        continue
                    elif len(b.jump_targets) == 1:
                        if b.jump_targets[0] == name:
                            b.jump_targets[0] = it
                    elif len(b.jump_targets) == 2:
                        if b.jump_targets[0] == name:
                            b.jump_targets[0] = it
                        elif b.jump_targets[1] == name:
                            b.jump_targets[1] = it
        self.empty = empty
        return empty


@dataclass(frozen=True)
class LoopIndices:
    """Structure to hold the head and exit block indices of a loop."""

    head: int
    exit: int


class AST2SCFGTransformer:
    """AST2SCFGTransformer

    The AST2SCFGTransformer class is responsible for transforming code in the
    form of a Python Abstract Syntax Tree (AST) into CFG/SCFG.

    """

    def __init__(self, code: Callable[..., Any], prune: bool = True) -> None:
        # Prune empty and unreachable blocks from the CFG.
        self.prune: bool = prune
        # Save the code for transformation.
        self.code: Callable[..., Any] = code
        # Monotonically increasing block index, 0 is reserved for genesis.
        self.block_index: int = 1
        # Dict mapping block indices as strings to WritableASTBlocks.
        # (This is the data structure to hold the CFG.)
        self.blocks: ASTCFG = ASTCFG()
        # Initialize first (genesis) block, assume it's named zero.
        # (This also initializes the self.current_block attribute.)
        self.add_block(0)
        # Stack for header and exiting block of current loop.
        self.loop_stack: list[LoopIndices] = []

    def transform_to_ASTCFG(self) -> ASTCFG:
        """Generate ASTCFG from Python function."""
        self.transform()
        return self.blocks

    def transform_to_SCFG(self) -> SCFG:
        """Generate SCFG from Python function."""
        self.transform()
        return self.blocks.to_SCFG()

    def add_block(self, index: int) -> None:
        """Create block, add to CFG and set as current_block."""
        self.blocks[str(index)] = self.current_block = WritableASTBlock(
            name=str(index)
        )

    def seal_block(self, default_index: int) -> None:
        """Seal the current block by setting the jump_targets."""
        if self.loop_stack:
            self.current_block.seal_inside_loop(
                self.loop_stack[-1].head,
                self.loop_stack[-1].exit,
                default_index,
            )
        else:
            self.current_block.seal_outside_loop(default_index)

    def transform(self) -> None:
        """Transform Python function stored as self.code."""
        # Convert source code into AST.
        tree = ast.parse(textwrap.dedent(inspect.getsource(self.code))).body
        # Assert that the code handed in was a function, we can only transform
        # functions.
        assert isinstance(tree[0], ast.FunctionDef)
        # Run recursive code generation.
        self.codegen(tree)
        # Prune if requested.
        if self.prune:
            _ = self.blocks.prune_unreachable()
            _ = self.blocks.prune_empty()

    def codegen(self, tree: list[type[ast.AST]] | list[ast.stmt]) -> None:
        """Recursively transform from a list of AST nodes.

        The function is called 'codegen' as it generates an intermediary
        representation (IR) from an abstract syntax tree (AST). The name was
        chosen to honour the compiler writing tradition, where this type of
        recursive function is commonly called 'codegen'.

        """
        for node in tree:
            self.handle_ast_node(node)

    def handle_ast_node(self, node: type[ast.AST] | ast.stmt) -> None:
        """Dispatch an AST node to handle."""
        if isinstance(node, ast.FunctionDef):
            self.handle_function_def(node)
        elif isinstance(
            node,
            (
                ast.Assign,
                ast.AugAssign,
                ast.Expr,
                ast.Return,
                ast.Break,
                ast.Continue,
            ),
        ):
            self.current_block.instructions.append(node)
        elif isinstance(node, ast.If):
            self.handle_if(node)
        elif isinstance(node, ast.While):
            self.handle_while(node)
        elif isinstance(node, ast.For):
            self.handle_for(node)
        else:
            raise NotImplementedError(f"Node type {node} not implemented")

    def handle_function_def(self, node: ast.FunctionDef) -> None:
        """Handle a function definition."""
        # Insert implicit return None, if the function isn't terminated. May
        # end up being an unreachable block if all other paths through the
        # program already call return.
        if not isinstance(node.body[-1], ast.Return):
            node.body.append(ast.Return(None))
        self.codegen(node.body)

    def handle_if(self, node: ast.If) -> None:
        """Handle if statement."""
        # Preallocate block indices for then, else, and end-if.
        then_index = self.block_index
        else_index = self.block_index + 1
        enif_index = self.block_index + 2
        self.block_index += 3

        # Emit comparison value to current/header block.
        self.current_block.instructions.append(node.test)
        # Setup jump targets for current/header block.
        self.current_block.set_jump_targets(then_index, else_index)

        # Create a new block for the then branch.
        self.add_block(then_index)
        # Recursively transform then branch (this may alter the current_block).
        self.codegen(node.body)
        # After recursion, current_block may need a jump target.
        self.seal_block(enif_index)

        # Create a new block for the else branch.
        self.add_block(else_index)
        # Recursively transform then branch (this may alter the current_block).
        self.codegen(node.orelse)
        # After recursion, current_block may need a jump target.
        self.seal_block(enif_index)

        # Create a new block and assign it to the be the current_block, this
        # will hold the end-if statements if any exist. We leave 'open' for
        # modification.
        self.add_block(enif_index)

    def handle_while(self, node: ast.While) -> None:
        """Handle while statement."""
        # If the current block already has instructions, we need a new block as
        # header. Otherwise just re-use the current_block. This happens
        # when the previous statement was an if-statement with an empty
        # endif_block, for example. This is possible because the Python
        # while-loop does not need to modify it's preheader.
        if self.current_block.instructions:
            # Preallocate header, body and exiting indices.
            head_index = self.block_index
            body_index = self.block_index + 1
            exit_index = self.block_index + 2
            self.block_index += 3

            self.current_block.set_jump_targets(head_index)
            # And create new header block
            self.add_block(head_index)
        else:  # reuse existing current_block
            # Preallocate body and exiting indices.
            head_index = int(self.current_block.name)
            body_index = self.block_index
            exit_index = self.block_index + 1
            self.block_index += 2

        # Emit comparison expression into header.
        self.current_block.instructions.append(node.test)
        # Set the jump targets to be the body and the exiting latch.
        self.current_block.set_jump_targets(body_index, exit_index)

        # Create body block.
        self.add_block(body_index)

        # Push to loop stack for recursion.
        self.loop_stack.append(LoopIndices(head_index, exit_index))

        # Recurs into the body of the while statement. (This may modify
        # current_block).
        self.codegen(node.body)
        # After recursion, seal current_block. This sets the jump targets based
        # on the last instruction in the current_block.
        self.seal_block(head_index)

        # Pop values from loop stack post recursion.
        loop_indices = self.loop_stack.pop()
        assert (
            loop_indices.head == head_index and loop_indices.exit == exit_index
        )

        # Create exit block and leave open for modifictaion.
        self.add_block(exit_index)

    def handle_for(self, node: ast.For) -> None:
        """Handle for statement.

        The Python 'for' statement needs to be decomposed into a series of
        equivalent Python statements, since the semantics of the statement can
        not be represented in the control flow graph (CFG) formalism of blocks
        with directed edges. We note that the for-loop in Python is effectively
        syntactic sugar for a generalised c-style while-loop. To our advantage,
        this while-loop can indeed be represented using the blocks and directed
        edges of the CFG formalism and allows us to transform the Python
        for-loop construct. This docstring explains the decomposition
        from for- into while-loop.

        Remember that the for-loop has a target variable that will be assigned,
        an iterator to iterate over, a loop body and an else clause. The AST
        node has the following signature:

            ast.For(target, iter, body, orelse, type_comment)

        Remember also that Python for-loops can have an else-branch, that is
        executed upon regular loop conclusion.

        def function(a: int) -> None
            c = 0
            for i in range(10):
                c += i
                if i == a:
                    i = 420  # set i arbitrarily
                    break    # early exit, break from loop, bypass else-branch
            else:
                c += 1       # loop conclusion, i.e. we have not hit the  break
        return c, i

        So, effectively, to decompose the for-loop, we need to setup the
        iterator by calling 'iter(iter)' and assign it to a variable,
        initialize the target variable to be None and then check if the
        iterator has a next value. If it does, we need to assign that value to
        the target variable, enter the body and then check the iterator again
        and again and again.. until there are no items left, at which point we
        execute the else-branch.

        The Python for-loop usually waits for the iterator to raise a
        StopIteration exception to determine when the iteration has concluded.
        However, it is possible to use the 'next()' method with a second
        argument to avoid exception handling here. We do this so we don't need
        to rely on being able to transform exceptions as part of this
        transformer.

            i = next(iter, "__sentinel__")
            if i != "__sentinel__":
                ...

        Lastly, it is important to also remember that the target variable
        escapes the scope of the for loop:

            >>> for i in range(1):
            ...     print("hello loop")
            ...
            hello loop
            >>> i
            0
            >>>

        So, to summarize: we want to decompose a Python for loop into a while
        loop with some assignments and he target variable must escape the
        scope.

        Consider again the following function:

        def function(a: int) -> None
            c = 0
            for i in range(10):
                c += i
                if i == a:
                    i = 420
                    break
            else:
                c += 1
            return c, i

        This will be decomposed as the following construct that can be encoded
        using the available block and edge primitives of the CFG.

        def function(a: int) -> None
            c = 0
         *  __iterator_1__ = iter(range(10))  # setup iterator
         *  i = None                          # assign target, in this case i
            while True:                       # loop until we break
         *      __iter_last_1__ = i           # backup value of i
         *      i = next(__iterator_1__, '__sentinel__')  # get next i
         *      if i != '__sentinel__':       # regular iteration
                    c += i                    # add to accumulator
                    if i == a:                # check for early exit
                        i = 420               # set i to some wild value
                        break                 # early exit break while True
                else:                         # for-else clause
         *          i == __iter_last_1__      # restore value of i
                    c += 1                    # execute code in for-else clause
                    break                     # regular exit break while True
            return c, i

        The above is actually a full Python source reconstruction. In the
        implementation below, it is only necessary to emit some of the special
        assignments (marked above with a *-prefix above) into the blocks of the
        CFG.  All of the control-flow inside the function will be represented
        by the directed edges of the CFG.

        The first two assignments are for the pre-header:

         *  __iterator_1__ = iter(range(10))  # setup iterator
         *  i = None                          # assign target, in this case i

        The next three is for the header, the predicate determines the end of
        the loop.

         *      __iter_last_1__ = i           # backup value of i
         *      i = next(__iterator_1__, '__sentinel__')  # get next i
         *      if i != '__sentinel__':       # regular iteration

         And lastly, one assignment in the for-else clause

         *          i == __iter_last_1__      # restore value of i

        We modify the pre-header, the header and the else blocks with
        appropriate Python statements in the following implementation. The
        Python code is injected by generating Python source using f-strings and
        then using the 'unparse()' function of the 'ast' module to then use the
        'codegen' method of this transformer to emit the required 'ast.AST'
        objects into the blocks of the CFG.

        Lastly the important thing to observe is that we can not ignore the
        else clause, since this must contain the reset of the variable i, which
        will have been set to '__sentinel__'. This reset is required such that
        the target variable 'i' will escape the scope of the for-loop.

        """
        # Preallocate indices for header, body, else, and exiting blocks.
        head_index = self.block_index
        body_index = self.block_index + 1
        else_index = self.block_index + 2
        exit_index = self.block_index + 3
        self.block_index += 4

        # Assign the components of the for-loop to variables. These variables
        # are versioned using the index of the loop header so that scopes can
        # be nested. While this is strictly required for the 'iter_setup' it is
        # technically optional for the 'last_target_value'... But, we version
        # it too so that the two can easily be matched when visually inspecting
        # the CFG.
        target = ast.unparse(node.target)
        iter_setup = ast.unparse(node.iter)
        iter_assign = f"__iterator_{head_index}__"
        last_target_value = f"__iter_last_{head_index}__"

        # Emit iterator setup to pre-header.
        preheader_code = textwrap.dedent(
            f"""
            {iter_assign} = iter({iter_setup})
            {target} = None
        """
        )
        self.codegen(ast.parse(preheader_code).body)

        # Point the current_block to header block.
        self.current_block.set_jump_targets(head_index)
        # And create new header block.
        self.add_block(head_index)

        # Emit header instructions. This first makes a backup of the iteration
        # target and then checks if the iterator is exhausted and if the loop
        # should continue.  The '__sentinel__' is an singleton style marker, so
        # it need not be versioned.

        header_code = textwrap.dedent(
            f"""
            {last_target_value} = {target}
            {target} = next({iter_assign}, "__sentinel__")
            {target} != "__sentinel__"
        """
        )
        self.codegen(ast.parse(header_code).body)
        # Set the jump targets to be the body and the else block.
        self.current_block.set_jump_targets(body_index, else_index)

        # Create body block.
        self.add_block(body_index)

        # Setup loop stack for recursion.
        self.loop_stack.append(LoopIndices(head_index, exit_index))

        # Recurs into the loop body (this may modify current_block).
        self.codegen(node.body)
        # After recursion, seal current block.
        self.seal_block(head_index)

        # Pop values from loop stack post recursion.
        loop_indices = self.loop_stack.pop()
        assert (
            loop_indices.head == head_index and loop_indices.exit == exit_index
        )

        # Create else block.
        self.add_block(else_index)
        self.current_block.set_jump_targets(exit_index)

        # Emit orelse instructions. Needs to be prefixed with an assignment
        # such that the for loop target can escape the scope of the loop.
        else_code = textwrap.dedent(
            f"""
            {target} = {last_target_value}
        """
        )
        self.codegen(ast.parse(else_code).body)
        self.codegen(node.orelse)

        # Create exit block.
        self.add_block(exit_index)

    def render(self) -> None:
        """Render the CFG contained in this transformer as a SCFG.

        Useful for debugging purposes, set a breakpoint and then render to view
        intermediary results.

        """
        self.blocks.to_SCFG().render()


def AST2SCFG(code: Callable[..., Any]) -> SCFG:
    """Transform Python function into an SCFG."""
    return AST2SCFGTransformer(code).transform_to_SCFG()


def SCFG2AST(scfg: SCFG) -> ast.FunctionDef:  # type: ignore
    """Transform SCFG with PythonASTBlocks into an AST FunctionDef."""
    # TODO
