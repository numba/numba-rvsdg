import ast
import inspect
import itertools
from typing import Callable, Any, MutableMapping, MutableSequence, cast
import textwrap
from dataclasses import dataclass
from collections import defaultdict

from numba_scfg.core.datastructures.scfg import SCFG
from numba_scfg.core.datastructures.basic_block import (
    PythonASTBlock,
    RegionBlock,
    SyntheticHead,
    SyntheticTail,
    SyntheticFill,
    SyntheticReturn,
    SyntheticAssignment,
    SyntheticExitingLatch,
    SyntheticExitBranch,
)


def unparse_code(
    code: str | list[ast.FunctionDef] | Callable[..., Any],
) -> list[type[ast.AST]]:
    # Convert source code into AST.
    if isinstance(code, str):
        tree = ast.parse(code).body
    elif callable(code):
        tree = ast.parse(textwrap.dedent(inspect.getsource(code))).body
    elif (
        isinstance(code, list)
        and len(code) > 0
        and all([isinstance(i, ast.AST) for i in code])
    ):
        tree = code  # type: ignore
    else:
        msg = "Type: '{type(self.code}}' is not implemented."
        raise NotImplementedError(msg)
    return tree  # type: ignore


class WritableASTBlock:
    """A basic block containing Python AST that can be written to.

    The recursive AST -> CFG algorithm requires a basic block that can be
    written to.

    """

    name: str
    instructions: list[ast.AST]
    jump_targets: list[str]

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
        elif self.is_break():
            self.set_jump_targets(exit_index)
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
            f"{self.instructions}, {self.jump_targets})"
        )


class ASTCFG(dict[str, WritableASTBlock]):
    """A CFG consisting of WritableASTBlocks."""

    unreachable: set[WritableASTBlock]
    empty: set[WritableASTBlock]
    noops: set[type[ast.AST]]

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

    def prune_noops(self) -> set[type[ast.AST]]:
        """Prune no-op instructions from the CFG."""
        noops = set()
        exclude = (ast.Pass, ast.Continue, ast.Break)
        for block in self.values():
            block.instructions = [
                i for i in block.instructions if not isinstance(i, exclude)
            ]
            noops.update(
                [i for i in block.instructions if isinstance(i, exclude)]
            )
        self.noops = noops  # type: ignore
        return noops  # type: ignore

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

    # Prune noop statements and unreachable/empty blocks from the CFG.
    prune: bool
    # The code to be transformed.
    code: str | list[ast.FunctionDef] | Callable[..., Any]
    tree: list[type[ast.AST]]
    # Monotonically increasing block index, starts at 1.
    block_index: int
    # The current block being modified
    current_block: WritableASTBlock
    # Dict mapping block indices as strings to WritableASTBlocks.
    # (This is the data structure to hold the CFG.)
    blocks: ASTCFG
    # Stack for header and exiting block of current loop.
    loop_stack: list[LoopIndices]

    def __init__(
        self,
        code: str | list[ast.FunctionDef] | Callable[..., Any],
        prune: bool = True,
    ) -> None:
        self.prune = prune
        self.code = code
        self.tree = unparse_code(code)
        self.block_index: int = 1  # 0 is reserved for genesis block
        self.bool_op_index = 0  # can have multiple of these per block
        self.blocks = ASTCFG()
        # Initialize first (genesis) block, assume it's named zero.
        # (This also initializes the self.current_block attribute.)
        self.add_block(0)
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
        # Assert that the code handed in was a function, we can only transform
        # functions.
        assert isinstance(self.tree[0], ast.FunctionDef)
        # Run recursive code generation.
        self.codegen(self.tree)
        # Prune if requested.
        if self.prune:
            _ = self.blocks.prune_unreachable()
            _ = self.blocks.prune_noops()
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
                ast.AugAssign,
                ast.Assign,
                ast.Expr,
                ast.Return,
            ),
        ):
            # Node has an expression, must handle it.
            node.value = self.handle_expression(node.value)
            self.current_block.instructions.append(node)
        elif isinstance(
            node,
            (
                ast.Break,
                ast.Continue,
                ast.Pass,
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

    def handle_expression(self, node: Any) -> Any:
        """Recursively handle expression nodes and their subexpressions.
        Returns the processed expression."""

        if isinstance(node, ast.BoolOp):
            # Handle or/and operations.
            if len(node.values) > 2:
                # In this case the bool operation has more than two operands,
                # we need to deconstruct this into a binary tree of bool
                # operations and recursively deal with those. The tail_node
                # contains the tail of the operand list.
                tail_node = ast.BoolOp(node.op, node.values[1:])
                return self.handle_bool_op(
                    ast.BoolOp(
                        self.handle_expression(node.op),
                        [node.values[0], tail_node],
                    )
                )
            elif len(node.values) == 2:
                # Base case, boolean operation has only two operands.
                return self.handle_bool_op(
                    ast.BoolOp(
                        node.op,
                        [self.handle_expression(v) for v in node.values],
                    )
                )
            else:
                raise NotImplementedError("unreachable")
        elif isinstance(node, ast.Compare):
            # Recursively handle left and right sides of comparison.
            node.left = self.handle_expression(node.left)
            node.comparators = [
                self.handle_expression(c) for c in node.comparators
            ]
            return node
        elif isinstance(node, ast.BinOp):
            # Handle binary operations (+, -, *, / etc).
            node.left = self.handle_expression(node.left)
            node.right = self.handle_expression(node.right)
            return node
        elif isinstance(node, ast.Call):
            # Handle function calls.
            node.args = [self.handle_expression(a) for a in node.args]
            return node
        else:
            # Base case: literals, names, etc.
            return node

    def handle_bool_op(self, node: ast.BoolOp) -> ast.Name:
        """Handle boolean operations (and/or).
        Returns an ast.Name representing the result variable."""

        # Create a new temp variable to store the result.
        self.bool_op_index += 1
        result_var = f"__scfg_bool_op_{self.bool_op_index}__"

        # Create an assignment to bin temp variable from above to the left most
        # value in the expression.
        left = self.handle_expression(node.values[0])
        self.current_block.instructions.append(
            ast.Assign(
                targets=[ast.Name(id=result_var, ctx=ast.Store())],
                value=left,
                lineno=0,
            )
        )

        # Handle the or operator.
        if isinstance(node.op, ast.Or):
            # Create blocks for the true and false paths.
            false_block_index = self.block_index
            merge_block_index = self.block_index + 1
            self.block_index += 2

            # Test and jump based on first value.
            self.current_block.instructions.append(
                ast.Name(id=result_var, ctx=ast.Load())
            )
            self.current_block.set_jump_targets(
                merge_block_index, false_block_index
            )

            # False block evaluates second value.
            self.add_block(false_block_index)
            right = self.handle_expression(node.values[1])
            self.current_block.instructions.append(
                ast.Assign(
                    targets=[ast.Name(id=result_var, ctx=ast.Store())],
                    value=right,
                    lineno=0,
                )
            )
            self.current_block.set_jump_targets(merge_block_index)

            # Create merge block
            self.add_block(merge_block_index)

        # Handle the and operator.
        elif isinstance(node.op, ast.And):
            # Create blocks for the true and false paths.
            true_block_index = self.block_index
            merge_block_index = self.block_index + 1
            self.block_index += 2

            # Test and jump based on first value.
            self.current_block.instructions.append(
                ast.Name(id=result_var, ctx=ast.Load())
            )
            self.current_block.set_jump_targets(
                true_block_index, merge_block_index
            )

            # True block evaluates second value.
            self.add_block(true_block_index)
            right = self.handle_expression(node.values[1])
            self.current_block.instructions.append(
                ast.Assign(
                    targets=[ast.Name(id=result_var, ctx=ast.Store())],
                    value=right,
                    lineno=0,
                )
            )
            self.current_block.set_jump_targets(merge_block_index)

            # Create merge block.
            self.add_block(merge_block_index)

        else:
            raise NotImplementedError("unreachable")

        # Return name node referencing our result variable.
        return ast.Name(id=result_var, ctx=ast.Load())

    def handle_function_def(self, node: ast.FunctionDef) -> None:
        """Handle a function definition."""
        # Insert implicit return None, if the function isn't terminated. May
        # end up being an unreachable block if all other paths through the
        # program already call return.
        if not isinstance(node.body[-1], ast.Return):
            node.body.append(ast.Return())
        self.codegen(node.body)

    def handle_if(self, node: ast.If) -> None:
        """Handle if statement."""
        # Preallocate block indices for then, else, and end-if.
        then_index = self.block_index
        else_index = self.block_index + 1
        enif_index = self.block_index + 2
        self.block_index += 3

        # Desugar test expression if needed, may modify current_block.
        test_name = self.handle_expression(node.test)
        # Emit comparison value to current/header block.
        self.current_block.instructions.append(test_name)
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
        # Preallocate header, body, else and exiting indices.
        # (Technically, we could re-use the current block as header if it is
        # still empty. We elect to potentially leave a block empty instead,
        # since there is a pass to prune empty blocks anyway.)
        head_index = self.block_index
        body_index = self.block_index + 1
        exit_index = self.block_index + 2
        else_index = self.block_index + 3
        self.block_index += 4

        self.current_block.set_jump_targets(head_index)
        # And create new header block
        self.add_block(head_index)

        # Desugar test expression if needed, may modify current_block.
        test_name = self.handle_expression(node.test)
        # Emit comparison expression into header.
        self.current_block.instructions.append(test_name)
        # Set the jump targets to be the body and the else branch.
        self.current_block.set_jump_targets(body_index, else_index)

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

        # Create else block.
        self.add_block(else_index)

        # Recurs into the body of the else-branch, again this may modify the
        # current_block.
        self.codegen(node.orelse)

        # Seal current_block.
        self.seal_block(exit_index)

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
        node has the following signature::

            ast.For(target, iter, body, orelse, type_comment)

        Remember also that Python for-loops can have an else-branch, that is
        executed upon regular loop conclusion::

            def function(a: int) -> None:
                c = 0
                for i in range(10):
                    c += i
                    if i == a:
                        i = 420  # set i arbitrarily
                        break    # early exit, break from loop, bypass else
                else:
                    c += 1       # loop conclusion, i.e. not hit break
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
        transformer::

            i = next(iter, "__sentinel__")
            if i != "__sentinel__":
                ...

        Lastly, it is important to also remember that the target variable
        escapes the scope of the for loop::

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

        Consider again the following function::

            def function(a: int) -> None:
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
        using the available block and edge primitives of the CFG::

            def function(a: int) -> None:
                c = 0
                __iterator_1__ = iter(range(10))  ## setup iterator
                i = None                          ## assign target, i
                while True:                       # loop until we break
                    __iter_last_1__ = i           ## backup value of i
                    i = next(__iterator_1__, '__sentinel__')  ## get next i
                    if i != '__sentinel__':       ## regular iteration
                        c += i                    # add to accumulator
                        if i == a:                # check for early exit
                            i = 420               # set i to some wild value
                            break                 # early exit break while True
                    else:                         # for-else clause
                        i == __iter_last_1__      ## restore value of i
                        c += 1                    # execute code in for-else
                        break                     # exit break while True
                return c, i

        The above is actually a full Python source reconstruction. In the
        implementation below, it is only necessary to emit some of the special
        assignments (marked above with a #-prefix above) into the blocks of the
        CFG.  All of the control-flow inside the function will be represented
        by the directed edges of the CFG.

        The first two assignments are for the pre-header:

         *  ``__iterator_1__ = iter(range(10))  ## setup iterator``
         *  ``i = None                          ## assign target, i``

        The next three is for the header, the predicate determines the end of
        the loop.

         *      ``__iter_last_1__ = i           ## backup value of i``
         *      ``i = next(__iterator_1__, '__sentinel__')  ## get next i``
         *      ``if i != '__sentinel__':       ## regular iteration``

         And lastly, one assignment in the for-else clause

         *          ``i == __iter_last_1__      ## restore value of i``

        We modify the pre-header, the header and the else blocks with
        appropriate Python statements in the following implementation. The
        Python code is injected by generating Python source using f-strings and
        then using the ``unparse()`` function of the ``ast`` module to then use
        the 'codegen' method of this transformer to emit the required
        ``ast.AST`` objects into the blocks of the CFG.

        Lastly the important thing to observe is that we can not ignore the
        else clause, since this must contain the reset of the variable i, which
        will have been set to ``__sentinel__``. This reset is required such
        that the target variable ``i`` will escape the scope of the for-loop.

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
        iter_assign = f"__scfg_iterator_{head_index}__"
        last_target_value = f"__scfg_iter_last_{head_index}__"

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
        # should continue.  The '__scfg__sentinel__' is an singleton style
        # marker, so it need not be versioned.

        header_code = textwrap.dedent(
            f"""
            {last_target_value} = {target}
            {target} = next({iter_assign}, "__scfg_sentinel__")
            {target} != "__scfg_sentinel__"
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

        # Emit orelse instructions. Needs to be prefixed with an assignment
        # such that the for loop target can escape the scope of the loop.
        else_code = textwrap.dedent(
            f"""
            {target} = {last_target_value}
        """
        )
        self.codegen(ast.parse(else_code).body)

        # Recurs into the body of the else-branch.
        self.codegen(node.orelse)

        # Seal current block, whatever it may be.
        self.seal_block(exit_index)

        # Create exit block and leave open for modification
        self.add_block(exit_index)

    def render(self) -> None:
        """Render the CFG contained in this transformer as a SCFG.

        Useful for debugging purposes, set a breakpoint and then render to view
        intermediary results.

        """
        self.blocks.to_SCFG().render()


class SCFG2ASTTransformer:

    def transform(
        self, original: ast.FunctionDef, scfg: SCFG
    ) -> ast.FunctionDef:
        body: MutableSequence[ast.AST] = []
        self.region_stack = [scfg.region]
        self.scfg = scfg
        self.loop_cont_counter = 0
        for name, block in scfg.concealed_region_view.items():
            if type(block) is RegionBlock and block.kind == "branch":
                continue
            body.extend(self.codegen(block))
        return ast.FunctionDef(
            name=f"transformed_{original.name}",
            args=original.args,
            body=cast(list[ast.stmt], body),
            decorator_list=original.decorator_list,
            returns=original.returns,
            type_comment=original.type_comment,
            type_params=original.type_params,
            lineno=0,
        )

    def lookup(self, item: Any) -> Any:
        subregion_scfg = self.region_stack[-1].subregion
        parent_region_block = self.region_stack[-1].parent_region
        if item in subregion_scfg:  # type: ignore
            return subregion_scfg[item]  # type: ignore
        else:
            return self.rlookup(parent_region_block, item)  # type: ignore

    def rlookup(self, region_block: RegionBlock, item: Any) -> Any:
        if item in region_block.subregion:  # type: ignore
            return region_block.subregion[item]  # type: ignore
        elif region_block.parent_region is not None:
            return self.rlookup(region_block.parent_region, item)
        else:
            raise KeyError(f"Item {item} not found in subregion or parent")

    def codegen(self, block: Any) -> MutableSequence[ast.AST]:
        if type(block) is PythonASTBlock:
            if len(block.jump_targets) == 2:
                test: ast.expr
                if type(block.tree[-1]) in (ast.Name, ast.Compare):
                    test = cast(ast.expr, block.tree[-1])
                else:
                    test = cast(ast.Expr, block.tree[-1]).value
                body: list[ast.stmt] = cast(
                    list[ast.stmt],
                    self.codegen(self.lookup(block.jump_targets[0])),
                )
                orelse: list[ast.stmt] = cast(
                    list[ast.stmt],
                    self.codegen(self.lookup(block.jump_targets[1])),
                )
                if_node = ast.If(test, body, orelse)
                return block.tree[:-1] + [if_node]
            elif block.fallthrough and type(block.tree[-1]) is ast.Return:
                # The value of the ast.Return could be either None or an
                # ast.AST type. In the case of None, this refers to a plain
                # 'return', which is implicitly 'return None'. So, if it is
                # None, we assign the __scfg_return_value__ an
                # ast.Constant(None) and whatever the ast.AST node is
                # otherwise.
                val = block.tree[-1].value
                return block.tree[:-1] + [
                    ast.Assign(
                        [ast.Name("__scfg_return_value__")],
                        (ast.Constant(None) if val is None else val),
                        lineno=0,
                    )
                ]
            elif block.fallthrough or block.is_exiting:
                return block.tree
            else:
                raise NotImplementedError
        elif type(block) is RegionBlock:
            # We maintain a stack of the current region, in order to allow for
            # random node lookup by name.
            self.region_stack.append(block)

            # This is a custom view that uses the concealed_region_view and
            # additionally filters all branch regions. Essentially, branch
            # regions will be visited by calling codegen recursively from
            # blocks with multiple jump targets and all other regions must be
            # visited linearly.
            def codegen_view() -> list[Any]:
                return list(
                    itertools.chain.from_iterable(
                        self.codegen(b)
                        for b in block.subregion.concealed_region_view.values()  # type: ignore  # noqa
                        if not (type(b) is RegionBlock and b.kind == "branch")
                    )
                )

            if block.kind in ("head", "tail", "branch"):
                rval = codegen_view()
            elif block.kind == "loop":
                # A loop region gives rise to a Python while __scfg_loop_cont__
                # loop. We recursively visit the body. The exiting latch will
                # update __scfg_loop_continue__.
                self.loop_cont_counter += 1
                loop_continue = f"__scfg_loop_cont_{self.loop_cont_counter}__"
                rval = [
                    ast.Assign(
                        [ast.Name(loop_continue)],
                        ast.Constant(True),
                        lineno=0,
                    ),
                    ast.While(
                        test=ast.Name(loop_continue),
                        body=codegen_view(),
                        orelse=[],
                    ),
                ]
            else:
                raise NotImplementedError
            self.region_stack.pop()
            return rval
        elif type(block) is SyntheticAssignment:
            # Synthetic assignments just create Python assignments, one for
            # each variable..
            return [
                ast.Assign([ast.Name(t)], ast.Constant(v), lineno=0)
                for t, v in block.variable_assignment.items()
            ]
        elif type(block) is SyntheticTail:
            # Synthetic tails do nothing.
            return []
        elif type(block) is SyntheticFill:
            # Synthetic fills must have a pass statement to main syntactical
            # correctness of the final program.
            return [ast.Pass()]
        elif type(block) is SyntheticReturn:
            # Synthetic return blocks must re-assigne the return value to a
            # special reserved variable.
            return [ast.Return(ast.Name("__scfg_return_value__"))]
        elif type(block) is SyntheticExitingLatch:
            # The synthetic exiting latch simply assigns the negated value of
            # the exit variable to '__scfg_loop_cont__'.
            assert len(block.jump_targets) == 1
            assert len(block.backedges) == 1
            loop_continue = f"__scfg_loop_cont_{self.loop_cont_counter}__"
            self.loop_cont_counter -= 1
            return [
                ast.Assign(
                    [ast.Name(loop_continue)],
                    ast.UnaryOp(ast.Not(), ast.Name(block.variable)),
                    lineno=0,
                )
            ]
        elif type(block) in (SyntheticExitBranch, SyntheticHead):
            # Both the Synthetic exit branch and the synthetic head contain a
            # branching statement with potentially multiple outgoing branches.
            # This means we must recursively generate an if-cascade in Python,
            # such that all jump targets may be visisted. Looking at the
            # resulting AST, it does appear as though the compilation of the
            # AST to source code will use `elif` statements.

            # Create a reverse lookup from the branch_value_table
            # branch_name --> list of variables that lead there
            reverse = defaultdict(list)
            for (
                variable_value,
                jump_target,
            ) in block.branch_value_table.items():
                reverse[jump_target].append(variable_value)
            # recursive generation of if-cascade

            def if_cascade(
                jump_targets: list[str],
            ) -> MutableSequence[ast.AST]:
                if len(jump_targets) == 1:
                    # base case, final else
                    return self.codegen(self.lookup(jump_targets.pop()))
                else:
                    # otherwise generate if statement for current jump_target
                    current = jump_targets.pop()
                    # compare to all variable values that point to this
                    # jump_target
                    if_test = ast.Compare(
                        left=ast.Name(block.variable),
                        ops=[ast.In()],
                        comparators=[
                            ast.Tuple(
                                elts=[
                                    ast.Constant(i) for i in reverse[current]
                                ],
                                ctx=ast.Load(),
                            )
                        ],
                    )
                    # Create the the if-statement itself, using the test. Do
                    # code-gen for the block that the is being pointed to and
                    # recurse for the rest of the jump_targets.
                    if_node = ast.If(
                        test=if_test,
                        body=cast(
                            list[ast.stmt], self.codegen(self.lookup(current))
                        ),
                        orelse=cast(list[ast.stmt], if_cascade(jump_targets)),
                    )
                    return [if_node]

            # Send in a copy of the jump_targets as this list will be mutated.
            return if_cascade(list(block.jump_targets[::-1]))
        else:
            raise NotImplementedError

        raise NotImplementedError("unreachable")


def AST2SCFG(code: str | list[ast.FunctionDef] | Callable[..., Any]) -> SCFG:
    """Transform Python function into an SCFG."""
    return AST2SCFGTransformer(code).transform_to_SCFG()


def SCFG2AST(
    code: str | list[ast.FunctionDef] | Callable[..., Any], scfg: SCFG
) -> ast.FunctionDef:
    """Transform SCFG with PythonASTBlocks into an AST FunctionDef defined in
    code."""
    original_ast = unparse_code(code)[0]
    return SCFG2ASTTransformer().transform(
        original=original_ast, scfg=scfg  # type: ignore
    )
