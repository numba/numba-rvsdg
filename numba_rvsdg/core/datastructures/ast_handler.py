import ast
import inspect
from typing import Callable
from collections import deque


from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.core.datastructures.basic_block import PythonASTBlock


class ASTHandler:
    """ASTHandler class.

    The ASTHandler class is responsible for converting code in the form of a
    Python Abstract Syntax Tree (ast) into CFG/SCFG.

    """

    def __init__(self, code: Callable) -> None:
        self.code = code
        self.block_index = 0
        self.look_ahead_index = 0
        self.blocks = {}
        self.current_block = []

    def process(self) -> SCFG:
        """Create an SCFG from a Python function. """
        tree = ast.parse(inspect.getsource(self.code))
        self.queue = deque(tree.body)
        while self.queue:
            print(self.queue, self.blocks, self.current_block, self.block_index)
            #breakpoint()
            self.handle_ast_node(self.queue.popleft())

        return SCFG(graph=self.blocks)

    def handle_ast_node(self, node: ast.AST) -> None:
        """Handle an AST node. """
        if isinstance(node, ast.FunctionDef):
            self.handle_function_def(node)
        elif isinstance(node, ast.Assign):
            self.handle_assign(node)
        elif isinstance(node, ast.Expr):
            self.handle_expr(node)
        elif isinstance(node, ast.Return):
            self.handle_return(node)
        elif isinstance(node, ast.If):
            self.handle_if(node)
        elif isinstance(node, ast.While):
            self.handle_while(node)
        elif isinstance(node, ast.For):
            self.handle_for(node)
        elif isinstance(node, str):
            if node == "ENDFOR":
                self.new_block()
            elif node.startswith("ENDIF"):
                index = int(node[5:])
                self.new_block(index)
        else:
            raise NotImplementedError(f"Node type {node} not implemented")

    def new_block(self, index: int) -> None:
        """Create a new block. """
        self.blocks[str(index)] = PythonASTBlock(
            name=str(index),
            tree=self.current_block)
        self.current_block = []

    def new_branch_block(self, index) -> tuple[int, int]:
        """Create a new block. """
        self.blocks[str(index)] = PythonASTBlock(
            name=str(index),
            _jump_targets=(str(self.block_index),
                           str(self.block_index + 1)),
            tree=self.current_block)
        self.current_block = []
        return_value = (self.block_index, self.block_index + 1)
        self.block_index += 2
        return return_value

    def handle_function_def(self, node: ast.FunctionDef) -> None:
        """Handle a function definition. """
        self.queue.extend(node.body)

    def handle_assign(self, node: ast.Assign) -> None:
        """Handle an assignment. """
        self.current_block.append(node)

    def handle_expr(self, node: ast.Expr) -> None:
        """Handle an expression. """
        self.current_block.append(node)

    def handle_return(self, node: ast.Return) -> None:
        """Handle a return statement. """
        self.current_block.append(node)

    def handle_for(self, node: ast.For) -> None:
        """Handle a for loop. """
        self.new_block()
        self.current_block.append(node)
        self.queue.extend(node.body)
        self.queue.append("ENDFOR")

    def handle_if(self, node: ast.If) -> None:
        """Handle an if statement. """
        self.current_block.append(node.test)
        if len(self.queue) >= 1:
            index = self.queue.popleft()
            if isinstance(index, str) and index.startswith("ENDIF"):
                index = int(index[5:])
            else:
                self.queue.appendleft(index)
                index = self.block_index
                self.block_index += 1
        else:
            index = self.block_index
            self.block_index += 1
        t,f = self.new_branch_block(index)
        self.queue.extend(node.body)
        self.queue.append(f"ENDIF{t}")
        if node.orelse:
            self.queue.extend(node.orelse)
            self.queue.append(f"ENDIF{f}")


def acc():
    r = 0
    for i in range(10):
        r = r + 1
    return r


def branch01(a: int, b:int) -> None:
    if x < 10:
        return 1
    else:
        return 2

def branch02(a: int, b:int) -> None:
    x = a + b
    if x < 10:
        return 1
    else:
        if x < 5:
            return 2
        else:
            return 3


def branch03(a: int, b:int) -> None:
    x = a + b
    if x < 10:
        if x < 2:
            return 1
        else:
            return 2
    else:
        if x < 5:
            return 3
        else:
            return 4

def branch04(a: int, b:int) -> None:
    if x < 10:
        return 1
    y = b + 2
    if y < 5:
        return 2
    return 0


h = ASTHandler(branch03)
s = h.process()
#breakpoint()
from numba_rvsdg.rendering.rendering import render_scfg
render_scfg(s)
