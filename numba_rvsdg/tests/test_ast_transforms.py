# mypy: ignore-errors
import ast
import textwrap
from typing import Callable, Any
from unittest import main, TestCase
from sys import monitoring as sm

from numba_rvsdg.core.datastructures.ast_transforms import (
    unparse_code,
    AST2SCFGTransformer,
    SCFG2ASTTransformer,
    AST2SCFG,
    SCFG2AST,
)

sm.use_tool_id(sm.PROFILER_ID, "custom_tracer")


class LineTraceCallback:

    def __init__(self):
        self.lines = set()

    def __call__(self, code, line):
        self.lines.add(line)


class TestAST2SCFGTransformer(TestCase):

    def compare(
        self,
        function: Callable[..., Any],
        expected: dict[str, dict[str, Any]],
        unreachable: set[int] = set(),
        empty: set[int] = set(),
        arguments: list[any] = [],
    ):
        # Execute function with first argument, if given. Ensure function is
        # sane and make sure it's picked up by coverage.
        try:
            if arguments:
                for a in arguments:
                    _ = function(*a)
            else:
                _ = function()
        except Exception:
            pass
        # First, test against the expected CFG...
        ast2scfg_transformer = AST2SCFGTransformer(function)
        astcfg = ast2scfg_transformer.transform_to_ASTCFG()
        self.assertEqual(expected, astcfg.to_dict())
        self.assertEqual(unreachable, {i.name for i in astcfg.unreachable})
        self.assertEqual(empty, {i.name for i in astcfg.empty})

        # Then restructure, synthesize python and run original and transformed
        # on the same arguments and assert they are the same.
        scfg = astcfg.to_SCFG()
        scfg.restructure()
        scfg2ast = SCFG2ASTTransformer()
        original_ast = unparse_code(function)[0]
        transformed_ast = scfg2ast.transform(original=original_ast, scfg=scfg)

        # use exec to obtin the function and the transformed_function
        original_exec_locals = {}
        exec(ast.unparse(original_ast), {}, original_exec_locals)
        temporary_function = original_exec_locals["function"]
        temporary_exec_locals = {}
        exec(ast.unparse(transformed_ast), {}, temporary_exec_locals)
        temporary_transformed_function = temporary_exec_locals[
            "transformed_function"
        ]

        # Setup the profiler for both funstions and initialize the callbacks
        sm.set_local_events(
            sm.PROFILER_ID, temporary_function.__code__, sm.events.LINE
        )
        sm.set_local_events(
            sm.PROFILER_ID,
            temporary_transformed_function.__code__,
            sm.events.LINE,
        )
        original_callback = LineTraceCallback()
        transformed_callback = LineTraceCallback()

        # Register the callbacks one at a time and collect results.
        sm.register_callback(sm.PROFILER_ID, sm.events.LINE, original_callback)
        if arguments:
            original_results = [temporary_function(*a) for a in arguments]
        else:
            original_results = [temporary_function()]

        # Only one callback can be registered at a time.
        sm.register_callback(
            sm.PROFILER_ID, sm.events.LINE, transformed_callback
        )
        if arguments:
            transformed_results = [
                temporary_transformed_function(*a) for a in arguments
            ]
        else:
            transformed_results = [temporary_transformed_function()]

        assert len(original_callback.lines) > 0
        assert len(transformed_callback.lines) > 0

        # Check call results
        assert original_results == transformed_results

        # Check line trace of original
        original_source = ast.unparse(original_ast).splitlines()
        assert [
            i + 1
            for i, l in enumerate(original_source)
            if not l.startswith("def") and "else:" not in l
        ] == sorted(original_callback.lines)

        # Check line trace of transformed
        transformed_source = ast.unparse(transformed_ast).splitlines()
        assert [
            i + 1
            for i, l in enumerate(transformed_source)
            if not l.startswith("def") and "else:" not in l
        ] == sorted(transformed_callback.lines)

    def setUp(self):
        # Enable pytest verbose output.
        self.maxDiff = None

    def test_solo_return(self):
        def function() -> int:
            return 1

        expected = {
            "0": {
                "instructions": ["return 1"],
                "jump_targets": [],
                "name": "0",
            }
        }
        self.compare(function, expected)

    def test_solo_return_from_string(self):
        function = textwrap.dedent(
            """
            def function() -> int:
                return 1
        """
        )

        expected = {
            "0": {
                "instructions": ["return 1"],
                "jump_targets": [],
                "name": "0",
            }
        }
        self.compare(function, expected)

    def test_solo_return_from_AST(self):
        function = ast.parse(
            textwrap.dedent(
                """
            def function() -> int:
                return 1
        """
            )
        ).body

        expected = {
            "0": {
                "instructions": ["return 1"],
                "jump_targets": [],
                "name": "0",
            }
        }
        self.compare(function, expected)

    def test_solo_assign(self):
        def function() -> None:
            x = 1  # noqa: F841

        expected = {
            "0": {
                "instructions": ["x = 1", "return"],
                "jump_targets": [],
                "name": "0",
            }
        }
        self.compare(function, expected)

    def test_solo_pass(self):
        def function() -> None:
            pass

        expected = {
            "0": {
                "instructions": ["return"],
                "jump_targets": [],
                "name": "0",
            }
        }
        self.compare(function, expected)

    def test_assign_return(self):
        def function() -> int:
            x = 1
            return x

        expected = {
            "0": {
                "instructions": ["x = 1", "return x"],
                "jump_targets": [],
                "name": "0",
            }
        }
        self.compare(function, expected)

    def test_if_return(self):
        def function(x: int) -> int:
            if x < 1:
                return 1
            return 2

        expected = {
            "0": {
                "instructions": ["x < 1"],
                "jump_targets": ["1", "3"],
                "name": "0",
            },
            "1": {
                "instructions": ["return 1"],
                "jump_targets": [],
                "name": "1",
            },
            "3": {
                "instructions": ["return 2"],
                "jump_targets": [],
                "name": "3",
            },
        }
        self.compare(function, expected, empty={"2"}, arguments=[(0,), (1,)])

    def test_if_else_return(self):
        def function(x: int) -> int:
            if x < 1:
                return 1
            else:
                return 2

        expected = {
            "0": {
                "instructions": ["x < 1"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["return 1"],
                "jump_targets": [],
                "name": "1",
            },
            "2": {
                "instructions": ["return 2"],
                "jump_targets": [],
                "name": "2",
            },
        }
        self.compare(
            function, expected, unreachable={"3"}, arguments=[(0,), (1,)]
        )

    def test_if_else_assign(self):
        def function(x: int) -> int:
            if x < 1:
                y = 1
            else:
                y = 2
            return y

        expected = {
            "0": {
                "instructions": ["x < 1"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["y = 1"],
                "jump_targets": ["3"],
                "name": "1",
            },
            "2": {
                "instructions": ["y = 2"],
                "jump_targets": ["3"],
                "name": "2",
            },
            "3": {
                "instructions": ["return y"],
                "jump_targets": [],
                "name": "3",
            },
        }
        self.compare(function, expected, arguments=[(0,), (1,)])

    def test_nested_if(self):
        def function(x: int, y: int) -> int:
            if x < 1:
                if y < 1:
                    z = 1
                else:
                    z = 2
            else:
                if y < 2:
                    z = 3
                else:
                    z = 4
            return z

        expected = {
            "0": {
                "instructions": ["x < 1"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["y < 1"],
                "jump_targets": ["4", "5"],
                "name": "1",
            },
            "2": {
                "instructions": ["y < 2"],
                "jump_targets": ["7", "8"],
                "name": "2",
            },
            "3": {
                "instructions": ["return z"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["z = 1"],
                "jump_targets": ["3"],
                "name": "4",
            },
            "5": {
                "instructions": ["z = 2"],
                "jump_targets": ["3"],
                "name": "5",
            },
            "7": {
                "instructions": ["z = 3"],
                "jump_targets": ["3"],
                "name": "7",
            },
            "8": {
                "instructions": ["z = 4"],
                "jump_targets": ["3"],
                "name": "8",
            },
        }
        self.compare(
            function,
            expected,
            empty={"6", "9"},
            arguments=[(0, 0), (0, 1), (1, 1), (1, 2)],
        )

    def test_nested_if_with_empty_else_and_return(self):
        def function(x: int, y: int) -> None:
            z = 0
            if x < 1:
                y << 2
                if y < 1:
                    z = 1
            else:
                if y < 2:
                    z = 2
                else:
                    return y, 4
                y += 1
            return y, z

        expected = {
            "0": {
                "instructions": ["z = 0", "x < 1"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["y << 2", "y < 1"],
                "jump_targets": ["4", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["y < 2"],
                "jump_targets": ["7", "8"],
                "name": "2",
            },
            "3": {
                "instructions": ["return (y, z)"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["z = 1"],
                "jump_targets": ["3"],
                "name": "4",
            },
            "7": {
                "instructions": ["z = 2"],
                "jump_targets": ["9"],
                "name": "7",
            },
            "8": {
                "instructions": ["return (y, 4)"],
                "jump_targets": [],
                "name": "8",
            },
            "9": {
                "instructions": ["y += 1"],
                "jump_targets": ["3"],
                "name": "9",
            },
        }
        self.compare(
            function,
            expected,
            empty={"5", "6"},
            arguments=[
                (0, 0),
                (0, 1),
                (1, 1),
                (1, 2),
            ],
        )

    def test_elif(self):
        def function(x: int) -> int:
            if x < 1:
                return 10
            elif x < 2:
                y = 20
            elif x < 3:
                y = 30
            else:
                y = 40
            return y

        expected = {
            "0": {
                "instructions": ["x < 1"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["return 10"],
                "jump_targets": [],
                "name": "1",
            },
            "2": {
                "instructions": ["x < 2"],
                "jump_targets": ["4", "5"],
                "name": "2",
            },
            "3": {
                "instructions": ["return y"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["y = 20"],
                "jump_targets": ["3"],
                "name": "4",
            },
            "5": {
                "instructions": ["x < 3"],
                "jump_targets": ["7", "8"],
                "name": "5",
            },
            "7": {
                "instructions": ["y = 30"],
                "jump_targets": ["3"],
                "name": "7",
            },
            "8": {
                "instructions": ["y = 40"],
                "jump_targets": ["3"],
                "name": "8",
            },
        }
        self.compare(
            function,
            expected,
            empty={"9", "6"},
            arguments=[
                (0,),
                (1,),
                (2,),
                (3,),
            ],
        )

    def test_simple_while(self):
        def function() -> int:
            x = 0
            while x < 10:
                x += 1
            return x

        expected = {
            "0": {
                "instructions": ["x = 0"],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": ["x < 10"],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["x += 1"],
                "jump_targets": ["1"],
                "name": "2",
            },
            "3": {
                "instructions": ["return x"],
                "jump_targets": [],
                "name": "3",
            },
        }
        self.compare(function, expected, empty={"4"})

    def test_nested_while(self):
        def function() -> tuple[int, int]:
            x, y = 0, 0
            while x < 10:
                while y < 5:
                    x += 1
                    y += 1
                x += 1
            return x, y

        expected = {
            "0": {
                "instructions": ["x, y = (0, 0)"],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": ["x < 10"],
                "jump_targets": ["5", "3"],
                "name": "1",
            },
            "3": {
                "instructions": ["return (x, y)"],
                "jump_targets": [],
                "name": "3",
            },
            "5": {
                "instructions": ["y < 5"],
                "jump_targets": ["6", "7"],
                "name": "5",
            },
            "6": {
                "instructions": ["x += 1", "y += 1"],
                "jump_targets": ["5"],
                "name": "6",
            },
            "7": {
                "instructions": ["x += 1"],
                "jump_targets": ["1"],
                "name": "7",
            },
        }

        self.compare(function, expected, empty={"2", "4", "8"})

    def test_if_in_while(self):
        def function() -> int:
            x = 0
            while x < 10:
                if x < 5:
                    x += 2
                else:
                    x += 1
            return x

        expected = {
            "0": {
                "instructions": ["x = 0"],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": ["x < 10"],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["x < 5"],
                "jump_targets": ["5", "6"],
                "name": "2",
            },
            "3": {
                "instructions": ["return x"],
                "jump_targets": [],
                "name": "3",
            },
            "5": {
                "instructions": ["x += 2"],
                "jump_targets": ["1"],
                "name": "5",
            },
            "6": {
                "instructions": ["x += 1"],
                "jump_targets": ["1"],
                "name": "6",
            },
        }
        self.compare(function, expected, empty={"4", "7"})

    def test_while_in_if(self):
        def function(y: int) -> int:
            x = 0
            if y == 0:
                while x < 10:
                    x += 2
            else:
                while x < 10:
                    x += 1
            return x

        expected = {
            "0": {
                "instructions": ["x = 0", "y == 0"],
                "jump_targets": ["4", "8"],
                "name": "0",
            },
            "3": {
                "instructions": ["return x"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["x < 10"],
                "jump_targets": ["5", "3"],
                "name": "4",
            },
            "5": {
                "instructions": ["x += 2"],
                "jump_targets": ["4"],
                "name": "5",
            },
            "8": {
                "instructions": ["x < 10"],
                "jump_targets": ["9", "3"],
                "name": "8",
            },
            "9": {
                "instructions": ["x += 1"],
                "jump_targets": ["8"],
                "name": "9",
            },
        }
        self.compare(
            function,
            expected,
            empty={"1", "2", "6", "7", "10", "11"},
            arguments=[(0,), (1,)],
        )

    def test_while_break_continue(self):
        def function(x: int) -> int:
            y = 0
            while y < 10:
                y += 1
                if x == 0:
                    continue
                elif x == 1:
                    break
                else:
                    y += 10
            return y

        expected = {
            "0": {
                "instructions": ["y = 0"],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": ["y < 10"],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["y += 1", "x == 0"],
                "jump_targets": ["1", "6"],
                "name": "2",
            },
            "3": {
                "instructions": ["return y"],
                "jump_targets": [],
                "name": "3",
            },
            "6": {
                "instructions": ["x == 1"],
                "jump_targets": ["3", "9"],
                "name": "6",
            },
            "9": {
                "instructions": ["y += 10"],
                "jump_targets": ["1"],
                "name": "9",
            },
        }
        self.compare(
            function,
            expected,
            empty={"4", "5", "7", "8", "10"},
            arguments=[(0,), (1,), (2,)],
        )

    def test_while_else(self):
        def function() -> int:
            x = 0
            while x < 10:
                x += 1
            else:
                x += 1
            return x

        expected = {
            "0": {
                "instructions": ["x = 0"],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": ["x < 10"],
                "jump_targets": ["2", "4"],
                "name": "1",
            },
            "2": {
                "instructions": ["x += 1"],
                "jump_targets": ["1"],
                "name": "2",
            },
            "3": {
                "instructions": ["return x"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["x += 1"],
                "jump_targets": ["3"],
                "name": "4",
            },
        }
        self.compare(function, expected)

    def test_simple_for(self):
        def function() -> int:
            x = 0
            for i in range(10):
                x += i
            return x, i

        expected = {
            "0": {
                "instructions": [
                    "x = 0",
                    "__iterator_1__ = iter(range(10))",
                    "i = None",
                ],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": [
                    "__iter_last_1__ = i",
                    "i = next(__iterator_1__, '__sentinel__')",
                    "i != '__sentinel__'",
                ],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["x += i"],
                "jump_targets": ["1"],
                "name": "2",
            },
            "3": {
                "instructions": ["i = __iter_last_1__"],
                "jump_targets": ["4"],
                "name": "3",
            },
            "4": {
                "instructions": ["return (x, i)"],
                "jump_targets": [],
                "name": "4",
            },
        }
        self.compare(function, expected)

    def test_nested_for(self):
        def function() -> int:
            x = 0
            for i in range(3):
                x += i
                for j in range(3):
                    x += j
            return x, i, j

        expected = {
            "0": {
                "instructions": [
                    "x = 0",
                    "__iterator_1__ = iter(range(3))",
                    "i = None",
                ],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": [
                    "__iter_last_1__ = i",
                    "i = next(__iterator_1__, '__sentinel__')",
                    "i != '__sentinel__'",
                ],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": [
                    "x += i",
                    "__iterator_5__ = iter(range(3))",
                    "j = None",
                ],
                "jump_targets": ["5"],
                "name": "2",
            },
            "3": {
                "instructions": ["i = __iter_last_1__"],
                "jump_targets": ["4"],
                "name": "3",
            },
            "4": {
                "instructions": ["return (x, i, j)"],
                "jump_targets": [],
                "name": "4",
            },
            "5": {
                "instructions": [
                    "__iter_last_5__ = j",
                    "j = next(__iterator_5__, '__sentinel__')",
                    "j != '__sentinel__'",
                ],
                "jump_targets": ["6", "7"],
                "name": "5",
            },
            "6": {
                "instructions": ["x += j"],
                "jump_targets": ["5"],
                "name": "6",
            },
            "7": {
                "instructions": ["j = __iter_last_5__"],
                "jump_targets": ["1"],
                "name": "7",
            },
        }
        self.compare(function, expected, empty={"8"})

    def test_for_with_return_break_and_continue(self):
        def function(x: int, y: int) -> int:
            for i in range(2):
                if i == x:
                    i = 3
                    return i
                elif i == y:
                    i = 4
                    break
                else:
                    continue
            return i

        expected = {
            "0": {
                "instructions": [
                    "__iterator_1__ = iter(range(2))",
                    "i = None",
                ],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": [
                    "__iter_last_1__ = i",
                    "i = next(__iterator_1__, '__sentinel__')",
                    "i != '__sentinel__'",
                ],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["i == x"],
                "jump_targets": ["5", "6"],
                "name": "2",
            },
            "3": {
                "instructions": ["i = __iter_last_1__"],
                "jump_targets": ["4"],
                "name": "3",
            },
            "4": {
                "instructions": ["return i"],
                "jump_targets": [],
                "name": "4",
            },
            "5": {
                "instructions": ["i = 3", "return i"],
                "jump_targets": [],
                "name": "5",
            },
            "6": {
                "instructions": ["i == y"],
                "jump_targets": ["8", "1"],
                "name": "6",
            },
            "8": {
                "instructions": ["i = 4"],
                "jump_targets": ["4"],
                "name": "8",
            },
        }
        self.compare(
            function,
            expected,
            unreachable={"7", "10"},
            empty={"9"},
            arguments=[(0, 0), (2, 0), (2, 2)],
        )

    def test_for_with_if_in_else(self):
        def function(y: int):
            x = 0
            for i in range(10):
                x += i
            else:
                if y == 0:
                    x += 10
                else:
                    x += 20
            return (x, y, i)

        expected = {
            "0": {
                "instructions": [
                    "x = 0",
                    "__iterator_1__ = iter(range(10))",
                    "i = None",
                ],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": [
                    "__iter_last_1__ = i",
                    "i = next(__iterator_1__, '__sentinel__')",
                    "i != '__sentinel__'",
                ],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["x += i"],
                "jump_targets": ["1"],
                "name": "2",
            },
            "3": {
                "instructions": ["i = __iter_last_1__", "y == 0"],
                "jump_targets": ["5", "6"],
                "name": "3",
            },
            "4": {
                "instructions": ["return (x, y, i)"],
                "jump_targets": [],
                "name": "4",
            },
            "5": {
                "instructions": ["x += 10"],
                "jump_targets": ["4"],
                "name": "5",
            },
            "6": {
                "instructions": ["x += 20"],
                "jump_targets": ["4"],
                "name": "6",
            },
        }
        self.compare(function, expected, empty={"7"}, arguments=[(0,), (1,)])

    def test_for_with_nested_for_else(self):
        def function(y: int) -> int:
            x = 1
            for i in range(1):
                for j in range(1):
                    if y == 0:
                        x *= 3
                        break  # This break decides, if True skip continue.
                else:
                    x *= 5
                    continue  # Causes break below to be skipped.
                x *= 7
                break  # Causes the else below to be skipped
            else:
                x *= 9  # Not breaking in inner loop leads here
            return x, y, i

        self.assertEqual(function(1)[0], 5 * 9)
        self.assertEqual(function(0)[0], 3 * 7)
        expected = {
            "0": {
                "instructions": [
                    "x = 1",
                    "__iterator_1__ = iter(range(1))",
                    "i = None",
                ],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": [
                    "__iter_last_1__ = i",
                    "i = next(__iterator_1__, '__sentinel__')",
                    "i != '__sentinel__'",
                ],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": [
                    "__iterator_5__ = iter(range(1))",
                    "j = None",
                ],
                "jump_targets": ["5"],
                "name": "2",
            },
            "3": {
                "instructions": ["i = __iter_last_1__", "x *= 9"],
                "jump_targets": ["4"],
                "name": "3",
            },
            "4": {
                "instructions": ["return (x, y, i)"],
                "jump_targets": [],
                "name": "4",
            },
            "5": {
                "instructions": [
                    "__iter_last_5__ = j",
                    "j = next(__iterator_5__, '__sentinel__')",
                    "j != '__sentinel__'",
                ],
                "jump_targets": ["6", "7"],
                "name": "5",
            },
            "6": {
                "instructions": ["y == 0"],
                "jump_targets": ["9", "5"],
                "name": "6",
            },
            "7": {
                "instructions": ["j = __iter_last_5__", "x *= 5"],
                "jump_targets": ["1"],
                "name": "7",
            },
            "8": {
                "instructions": ["x *= 7"],
                "jump_targets": ["4"],
                "name": "8",
            },
            "9": {
                "instructions": ["x *= 3"],
                "jump_targets": ["8"],
                "name": "9",
            },
        }

        self.compare(
            function, expected, empty={"11", "10"}, arguments=[(0,), (1,)]
        )

    def test_for_with_nested_else_return_break_and_continue(self):
        def function(x: int) -> int:
            for i in range(2):
                if x == 1:
                    i += 1
                    return i
                elif x == 2:
                    i += 2
                    break
                elif x == 3:
                    i += 3
                    continue
                else:
                    while i < 10:
                        i += 1
                        if x == 4:
                            i += 4
                            return i
                        elif x == 5:
                            i += 5
                            break
                        elif x == 6:
                            i += 6
                            continue
                        else:
                            i += 7
            return x, i

        expected = {
            "0": {
                "instructions": [
                    "__iterator_1__ = iter(range(2))",
                    "i = None",
                ],
                "jump_targets": ["1"],
                "name": "0",
            },
            "1": {
                "instructions": [
                    "__iter_last_1__ = i",
                    "i = next(__iterator_1__, '__sentinel__')",
                    "i != '__sentinel__'",
                ],
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "11": {
                "instructions": ["i += 3"],
                "jump_targets": ["1"],
                "name": "11",
            },
            "14": {
                "instructions": ["i < 10"],
                "jump_targets": ["15", "1"],
                "name": "14",
            },
            "15": {
                "instructions": ["i += 1", "x == 4"],
                "jump_targets": ["18", "19"],
                "name": "15",
            },
            "18": {
                "instructions": ["i += 4", "return i"],
                "jump_targets": [],
                "name": "18",
            },
            "19": {
                "instructions": ["x == 5"],
                "jump_targets": ["21", "22"],
                "name": "19",
            },
            "2": {
                "instructions": ["x == 1"],
                "jump_targets": ["5", "6"],
                "name": "2",
            },
            "21": {
                "instructions": ["i += 5"],
                "jump_targets": ["1"],
                "name": "21",
            },
            "22": {
                "instructions": ["x == 6"],
                "jump_targets": ["24", "25"],
                "name": "22",
            },
            "24": {
                "instructions": ["i += 6"],
                "jump_targets": ["14"],
                "name": "24",
            },
            "25": {
                "instructions": ["i += 7"],
                "jump_targets": ["14"],
                "name": "25",
            },
            "3": {
                "instructions": ["i = __iter_last_1__"],
                "jump_targets": ["4"],
                "name": "3",
            },
            "4": {
                "instructions": ["return (x, i)"],
                "jump_targets": [],
                "name": "4",
            },
            "5": {
                "instructions": ["i += 1", "return i"],
                "jump_targets": [],
                "name": "5",
            },
            "6": {
                "instructions": ["x == 2"],
                "jump_targets": ["8", "9"],
                "name": "6",
            },
            "8": {
                "instructions": ["i += 2"],
                "jump_targets": ["4"],
                "name": "8",
            },
            "9": {
                "instructions": ["x == 3"],
                "jump_targets": ["11", "14"],
                "name": "9",
            },
        }
        empty = {"7", "10", "12", "13", "16", "17", "20", "23", "26"}
        arguments = [(1,), (2,), (3,), (4,), (5,), (6,), (7,)]
        self.compare(
            function,
            expected,
            empty=empty,
            arguments=arguments,
        )


class TestEntryPoints(TestCase):

    def test_rondtrip(self):

        def function() -> int:
            x = 0
            for i in range(2):
                x += i
            return x, i

        scfg = AST2SCFG(function)
        scfg.restructure()
        ast_ = SCFG2AST(function, scfg)

        exec_locals = {}
        exec(ast.unparse(ast_), {}, exec_locals)
        transformed = exec_locals["transformed_function"]
        assert function() == transformed()


if __name__ == "__main__":
    main()
