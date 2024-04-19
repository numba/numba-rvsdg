# mypy: ignore-errors
from typing import Callable, Any
from unittest import main, TestCase

from numba_rvsdg.core.datastructures.ast_transforms import AST2SCFGTransformer


class TestAST2SCFGTransformer(TestCase):

    def compare(
        self,
        function: Callable[..., Any],
        expected: dict[str, dict[str, Any]],
        unreachable: set[int] = set(),
        empty: set[int] = set(),
    ):
        transformer = AST2SCFGTransformer(function)
        astcfg = transformer.transform_to_ASTCFG()
        self.assertEqual(expected, astcfg.to_dict())
        self.assertEqual(unreachable, {i.name for i in astcfg.unreachable})
        self.assertEqual(empty, {i.name for i in astcfg.empty})

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
            if x < 10:
                return 1
            return 2

        expected = {
            "0": {
                "instructions": ["x < 10"],
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
        self.compare(function, expected, empty={"2"})

    def test_if_else_return(self):
        def function(x: int) -> int:
            if x < 10:
                return 1
            else:
                return 2

        expected = {
            "0": {
                "instructions": ["x < 10"],
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
        self.compare(function, expected, unreachable={"3"})

    def test_if_else_assign(self):
        def function(x: int) -> int:
            if x < 10:
                z = 1
            else:
                z = 2
            return z

        expected = {
            "0": {
                "instructions": ["x < 10"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["z = 1"],
                "jump_targets": ["3"],
                "name": "1",
            },
            "2": {
                "instructions": ["z = 2"],
                "jump_targets": ["3"],
                "name": "2",
            },
            "3": {
                "instructions": ["return z"],
                "jump_targets": [],
                "name": "3",
            },
        }
        self.compare(function, expected)

    def test_nested_if(self):
        def function(x: int, y: int) -> int:
            if x < 10:
                if y < 5:
                    y = 1
                else:
                    y = 2
            else:
                if y < 15:
                    y = 3
                else:
                    y = 4
            return y

        expected = {
            "0": {
                "instructions": ["x < 10"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["y < 5"],
                "jump_targets": ["4", "5"],
                "name": "1",
            },
            "2": {
                "instructions": ["y < 15"],
                "jump_targets": ["7", "8"],
                "name": "2",
            },
            "3": {
                "instructions": ["return y"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["y = 1"],
                "jump_targets": ["3"],
                "name": "4",
            },
            "5": {
                "instructions": ["y = 2"],
                "jump_targets": ["3"],
                "name": "5",
            },
            "7": {
                "instructions": ["y = 3"],
                "jump_targets": ["3"],
                "name": "7",
            },
            "8": {
                "instructions": ["y = 4"],
                "jump_targets": ["3"],
                "name": "8",
            },
        }
        self.compare(function, expected, empty={"6", "9"})

    def test_nested_if_with_empty_else_and_return(self):
        def function(x: int, y: int) -> None:
            y << 2
            if x < 10:
                y -= 1
                if y < 5:
                    y = 1
            else:
                if y < 15:
                    y = 2
                else:
                    return
                y += 1
            return y

        expected = {
            "0": {
                "instructions": ["y << 2", "x < 10"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["y -= 1", "y < 5"],
                "jump_targets": ["4", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["y < 15"],
                "jump_targets": ["7", "8"],
                "name": "2",
            },
            "3": {
                "instructions": ["return y"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["y = 1"],
                "jump_targets": ["3"],
                "name": "4",
            },
            "7": {
                "instructions": ["y = 2"],
                "jump_targets": ["9"],
                "name": "7",
            },
            "8": {"instructions": ["return"], "jump_targets": [], "name": "8"},
            "9": {
                "instructions": ["y += 1"],
                "jump_targets": ["3"],
                "name": "9",
            },
        }
        self.compare(function, expected, empty={"5", "6"})

    def test_elif(self):
        def function(x: int, a: int, b: int) -> int:
            if x < 10:
                return
            elif x < 15:
                y = b - a
            elif x < 20:
                y = a**2
            else:
                y = a - b
            return y

        expected = {
            "0": {
                "instructions": ["x < 10"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {"instructions": ["return"], "jump_targets": [], "name": "1"},
            "2": {
                "instructions": ["x < 15"],
                "jump_targets": ["4", "5"],
                "name": "2",
            },
            "3": {
                "instructions": ["return y"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["y = b - a"],
                "jump_targets": ["3"],
                "name": "4",
            },
            "5": {
                "instructions": ["x < 20"],
                "jump_targets": ["7", "8"],
                "name": "5",
            },
            "7": {
                "instructions": ["y = a ** 2"],
                "jump_targets": ["3"],
                "name": "7",
            },
            "8": {
                "instructions": ["y = a - b"],
                "jump_targets": ["3"],
                "name": "8",
            },
        }
        self.compare(function, expected, empty={"9", "6"})

    def test_simple_loop(self):
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
        self.compare(function, expected)

    def test_nested_loop(self):
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
                "jump_targets": ["2", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["y < 5"],
                "jump_targets": ["4", "5"],
                "name": "2",
            },
            "3": {
                "instructions": ["return (x, y)"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["x += 1", "y += 1"],
                "jump_targets": ["2"],
                "name": "4",
            },
            "5": {
                "instructions": ["x += 1"],
                "jump_targets": ["1"],
                "name": "5",
            },
        }
        self.compare(function, expected)

    def test_if_in_loop(self):
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
                "jump_targets": ["4", "5"],
                "name": "2",
            },
            "3": {
                "instructions": ["return x"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["x += 2"],
                "jump_targets": ["1"],
                "name": "4",
            },
            "5": {
                "instructions": ["x += 1"],
                "jump_targets": ["1"],
                "name": "5",
            },
        }
        self.compare(function, expected, empty={"6"})

    def test_loop_in_if(self):
        def function(a: bool) -> int:
            x = 0
            if a is True:
                while x < 10:
                    x += 2
            else:
                while x < 10:
                    x += 1
            return x

        expected = {
            "0": {
                "instructions": ["x = 0", "a is True"],
                "jump_targets": ["1", "2"],
                "name": "0",
            },
            "1": {
                "instructions": ["x < 10"],
                "jump_targets": ["4", "3"],
                "name": "1",
            },
            "2": {
                "instructions": ["x < 10"],
                "jump_targets": ["6", "3"],
                "name": "2",
            },
            "3": {
                "instructions": ["return x"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["x += 2"],
                "jump_targets": ["1"],
                "name": "4",
            },
            "6": {
                "instructions": ["x += 1"],
                "jump_targets": ["2"],
                "name": "6",
            },
        }
        self.compare(function, expected, empty={"5", "7"})

    def test_loop_break_continue(self):
        def function() -> int:
            x = 0
            while x < 10:
                x += 1
                if x % 2 == 0:
                    continue
                elif x == 9:
                    break
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
                "instructions": ["x += 1", "x % 2 == 0"],
                "jump_targets": ["4", "5"],
                "name": "2",
            },
            "3": {
                "instructions": ["return x"],
                "jump_targets": [],
                "name": "3",
            },
            "4": {
                "instructions": ["continue"],
                "jump_targets": ["1"],
                "name": "4",
            },
            "5": {
                "instructions": ["x == 9"],
                "jump_targets": ["7", "8"],
                "name": "5",
            },
            "7": {
                "instructions": ["break"],
                "jump_targets": ["3"],
                "name": "7",
            },
            "8": {
                "instructions": ["x += 1"],
                "jump_targets": ["1"],
                "name": "8",
            },
        }
        self.compare(function, expected, empty={"6", "9"})

    def test_simple_for(self):
        def function() -> int:
            c = 0
            for i in range(10):
                c += i
            return c

        expected = {
            "0": {
                "instructions": [
                    "c = 0",
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
                "instructions": ["c += i"],
                "jump_targets": ["1"],
                "name": "2",
            },
            "3": {
                "instructions": ["i = __iter_last_1__"],
                "jump_targets": ["4"],
                "name": "3",
            },
            "4": {
                "instructions": ["return c"],
                "jump_targets": [],
                "name": "4",
            },
        }
        self.compare(function, expected)

    def test_nested_for(self):
        def function() -> int:
            c = 0
            for i in range(3):
                c += i
                for j in range(3):
                    c += j
            return c

        expected = {
            "0": {
                "instructions": [
                    "c = 0",
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
                    "c += i",
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
                "instructions": ["return c"],
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
                "instructions": ["c += j"],
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
        def function(a: int, b: int) -> int:
            for i in range(2):
                if i == a:
                    i = 3
                    return i
                elif i == b:
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
                "instructions": ["i == a"],
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
                "instructions": ["i == b"],
                "jump_targets": ["8", "9"],
                "name": "6",
            },
            "8": {
                "instructions": ["i = 4", "break"],
                "jump_targets": ["4"],
                "name": "8",
            },
            "9": {
                "instructions": ["continue"],
                "jump_targets": ["1"],
                "name": "9",
            },
        }
        self.compare(function, expected, unreachable={"7", "10"})

    def test_for_with_nested_else_return_break_and_continue(self):
        def function(a: int, b: int, c: int, d: int, e: int, f: int) -> int:
            for i in range(2):
                if i == a:
                    i = 3
                    return i
                elif i == b:
                    i = 4
                    break
                elif i == c:
                    i = 5
                    continue
                else:
                    while i < 10:
                        i += 1
                        if i == d:
                            i = 3
                            return i
                        elif i == e:
                            i = 4
                            break
                        elif i == f:
                            i = 5
                            continue
                        else:
                            i += 1
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
            "11": {
                "instructions": ["i = 5", "continue"],
                "jump_targets": ["1"],
                "name": "11",
            },
            "12": {
                "instructions": ["i < 10"],
                "jump_targets": ["14", "1"],
                "name": "12",
            },
            "14": {
                "instructions": ["i += 1", "i == d"],
                "jump_targets": ["16", "17"],
                "name": "14",
            },
            "16": {
                "instructions": ["i = 3", "return i"],
                "jump_targets": [],
                "name": "16",
            },
            "17": {
                "instructions": ["i == e"],
                "jump_targets": ["19", "20"],
                "name": "17",
            },
            "19": {
                "instructions": ["i = 4", "break"],
                "jump_targets": ["1"],
                "name": "19",
            },
            "2": {
                "instructions": ["i == a"],
                "jump_targets": ["5", "6"],
                "name": "2",
            },
            "20": {
                "instructions": ["i == f"],
                "jump_targets": ["22", "23"],
                "name": "20",
            },
            "22": {
                "instructions": ["i = 5", "continue"],
                "jump_targets": ["12"],
                "name": "22",
            },
            "23": {
                "instructions": ["i += 1"],
                "jump_targets": ["12"],
                "name": "23",
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
                "instructions": ["i == b"],
                "jump_targets": ["8", "9"],
                "name": "6",
            },
            "8": {
                "instructions": ["i = 4", "break"],
                "jump_targets": ["4"],
                "name": "8",
            },
            "9": {
                "instructions": ["i == c"],
                "jump_targets": ["11", "12"],
                "name": "9",
            },
        }
        empty = {"7", "10", "13", "15", "18", "21", "24"}
        self.compare(function, expected, empty=empty)


if __name__ == "__main__":
    main()
