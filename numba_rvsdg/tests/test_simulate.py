from numba_rvsdg.core.datastructures.scfg import SCFG
from numba_rvsdg.tests.simulator import Simulator
import unittest


class SimulatorTest(unittest.TestCase):
    def _run(self, func, scfg, kwargs):
        with self.subTest():
            sim = Simulator(scfg, func.__globals__)
            self.assertEqual(sim.run(kwargs), func(**kwargs))

    def test_simple_branch(self):
        def foo(x):
            c = 0
            if x:
                c += 100
            else:
                c += 1000
            return c

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        # if case
        self._run(foo, scfg, {"x": 1})
        # else case
        self._run(foo, scfg, {"x": 0})

    def test_simple_for_loop(self):
        def foo(x):
            c = 0
            for i in range(x):
                c += i
            return c

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        # loop bypass case
        self._run(foo, scfg, {"x": 0})
        # loop case
        self._run(foo, scfg, {"x": 2})
        # extended loop case
        self._run(foo, scfg, {"x": 100})

    def test_simple_while_loop(self):
        def foo(x):
            c = 0
            i = 0
            while i < x:
                c += i
                i += 1
            return c

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        # loop bypass case
        self._run(foo, scfg, {"x": 0})
        # loop case
        self._run(foo, scfg, {"x": 2})
        # extended loop case
        self._run(foo, scfg, {"x": 100})

    def test_for_loop_with_exit(self):
        def foo(x):
            c = 0
            for i in range(x):
                c += i
                if i == 100:
                    break
            return c

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        # loop bypass case
        self._run(foo, scfg, {"x": 0})
        # loop case
        self._run(foo, scfg, {"x": 2})
        # break case
        self._run(foo, scfg, {"x": 15})

    def test_nested_for_loop_with_break_and_continue(self):
        def foo(x):
            c = 0
            for i in range(x):
                c += i
                if c <= 0:
                    continue
                else:
                    for j in range(c):
                        c += j
                        if c > 100:
                            break
            return c

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        # no loop
        self._run(foo, scfg, {"x": 0})
        # only continue
        self._run(foo, scfg, {"x": 1})
        # no break
        self._run(foo, scfg, {"x": 4})
        # will break
        self._run(foo, scfg, {"x": 5})

    def test_for_loop_with_multiple_backedges(self):
        def foo(x):
            c = 0
            for i in range(x):
                if i == 3:
                    c += 100
                elif i == 5:
                    c += 1000
                else:
                    c += 1
            return c

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        # loop bypass
        self._run(foo, scfg, {"x": 0})
        # default on every iteration
        self._run(foo, scfg, {"x": 2})
        # adding 100, via the if clause
        self._run(foo, scfg, {"x": 4})
        # adding 1000, via the elif clause
        self._run(foo, scfg, {"x": 7})

    def test_andor(self):
        def foo(x, y):
            return (x > 0 and x < 10) or (y > 0 and y < 10)

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        self._run(foo, scfg, {"x": 5, "y": 5})

    def test_while_count(self):
        def foo(s, e):
            i = s
            c = 0
            while i < e:
                c += i
                i += 1
            return c

        scfg = SCFG.from_bytecode(foo)
        scfg.restructure()

        # no looping
        self._run(foo, scfg, {"s": 0, "e": 0})
        # single execution
        self._run(foo, scfg, {"s": 0, "e": 1})
        # mutiple iterations
        self._run(foo, scfg, {"s": 0, "e": 5})

        # no looping
        self._run(foo, scfg, {"s": 23, "e": 0})
        # single execution
        self._run(foo, scfg, {"s": 23, "e": 24})
        # mutiple iterations
        self._run(foo, scfg, {"s": 23, "e": 28})


if __name__ == "__main__":
    unittest.main()
