from byteflow2 import ByteFlow, ByteFlowRenderer
from simulator import Simulator
from pprint import pprint
from dis import dis

def test_simple_for_loop():

    def foo(x):
        c = 0
        for i in range(x):
            c += i
        return c

    flow = ByteFlow.from_bytecode(foo)
    flow = flow.restructure()

    # loop bypass case
    sim = Simulator(flow, foo.__globals__)
    assert sim.run(dict(x=0)) == foo(x=0)

    # loop case
    sim = Simulator(flow, foo.__globals__)
    assert sim.run(dict(x=1)) == foo(x=1)

    # extended loop case
    sim = Simulator(flow, foo.__globals__)
    assert sim.run(dict(x=100)) == foo(x=100)


def test_for_loop_with_exit():

    def foo(x):
        c = 0
        for i in range(x):
            c += i
            if i == 100:
                break
        return c

    flow = ByteFlow.from_bytecode(foo)
    flow = flow.restructure()

    ByteFlowRenderer().render_byteflow(flow).view()

    # loop bypass case
    sim = Simulator(flow, foo.__globals__)
    ret = sim.run(dict(x=0))
    assert ret == foo(x=0)

    # loop case
    sim = Simulator(flow, foo.__globals__)
    ret = sim.run(dict(x=2))
    assert ret == foo(x=2)

    # break case
    sim = Simulator(flow, foo.__globals__)
    ret = sim.run(dict(x=15))
    assert ret == foo(x=15)



def bar(x):
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


def test_bar():
    flow = ByteFlow.from_bytecode(bar)
    pprint(flow.bbmap)
    flow = flow._join_returns()._restructure_loop()
    pprint(flow.bbmap)
    # pprint(rtsflow.bbmap)
    ByteFlowRenderer().render_byteflow(flow).view()
    sim = Simulator(flow, bar.__globals__)
    ret = sim.run(dict(x=10))
    breakpoint()
    assert ret == bar(x=10)

    # sim = Simulator(rtsflow, foo.__globals__)
    # ret = sim.run(dict(x=3))
    # assert ret == foo(x=3)


if __name__ == "__main__":
    test_simple_for_loop()
    #test_foo()
    #test_bar()
