from byteflow2 import ByteFlow, ByteFlowRenderer
from simulator import Simulator
from pprint import pprint

def foo(x):
    c = 0
    for i in range(x):
        c += i
        if i == 100:
            break
    return c


def test_foo():
    flow = ByteFlow.from_bytecode(foo)
    pprint(flow.bbmap)
    flow = flow._join_returns()._restructure_loop()
    pprint(flow.bbmap)
    # pprint(rtsflow.bbmap)
    ByteFlowRenderer().render_byteflow(flow).view()
    sim = Simulator(flow, foo.__globals__)
    ret = sim.run(dict(x=0))
    #breakpoint()
    assert ret == foo(x=0)

    # sim = Simulator(rtsflow, foo.__globals__)
    # ret = sim.run(dict(x=3))
    # assert ret == foo(x=3)

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
    #breakpoint()
    assert ret == bar(x=10)

    # sim = Simulator(rtsflow, foo.__globals__)
    # ret = sim.run(dict(x=3))
    # assert ret == foo(x=3)


if __name__ == "__main__":
    test_foo()
    # test_bar()
