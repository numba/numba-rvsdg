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

    # test the loop bypass case
    sim = Simulator(flow, foo.__globals__)
    assert sim.run(dict(x=0)) == foo(x=0)

    # test the loop case
    sim = Simulator(flow, foo.__globals__)
    assert sim.run(dict(x=1)) == foo(x=1)

    # test an extended loop case
    sim = Simulator(flow, foo.__globals__)
    assert sim.run(dict(x=100)) == foo(x=100)


def foo(x):
#    c = 0
#    for i in range(x):
#        c += i
#        if i == 100:
#            break
#    return c
    c = 0
    for i in range(x):
        c += i
    return c


def test_foo():
    flow = ByteFlow.from_bytecode(foo)
    #pprint(flow.bbmap)
    flow = flow.restructure()
    #pprint(flow.bbmap)
    # pprint(rtsflow.bbmap)
    ByteFlowRenderer().render_byteflow(flow).view()
    print(dis(foo))

    sim = Simulator(flow, foo.__globals__)
    ret = sim.run(dict(x=1))
    assert ret == foo(x=1)

    #sim = Simulator(flow, foo.__globals__)
    #ret = sim.run(dict(x=100))
    #assert ret == foo(x=100)


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
