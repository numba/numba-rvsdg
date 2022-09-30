from byteflow2 import ByteFlow, ByteFlowRenderer
from simulator import Simulator
from pprint import pprint
from dis import dis

#    flow = ByteFlow.from_bytecode(foo)
#    #pprint(flow.bbmap)
#    flow = flow.restructure()
#    #pprint(flow.bbmap)
#    # pprint(rtsflow.bbmap)
#    ByteFlowRenderer().render_byteflow(flow).view()
#    print(dis(foo))
#
#    sim = Simulator(flow, foo.__globals__)
#    ret = sim.run(dict(x=1))
#    assert ret == foo(x=1)
#
#    #sim = Simulator(flow, foo.__globals__)
#    #ret = sim.run(dict(x=100))
#    #assert ret == foo(x=100)


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
    assert sim.run(dict(x=2)) == foo(x=2)

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


def test_bar():
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

    #def bar(x):
    #    c = 0
    #    for i in range(x):
    #        c += i
    #        for j in range(x):
    #            c += j
    #            if c > 20:
    #                break

    #    return c

    flow = ByteFlow.from_bytecode(bar)
    flow = flow.restructure()

    ByteFlowRenderer().render_byteflow(flow).view()

    sim = Simulator(flow, bar.__globals__)
    ret = sim.run(dict(x=0))
    assert ret == bar(x=0)

    sim = Simulator(flow, bar.__globals__)
    ret = sim.run(dict(x=5))
    assert ret == bar(x=5)


if __name__ == "__main__":
    test_simple_for_loop()
    test_for_loop_with_exit()
    test_bar()
