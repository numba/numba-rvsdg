from byteflow2 import ByteFlow, ByteFlowRenderer
from simulator import Simulator
from pprint import pprint

def foo(x):
    c = 0
    for i in range(x):
        c += i
    return c


def test_foo():
    flow = ByteFlow.from_bytecode(foo)
    pprint(flow.bbmap)
    rtsflow = flow.restructure()
    pprint(rtsflow.bbmap)
    # ByteFlowRenderer().render_byteflow(rtsflow).view()
    # sim = Simulator(flow, foo.__globals__)
    # ret = sim.run(dict(x=3))
    # assert ret == foo(x=3)

    sim = Simulator(rtsflow, foo.__globals__)
    ret = sim.run(dict(x=3))
    assert ret == foo(x=3)

if __name__ == "__main__":
    test_foo()