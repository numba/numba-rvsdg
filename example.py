# # Demo Byteflow2
#
# Note: This file is written using the _light_ script format of 
#       jupytext and it is intended to be executed as a notebook.

import logging
from pprint import pprint
from byteflow2 import ByteFlow, ByteFlowRenderer


logging.basicConfig(level=logging.DEBUG)

# ## Example 1: For-loop

#def foo(n):
#    c = 0
#    for i in range(n):
#        c += i
#        if i % 2 == 0:
#            c += i
#        else:
#            c <<= 1
#            break
#    return c


# Parse function

#flow = ByteFlow.from_bytecode(foo)
#flow.render_dot().view("before")
#
#rflow =  flow.restructure()
#rflow.render_dot().view("after")

#
## `flow.bbmap.graph` shows a nested structure
#
#pprint(flow.bbmap.graph)
#
#
# ## Example 2: For-loop 2 level

#def foo(n):
#    c = 0
#    for j in range(n):
#        for i in range(i):
#            c += i
#            if i % 2 == 0:
#                c += i
#            else:
#                c <<= 1
#                break
#    return c
#
#
#flow = ByteFlow.from_bytecode(foo)
#ByteFlowRenderer().render_byteflow(flow).view("before")
#
#rflow = flow.restructure()
#ByteFlowRenderer().render_byteflow(rflow).view("after")

#def foo(a, b):
#    if a == 1:
#        for i in range(100):
#            b += 1
#            if b == 50:
#                for j in range(200):
#                    b += 1
#                    if b == 100:
#                        return a
#                    else:
#                        continue
#    elif b == 1:
#        return b


#def foo(x, y):
#    a = y > 0
#    if a:
#        b = x > 0
#        if b:
#            for i in range(42):
#                x = x + y
#            return x
#        else:
#            x = x - y
#    x = x * x
#    return x
#
# Andre's example

def foo(x, y):
    for i in range(100):
        y += 1
    if x == 0:
        return y + 3
    elif x > 0:
        return y + 2
    else:
        return y + 1


flow = ByteFlow.from_bytecode(foo)
ByteFlowRenderer().render_byteflow(flow).view("before")

cflow = flow._join_returns()
ByteFlowRenderer().render_byteflow(cflow).view("closed")

lflow = cflow._restructure_loop()
ByteFlowRenderer().render_byteflow(lflow).view("loop restructured")

bflow = lflow._restructure_branch()
ByteFlowRenderer().render_byteflow(bflow).view("branch restructured")
