# # Demo Byteflow2
#
# Note: This file is written using the _light_ script format of 
#       jupytext and it is intended to be executed as a notebook.

import logging
from pprint import pprint
from byteflow2 import parse_bytecode


logging.basicConfig(level=logging.DEBUG)

# ## Example 1: For-loop

def foo(n):
    c = 0
    for i in range(n):
        c += i
        if i % 2 == 0:
            c += i
        else:
            c <<= 1
            break
    return c


# Parse function

flow = parse_bytecode(foo)
flow.render_dot()

# `flow.bbmap.graph` shows a nested structure

pprint(flow.bbmap.graph)


# ## Example 2: For-loop 2 level

def foo(n):
    c = 0
    for j in range(n):
        for i in range(i):
            c += i
            if i % 2 == 0:
                c += i
            else:
                c <<= 1
                break
    return c



flow = parse_bytecode(foo)
flow.render_dot()


