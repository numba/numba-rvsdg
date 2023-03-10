# # Demo Byteflow2
#
# Note: This file is written using the _light_ script format of 
#       jupytext and it is intended to be executed as a notebook.

import logging
from pprint import pprint
from byteflow2 import ByteFlow, ByteFlowRenderer


logging.basicConfig(level=logging.DEBUG)

# ## Example: simple branch

#def foo(a, b):
#    if b == 52:
#        return 5
#    else:
#        return 6
#
#def foo(n):
#    for i in range(n):
#        pass

#def foo(n):
#    c = 0
#    for i in range(n):
#        c += 1
#        if i == 100:
#            break
#    return c

# Example: nested loop
#def foo(n, m):
#    c = 0
#    for i in range(n):
#        c += i
#        for j in range(m):
#            for k in range(z):
#                c += i
#    return c

# Example: for loop with exit
#def foo(n, m):
#    c = 0
#    for i in range(n):
#        c += i
#        if c > 100:
#            c += 5
#            return c
#    return c

# Example for loop with multiple backedges
#def foo(n, c):
#    for i in range(n):
#        if i == 42:
#            c += 1
#        elif i == 52:
#            c += 2
#        else:
#            c += 42

# ## Example: for loop with second exit

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
#
# ## Example: multiple backedges
#
#def foo(n):
#    c = 0
#    for i in range(n):
#        c += i
#        if i % 2 == 0:
#            c += i
#        elif i % 3 == 0:
#            c += 2
#        else:
#            c <<= 1
#            #break
#    return c
#
# ## Example: For-loop in for loop
#
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
# ## Example: For-loop in for loop with more conditions
#
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

# ## Example: For-loop in a nested condition

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
# ## Example: Andre's example
#
#def foo(x, y):
#    for i in range(100):
#        y += 1
#    if x == 0:
#        return y + 3
#    elif x > 0:
#        return y + 2
#    else:
#        return y + 1
#
# ## Example: Simple loop
#
#def foo():
#    for i in range(100):
#        print(i)


# ## Stuart's switch example
#def foo(a):
#    b = 0
#    match a:
#        case ["quit"]:
#            b += 1
#        case ["look"]:
#            b += 2
#        case ["get"]:
#            b += 3
#        case ["go"]:
#            b += 4
#    return b

#def foo(x):
#    c = 0
#    for i in range(x):
#        c += i
#        if c <= 0:
#           continue 
#        else:
#            for j in range(c):
#                c += j
#                if c > 100:
#                    break
#    return c



#def foo():
#    for i in range(10):
#        with call_context:
#            print(i)
#            if i == 5:
#                print("A")
#                break
#    return i

#def foo(x):
#    c = 0
#    for i in range(x):
#        c += i
#        for j in range(x):
#            c += j
#            if c > 100:
#                break
#    return c

#def foo(x):
#    c = 0
#    for i in range(x):
#        c += i
#        if i == 100:
#            break
#    return c
#
def foo(s, e):
    i = s
    c = 0
    while i < e:
        c += i
        i += 1
    return c

flow = ByteFlow.from_bytecode(foo)
ByteFlowRenderer().render_byteflow(flow).view("before")

cflow = flow._join_returns()
ByteFlowRenderer().render_byteflow(cflow).view("closed")

lflow = cflow._restructure_loop()
ByteFlowRenderer().render_byteflow(lflow).view("loop restructured")

bflow = lflow._restructure_branch()
ByteFlowRenderer().render_byteflow(bflow).view("branch restructured")
