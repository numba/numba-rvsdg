from byteflow2 import parse_bytecode
import logging
logging.basicConfig(level=logging.DEBUG)

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


# def foo(n):
#     c = 0
#     for j in range(n):
#         for i in range(i):
#             c += i
#             if i % 2 == 0:
#                 c += i
#             else:
#                 c <<= 1
#                 break
#     return c


def foo(i):
    c = 0


    if i % 2 == 0:
        c += i
        if i % 2 == 0:
            c += i
    else:
        c <<= 1

    c /= 10
    return c


flow = parse_bytecode(foo)
flow.render_dot().view()

assert False
"""
Next steps:

- add artificial ctrlflow nodes
    - avoid backedges
    - avoid conditional-branch head pointing from the start of branch subregion
"""