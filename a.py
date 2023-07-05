from rich import print
from numba_rvsdg.core.datastructures.byte_flow import ByteFlow


def foo(a):
    sz = len(a)
    s = 0
    if sz > 5:
        for i in range(sz):
            s += sz
        s += sz
    else:
        s -= sz
    s += 1
    return s


a = [1, 2, 3, 4]
flow = ByteFlow.from_bytecode(foo)
print(flow)
flow.scfg.view()


# class Animal:

#     i: int

#     def __init__(self, j):
#         pass

#     def __str__(self):
#         return f"Animal: {self.i}"


# print(Animal(3))
