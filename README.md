# numba-rvsdg

Numba compatible RVSDG (Regionalized Value State Dependence Graph)  utilities.

## about

This repository contains Numba compatible utilities for working with RVSDGs
(Regionalized Value State Dependency Graphs). RVSDGs are a type of
Intermediary Representation (IR) suitable for regularizing Python bytecode
within Numba.

The code in this repository is an implementation of the CFG restructuring
algorithms in Bahmann2015, specifically those from section 4.1 and 4.2: namely
"loop restructuring" and "branch restructuring". These are interesting for
Numba because they serve to clearly identify regions withing the Python
bytecode.

## dependencies

* Python 3.11
* graphviz
* pyyaml

As of 2023-03-06 you can create a conda env using the following:

```
conda env create -n numba-rvsdg python=3.11 python-graphviz
conda activate numba-rvsdg
pip install pyyaml
```

At the time of writing `pyyaml` was not available for Python 3.11 via
`defaults` so it had to be installed with `pip`.

## overview

The following files are included in this repository:

* byteflow2.py -- the algorithms from Bahmann2015
* example.py -- file for running and displaying examples
* scc.py -- strongly connected components, copied verbatim from networkx
* simulator.py -- a CFG simulator, for testing
* test_byteflow2.py -- tests for byteflow2 algorithms
* test_fig3.py -- test for fig3 from Bahmann2015
* test_fig4.py -- test for fig4 from Bahmann2015
* test_simulate.py -- simulator based tests
* testing.py -- more tests for byteflow2, should probably be merged

## example

The following will process the given example function and display the four
different stages. "initial" is the unprocessed bytecode as produced by
cpython. "closed" is simply the closed variant of the initial CFG. "loop
restructuring" is the loop-restructured version and "branch-restructured" is
the final form which includes closing, loop-restructuring and
branch-restructuring.


```python
# Example: for loop with branch and early exit
def foo(n):
    c = 0
    for i in range(n):
        c += 1
        if i == 100:
            break
    return c

flow = ByteFlow.from_bytecode(foo)
ByteFlowRenderer().render_byteflow(flow).view("initial")

cflow = flow._join_returns()
ByteFlowRenderer().render_byteflow(cflow).view("closed")

lflow = cflow._restructure_loop()
ByteFlowRenderer().render_byteflow(lflow).view("loop restructured")

bflow = lflow._restructure_branch()
ByteFlowRenderer().render_byteflow(bflow).view("branch restructured")
```

![initial](docs/images/initial.png "initial")
![closed](docs/images/closed.png "closed")
![loop-restructured](docs/images/loop_restructured.png "loop-restructured")
![branch-restructured](docs/images/branch_restructured.png "branch-restructured")

## references

* `Reismann2020` -- https://arxiv.org/pdf/1912.05036.pdf -- Describes the concept of RVSDGs
* `Bahmann2015` -- https://dl.acm.org/doi/pdf/10.1145/2693261 -- Describes the transformation
  algorithms implemented

## license

Copyright (c) 2022, Anaconda, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


