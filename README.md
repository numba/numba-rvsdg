# numba-rvsdg

Numba compatible RVSDG (Regionalized Value State Dependence Graph)  utilities.

## about

This repository contains Numba compatible utilities for working with RVSDGs
(Regionalized Value State Dependency Graphs). RVSDGs are a type of
Intermediary Representation (IR) suitable for regularizing Python bytecode
within Numba.

The code in this repository is an implementation of the CFG restructuring
algorithms in Bahmann 2015, specifically those from section 4.1 and 4.2: namely
"loop restructuring" and "branch restructuring". These are interesting for
Numba because they serve to clearly identify regions within the Python
bytecode.

## dependencies

* Python 3.11 or 3.12
* make (optional, build tool)
* graphviz
* pyyaml
* pytest (for testing)
* sphinx (for docs)
* sphinx_rt_theme (for docs)

You can create a conda env using the following:

```
$ conda env create -n numba-rvsdg python=3.12 python-graphviz pyyaml pytest sphinx sphinx_rtd_theme
$ conda activate numba-rvsdg
```

If you have `make` and `conda` available, a common workflow could be:

```
$ make conda-env                        # setup conda environment
$ conda activate numba-rvsdg            # activate it
$ make                                  # lint, build and test the project
```

Feel free to look at the `makefile` for low-level commands.

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


