# numba-rvsdg

Numba compatible RVSDG (Regionalized Value State Dependence Graph)  utilities.

## about

This repository contains Numba compatible utilities for working with RVSDGs
(Regionalized Value State Dependency Graphs). RVSDGs are a type of
Intermiediary Representation (IR) suitable for regularizing Python byetcode
within Numba.

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

## references

* `Reismann2020` -- https://arxiv.org/pdf/1912.05036.pdf -- Describes the concept of RVSDGs
* `Bahmann2015` -- https://dl.acm.org/doi/pdf/10.1145/2693261 -- Descibes the transformation
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


