# numba-scfg

Numba compatible SCFG (Structured Control Flow Graph)  utilities.

## About

This repository contains Numba compatible utilities for working with SCFG
(Structured Control FLow Graphs). SCFGs are a type of Intermediary
Representation (IR) suitable for regularizing Python source and bytecode.

The code in this repository is an implementation of the CFG restructuring
algorithms in Bahmann 2015, specifically those from section 4.1 and 4.2: namely
"loop restructuring" and "branch restructuring". These are interesting for
Numba because they serve to clearly identify regions within the Python source
and bytecode.

Note: The project was previously known as `numba-rvsdg` and was renamed to
`numba-scfg` in early 2025. The original scope was to implement Regional Value
State Dependence Graphs (RVSDG), where SCFGs are necessary intermediary for the
transformation from Python to RVSDG. Over time, it became evident that the SCFG
transformations are important and reusable enough in their own right to warrant
their own package and so this package was renamed.


## Development

If you have `make` and `conda` available, a common setting up workflow could
be:

```
$ make conda-env                        # setup conda environment
$ conda activate numba-scfg             # activate it
$ make conda-install                    # install dependencies
$ make                                  # lint, build and test the project
```

Feel free to look at the
[`makefile`](https://github.com/numba/numba-rvsdg/blob/main/makefile) for all
development commands.

## References

* `Bahmann2015` -- https://dl.acm.org/doi/pdf/10.1145/2693261 -- Describes the transformation
  algorithms implemented

## License

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


