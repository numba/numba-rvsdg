.. numba-rvsdg documentation master file, created by
   sphinx-quickstart on Tue May 16 17:22:12 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

===========
numba-rvsdg
===========

:emphasis:`Numba compatible RVSDG (Regionalized Value State Dependence Graph) utilities.`

This repository contains Numba_ compatible utilities for working with RVSDGs 
(Regionalized Value State Dependency Graphs). RVSDGs are a type of Intermediary 
Representation (IR) suitable for regularizing Python bytecode within Numba_.

The code in this repository is an implementation of the CFG restructuring 
algorithms in Bahmann2015, specifically those from section 4.1 and 4.2: namely 
"loop restructuring" and "branch restructuring". These are interesting for Numba_
because they serve to clearly identify regions withing the Python bytecode.


.. _Numba: http://numba.pydata.org/

Philosophy
==========

The transformational algorithms LOOP-RESTRUCTURE and BRANCH-RESTRUCTURE as
described in Bahmann2015 are to be implemented in this repository. The idea is
to use these algorithms to regularize Python bytecode such that it becomes
“structured”. Using the terms of the paper, to take an existing Control Flow
Graph (CFG) and restructure it into a Structured Control Flow Graph (SCFG). The
algorithms consist of two parts conceptually: restructuring the CFG such that
regions (Loop, Head, Branch and Tail) can be identified clearly. That is to say
there is “data” in the form of the blocks of a CFG and analysis in the form or
regions that are “overlayed” on top of the data.

API stability
=============

.. warning::
   The API currently being documented is still experimental and
   under heavy development. It is subject to potential changes,
   which as a result, may change the API's behavior, structure,
   and functionality suddenly without prior notice.

.. toctree::
   :caption: Documentation Overview
   :maxdepth: 1
   :hidden:

   user_guide/index.rst
   tutorials/index.rst
   reference/index.rst   
   faqs
   contributing
   release_notes

.. |reg| unicode:: U+000AE .. REGISTERED SIGN
