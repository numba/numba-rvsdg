=======================
Rendering API Reference
=======================

ByteFlowRenderer
================

.. currentmodule:: numba-rvsdg.rendering

.. contents::
   :local:
   :depth: 1


.. class:: ByteFlowRenderer

   The `ByteFlowRenderer`` class is used to render the visual representation of a `ByteFlow`` object

   .. method:: render_byteflow(byteflow: ByteFlow)

        Renders the provided `ByteFlow` object.

   .. method:: render_scfg(scfg: SCFG)

      Renders the provided `SCFG` object.
   
   .. method:: view(name: str)

      View the currently rendered object as an extenal graphviz document.

Utility Functions
=================

.. method:: render_flow(flow: ByteFlow)

    The `render_flow`` function takes a `flow` parameter as the `ByteFlow` to be transformed and rendered and performs the following operations:
        * Renders the pure `ByteFlow` representation of the function using `ByteFlowRenderer` and displays it as a document named "before".
        * Joins the return blocks in the `ByteFlow` object graph and renders the graph, displaying it as a document named "closed".
        * Restructures the loops recursively in the `ByteFlow` object graph and renders the graph, displaying it as named "loop restructured".
        * Restructures the branch recursively in the `ByteFlow` object graph and renders the graph, displaying it as named "branch restructured".
    
.. method:: render_func(func)

    The `render_func`` function takes a `func` parameter as the Python 
    function to be transformed and rendered and renders the byte flow 
    representation of the bytecode of the function. 
    Internally, it constucts a ByteFlow object using the bytecode of 
    the given function and calls `render_flow` on the generated 
    `ByteFlow` object.

.. method:: render_scfg(scfg: SCFG)

    The `render_scfg` function takes an `scfg` parameter as the `SCFG` 
    graph to be rendered and renders the Static Control Flow Graph (SCFG) 
    representation using `ByteFlowRenderer`, displaying it as a document 
    named  "scfg".

