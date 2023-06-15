=======================
Contributing Guidelines
=======================

numba-rvsdg originated to fulfill the needs of the Numba_ project.
It is maintained mostly by the Numba team. We tend to prioritize
the needs and constraints of Numba over other conflicting desires.

We do welcome any contributions in the form of Bug Reports or 
Pull Requests.

.. _Numba: http://numba.pydata.org/

.. contents::
   :local:
   :depth: 1

Communication methods
=====================

Forum
-----

numba-rvsdg uses the Numba Discourse as a forum for longer running threads such as
design discussions and roadmap planning. There are various categories available
and it can be reached at: `numba.discourse.group
<https://numba.discourse.group/>`_.

.. _report-bugs:

Bug reports
-----------

We use the
`Github issue tracker <https://github.com/numba/numba-rvsdg/issues>`_
to track both bug reports and feature requests. If you report an
issue, please include:

* What you are trying to do.

* Your operating system.

* What version of numba-rvsdg you are running.

* A description of the problem---for example, the full error
  traceback or the unexpected results you are getting.

* As far as possible, a code snippet that allows full
  reproduction of your problem.

.. _pull-requests:

Pull requests
-------------

To contribute code:

#. Fork our `Github repository <https://github.com/numba/numba-rvsdg>`_.

#. Create a branch representing your work.

#. When your work is ready, submit it as a pull request from the
   Github interface.


Development rules
=================

Coding conventions
------------------

* All Python code should follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_.
* Code and documentation should generally fit within 80 columns,
  for maximum readability with all existing tools, such as code
  review user interfaces.


Documentation
=============

This numba-rvsdg documentation is built using Sphinx and maintained
in the ``docs`` directory inside the
`numba-rvsdg repository <https://github.com/numba/numba-rvsdg>`_.

#. Edit the source files under ``docs/source/``.

#. Build the documentation::

     make html

#. Check the documentation::

     open build/html/index.html

.. |reg| unicode:: U+000AE .. REGISTERED SIGN
