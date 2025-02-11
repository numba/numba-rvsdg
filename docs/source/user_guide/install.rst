======================
Setup and Installation
======================


Setting up the conda development environment
--------------------------------------------

The conda environment and all necessary dependencies can be setup using the following commands::

        conda env create -n numba-scfg python=3.11
        conda activate numba-scfg
        pip install pyyaml python-graphviz

.. note::
    At the time of writing pyyaml was not available for Python 3.11 via defaults so it had to be installed with pip.

Installation using pip
----------------------

Users can install numba-scfg using pip as follows::

        pip install numba-scfg

Alternatively, after setting up the appropriate environment, Users can also manually install numba-scfg using it's git repository.
A development version of the package can be installed as follows::

        git clone https://github.com/numba/numba-scfg.git
        cd numba-scfg/
        conda activate numba-scfg # or the enviroment that you've setup
        pip install -e .[dev]
