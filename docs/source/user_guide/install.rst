======================
Setup and Installation
======================


Official numba-rvsdg packages are not yet available. You can manually install 
numba-rvsdg using the following steps:

Setting up the conda development environment
--------------------------------------------

The conda environment and all necessary dependencies can be setup using the following commands::

        conda env create -n numba-rvsdg python=3.11
        conda activate numba-rvsdg
        pip install pyyaml python-graphviz

.. note::
    At the time of writing pyyaml was not available for Python 3.11 via defaults so it had to be installed with pip.

Installation using pip
----------------------

After setting up the appropriate environment, a development version of the package can be installed as follows::

        git clone https://github.com/numba/numba-rvsdg.git
        cd numba-rvsdg/
        conda activate numba-rvsdg # or the enviroment that you've setup
        pip install -e .[dev]
