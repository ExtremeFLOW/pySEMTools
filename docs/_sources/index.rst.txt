.. PyNekTools documentation master file, created by
   sphinx-quickstart on Sat Jul 13 22:34:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyNekTools
==========

PyNekTools is a Python package for post-processing Spectral Element Method data for fluid simulations.

The following pages contain documentation for some of the most relevant classes and functions. However, some are missing.
For a complete list of classes and functions, please refer to the source code. Particularly for POD calculations.

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   ./pages/datatypes
   ./pages/interpolation
   ./pages/io

-----------------------

We beleive that the easiest way to get started with PyNekTools is to use the examples provided in the repository supported by the documentation.

Most examples are in jupyter notebooks, however those that can not generaly be run in series are in python scripts.

For your convenience, we compile the jupyter notebooks here. To run them, go directly to the example folder in the repository or
copy the cells you need into your python scripts.

There are more examples in the repository.

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   ./_notebooks/derivation.ipynb
   ./_notebooks/interpolation.ipynb
   ./_notebooks/pod.ipynb
   ./_notebooks/1d_data_processing.ipynb