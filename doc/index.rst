.. PySEMTools documentation master file, created by
   sphinx-quickstart on Sat Jul 13 22:34:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PySEMTools
==========

PySEMTools is a Python package for post-processing Spectral Element Method data for fluid simulations.

The following pages contain documentation for some of the most relevant classes and functions. However, some are missing.
For a complete list of classes and functions, please refer to the source code. Particularly for POD calculations.

.. toctree::
   :maxdepth: 1
   :caption: User Guide:

   ./pages/datatypes
   ./pages/interpolation
   ./pages/io

-----------------------

We beleive that the easiest way to get started with PySEMTools is to use the examples provided in the repository supported by the documentation.

Most examples are in jupyter notebooks, however those that can not generaly be run in series are in python scripts.

For your convenience, we compile the jupyter notebooks here. To run them, go directly to the example folder in the repository or
copy the cells you need into your python scripts.

There are more examples in the repository.

.. toctree::
   :maxdepth: 1
   :caption: Examples on data types and IO:

   ./_notebooks/1-datatypes_io.ipynb
   ./_notebooks/2-sem_subdomains.ipynb
   ./_notebooks/3-data_compression.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples on calculus on SEM mesh:

   ./_notebooks/1-differentiation.ipynb
   ./_notebooks/2-integration.ipynb

.. toctree::
   :maxdepth: 1
   :caption: Examples on interpolation:

   ./_notebooks/1-interpolation_to_query_points.ipynb
   ./_notebooks/2-element_interpolation.ipynb 
   ./_notebooks/3-interpolation_from_2d_sem_mesh.ipynb
   ./_notebooks/5-structured_mesh.ipynb 
   ./_notebooks/6-interpolating_file_sequences.ipynb
   ./_notebooks/7-visualizing_pointclouds.ipynb


.. toctree::
   :maxdepth: 1
   :caption: Examples on reduced order models:

   ./_notebooks/1-POD_from_pointclouds.ipynb
   ./_notebooks/2-POD_fft_from_pointclouds.ipynb
   

.. toctree::
   :maxdepth: 1
   :caption: Examples on statistics:
   
   
   ./_notebooks/1-post_processing_mean_fields.ipynb
   ./_notebooks/2-UQ_of_velocity.ipynb
   ./_notebooks/3-UQ_of_temperature.ipynb
   
   
   
   
   
   