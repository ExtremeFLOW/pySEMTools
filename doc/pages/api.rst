API
--------

PySEMTools is a collection of modules that together provide a comprehensive toolkit for working with spectral element method (SEM) data.

The presently included submodules are:

.. autosummary::
    :toctree: generated

    pysemtools.cli
    pysemtools.comm
    pysemtools.datatypes
    pysemtools.interpolation
    pysemtools.io.ppymech
    pysemtools.io.adios2
    pysemtools.io.wrappers
    pysemtools.monitoring
    pysemtools.rom

The compression module inside the package is still considered to be experimental and is prone to change, therefore its documentation is not yet included here.

The post-processing module is in a similar state. This module intends to use the tools to perform more specific post processing tasks. It is still under development and its API is prone to change, therefore 
the documentation is reserved for a future release.

The full API documentation for all the classes and functions is provided below.

-----------------------

cli
~~~

.. automodule :: pysemtools.cli
    :members:

comm
~~~~

.. automodule :: pysemtools.comm
    :members:

datatypes
~~~~~~~~~

.. automodule :: pysemtools.datatypes
    :members:

interpolation
~~~~~~~~~~~~~

.. automodule :: pysemtools.interpolation
    :members:

io
~~

There are multiple interfaces to perform IO operations. The main one relates to NEk5000/Neko files. These are read by using the ppymech submodule. 
It is also possible to use ADIOS2 to, for example, use the in-situ processing capabilities, like data streaming or compression. 
Finally, a set of wrappers to read and write data in Nek or HDF5 format is also provided.


ppymech
^^^^^^^

.. automodule :: pysemtools.io.ppymech
    :members:

ADIOS2
^^^^^^

.. automodule :: pysemtools.io.adios2
    :members:

wrappers
^^^^^^^^

.. automodule :: pysemtools.io.wrappers
    :members:

monitoring
~~~~~~~~~~

.. automodule :: pysemtools.monitoring
    :members:

rom
~~~

.. automodule :: pysemtools.rom
    :members:

