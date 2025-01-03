InputOutput (I/O)
-----------------

IO contains classes and functions to read and write SEM data.

The main modules are:

    - :doc:`ppymech`
    - :doc:`adios2`

We have developed wrapper functions that can be used to read and write data in a more user-friendly way. Here we show them:

------------------

.. automodule :: pynektools.io.wrappers
    :members:

Additionally, there exist a class to read probes written in the csv format. Here we show it:

.. autoclass :: pynektools.io.read_probes.ProbesReader
    :members:
    :exclude-members: __weakref__ __dict__
