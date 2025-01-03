:orphan:

Mesh
----

Descriptions of the contents of the Mesh class, which contains the coordinates of the domain and aditional suporting data.

.. autoclass :: pynektools.datatypes.msh.Mesh
    :members:
    :exclude-members: __weakref__ __dict__

MeshConnectivity
----

Descriptions of the contents of the MeshConnectivity class. This class determines the connectivity from the geometry and can be used to perform parallel operations like dssum.

.. autoclass :: pynektools.datatypes.msh_connectivity.MeshConnectivity
    :members:
    :exclude-members: __weakref__ __dict__

MeshPartitioner
----

Descriptions of the contents of the MeshPartitioner class.
This allows to redistribute elements among ranks based on a partitioning algorithm.

.. autoclass :: pynektools.datatypes.msh_partitioning.MeshPartitioner
    :members:
    :exclude-members: __weakref__ __dict__

