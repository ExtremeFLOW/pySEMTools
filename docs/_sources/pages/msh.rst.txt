:orphan:

Mesh
----

Descriptions of the contents of the Mesh class, which contains the coordinates of the domain and aditional suporting data.

.. autoclass :: pysemtools.datatypes.msh.Mesh
    :members:
    :exclude-members: __weakref__ __dict__

MeshConnectivity
----

Descriptions of the contents of the MeshConnectivity class. This class determines the connectivity from the geometry and can be used to perform parallel operations like dssum.

.. autoclass :: pysemtools.datatypes.msh_connectivity.MeshConnectivity
    :members:
    :exclude-members: __weakref__ __dict__

MeshPartitioner
----

Descriptions of the contents of the MeshPartitioner class.
This allows to redistribute elements among ranks based on a partitioning algorithm.

.. autoclass :: pysemtools.datatypes.msh_partitioning.MeshPartitioner
    :members:
    :exclude-members: __weakref__ __dict__

