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

NmshMesh
----

Descriptions of the contents of the NmshMesh class, the in-memory form of a Neko ``.nmsh`` mesh file: corner
vertices with global point ids, boundary zones and curved edges, distributed over the ranks like Mesh.
See :doc:`nmsh` for the tools built on it.

.. autoclass :: pysemtools.datatypes.nmsh.NmshMesh
    :members:
    :exclude-members: __weakref__ __dict__

The straight-sided geometry of an NmshMesh at the GLL points, used by ``NmshMesh.to_sem_mesh``:

.. automodule :: pysemtools.datatypes.nmsh_geometry
    :members:

MeshPartitioner
----

Descriptions of the contents of the MeshPartitioner class.
This allows to redistribute elements among ranks based on a partitioning algorithm.

.. autoclass :: pysemtools.datatypes.msh_partitioning.MeshPartitioner
    :members:
    :exclude-members: __weakref__ __dict__

