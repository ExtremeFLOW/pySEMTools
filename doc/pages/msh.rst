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

Re2Mesh
----

Descriptions of the contents of the Re2Mesh class, the in-memory form of a NEKTON ``.re2`` mesh file: element
corners with group ids, curved edges and boundary conditions, distributed over the ranks like Mesh.

.. autoclass :: pysemtools.datatypes.re2.Re2Mesh
    :members:
    :exclude-members: __weakref__ __dict__

Both mesh file types share the behaviour of their base class:

.. autoclass :: pysemtools.datatypes.corner_mesh.CornerMesh
    :members:
    :exclude-members: __weakref__ __dict__

The straight-sided geometry of the element corners at the GLL points, used by ``to_sem_mesh``:

.. automodule :: pysemtools.datatypes.corner_mesh_geometry
    :members:

MeshPartitioner
----

Descriptions of the contents of the MeshPartitioner class.
This allows to redistribute elements among ranks based on a partitioning algorithm.

.. autoclass :: pysemtools.datatypes.msh_partitioning.MeshPartitioner
    :members:
    :exclude-members: __weakref__ __dict__

