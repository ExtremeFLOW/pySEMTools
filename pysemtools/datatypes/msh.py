""" Module that contains msh class, which contains relevant data on the domain"""

import numpy as np
from ..monitoring.logger import Logger
from .element_slicing import fetch_elem_facet_data as fd
from .element_slicing import fetch_elem_edge_data as ed
from .element_slicing import fetch_elem_vertex_data as vd

NoneType = type(None)


class Mesh:
    """
    Class that contains coordinate and partitioning data of the domain.

    This class needs to be used generaly as it contains the coordinates of the domain and
    some information about the partitioning of the domain.

    Parameters
    ----------
    comm : Comm
        MPI comminicator object.
    data : HexaData, optional
        HexaData object that contains the coordinates of the domain.
    x : ndarray, optional
        X coordinates of the domain. shape is (nelv, lz, ly, lx).
    y : ndarray, optional
        Y coordinates of the domain. shape is (nelv, lz, ly, lx).
    z : ndarray, optional
        Z coordinates of the domain. shape is (nelv, lz, ly, lx).
    create_connectivity : bool, optional
        If True, the connectivity of the domain will be created. (Memory intensive).

    Attributes
    ----------
    x : ndarray
        X coordinates of the domain. shape is (nelv, lz, ly, lx).
    y : ndarray
        Y coordinates of the domain. shape is (nelv, lz, ly, lx).
    z : ndarray
        Z coordinates of the domain. shape is (nelv, lz, ly, lx).
    lx : int
        Polynomial degree in x direction.
    ly : int
        Polynomial degree in y direction.
    lz : int
        Polynomial degree in z direction.
    nelv : int
        Number of elements in the domain in current rank.
    glb_nelv : int
        Total number of elements in the domain.
    gdim : int
        Dimension of the domain.
    non_linear_shared_points : list, optional
        List that show the index where the points in the domain are shared, used by coef in dssum.

    Returns
    -------

    Examples
    --------
    If a hexadata object: data is read from disk, the mesh object can be created directly from it.

    >>> from pysemtools.datatypes.msh import Mesh
    >>> msh = Mesh(comm, data = data)

    If the coordinates are already available, the mesh object can be created from them.

    >>> from pysemtools.datatypes.msh import Mesh
    >>> msh = Mesh(comm, x = x, y = y, z = z)

    This is useful in situations where the coordinates are generated in the code or streamed into python from another source.
    """

    def __init__(
        self, comm, data=None, x=None, y=None, z=None, create_connectivity=True
    ):

        self.log = Logger(comm=comm, module_name="Mesh")
        self.create_connectivity_bool = create_connectivity

        if not isinstance(data, NoneType):
            self.init_from_data(comm, data)

        elif (
            not isinstance(x, NoneType)
            and not isinstance(y, NoneType)
            and not isinstance(z, NoneType)
        ):
            self.init_from_coords(comm, x, y, z)

        else:
            self.log.write("info", "Initializing empty Mesh object.")

    def init_from_data(self, comm, data):
        """
        Initialize form data.

        This function is used to initialize the mesh object from a hexadata object.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.
        data : HexaData
            HexaData object that contains the coordinates of the domain.

        Returns
        -------
        None
            Nothing is returned, the attributes are set in the object.
        """
        self.log.tic()
        self.log.write("info", "Initializing Mesh object from HexaData object.")

        self.x, self.y, self.z = get_coordinates_from_hexadata(data)

        self.init_common(comm)

        self.log.write("info", "Mesh object initialized.")
        self.log.write("info", f"Mesh data is of type: {self.x.dtype}")
        self.log.toc()

    def init_from_coords(self, comm, x, y, z):
        """
        Initialize from coordinates.

        This function is used to initialize the mesh object from x, y, z ndarrays.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.
        x : ndarray
            X coordinates of the domain. shape is (nelv, lz, ly, lx).
        y : ndarray
            Y coordinates of the domain. shape is (nelv, lz, ly, lx).
        z : ndarray
            Z coordinates of the domain. shape is (nelv, lz, ly, lx).

        Returns
        -------
        None
            Nothing is returned, the attributes are set in the object.
        """

        self.log.tic()
        self.log.write("info", "Initializing Mesh object from x,y,z ndarrays.")

        self.x = x
        self.y = y
        self.z = z

        self.init_common(comm)

        self.log.write("info", "Mesh object initialized.")
        self.log.write("info", f"Mesh data is of type: {self.x.dtype}")
        self.log.toc()

    def init_common(self, comm):
        """
        Initialize common attributes.

        This function is used to initialize the common attributes of the mesh object.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.

        Returns
        -------
        None
            Nothing is returned, the attributes are set in the object.
        """

        self.log.write("info", "Initializing common attributes.")

        self.lx = np.int64(
            self.x.shape[3]
        )  # This is not an error, the x data is on the last index
        self.ly = np.int64(
            self.x.shape[2]
        )  # This is not an error, the x data is on the last index
        self.lz = np.int64(
            self.x.shape[1]
        )  # This is not an error, the x data is on the last index
        self.lxyz = np.int64(self.lx * self.ly * self.lz)
        self.nelv = np.int64(self.x.shape[0])

        self.log.write("debug", "Performing MPI scan")
        # Find the element offset of each rank so you can store the global element number
        nelv = self.x.shape[0]
        sendbuf = np.ones((1), np.int64) * nelv
        recvbuf = np.zeros((1), np.int64)
        comm.Scan(sendbuf, recvbuf)
        self.offset_el = recvbuf[0] - nelv

        self.log.write("debug", "Getting global number of elements")
        # Find the total number of elements
        sendbuf = np.ones((1), np.int64) * self.nelv
        recvbuf = np.zeros((1), np.int64)
        comm.Allreduce(sendbuf, recvbuf)
        self.glb_nelv = recvbuf[0]

        if self.lz > 1:
            self.gdim = 3
        else:
            self.gdim = 2

        self.get_vertices()

        self.get_edge_centers()

        self.get_facet_centers()

        self.create_connectivity()

        self.global_element_number = np.arange(
            self.offset_el, self.offset_el + self.nelv, dtype=np.int64
        )

    def get_vertices(self):
        '''
        Get the vertices of the domain.
        
        Get all the vertices of the domain in 2D or 3D.

        Notes
        -----

        We need 4 vertices for 2D and 8 vertices for 3D. For all cases,
        we store 3 coordinates for each vertex.
        '''

        self.log.write("info", "Getting vertices")

        if self.gdim == 2:
            self.vertices = np.zeros((self.nelv, 4, 3), dtype=self.x.dtype) # 4 vertices, 3 coords (z = 0)

            for vertex in range(0, 4):
                self.vertices[:, vertex, 0] = vd(field=self.x, vertex=vertex)
                self.vertices[:, vertex, 1] = vd(field=self.y, vertex=vertex)
                self.vertices[:, vertex, 2] = 0

        elif self.gdim == 3:
            self.vertices = np.zeros((self.nelv, 8, 3), dtype=self.x.dtype) # 4 vertices, 3 coords

            for vertex in range(0, 8):
                self.vertices[:, vertex, 0] = vd(field=self.x, vertex=vertex)
                self.vertices[:, vertex, 1] = vd(field=self.y, vertex=vertex)
                self.vertices[:, vertex, 2] = vd(field=self.z, vertex=vertex)

    def get_edge_centers(self):
        '''
        Get the edge centers of the domain.
        
        Get all the edge centers of the domain in 2D or 3D.

        Notes
        -----

        We need 4 edges for 2D and 12 edges for 3D. For all cases
        we store 3 coordinates for each edge.
        '''

        self.log.write("info", "Getting edge centers")

        if self.gdim == 2:
            self.edge_centers = np.zeros((self.nelv, 4, 3), dtype=self.x.dtype) # 4 vertices, 3 coords (z = 0)

            for edge in range(0, 4):
                edge_data = ed(field=self.x, edge=edge)
                self.edge_centers[:, edge, 0] = np.min(edge_data, axis=(1)) + (np.max(edge_data, axis=(1)) - np.min(edge_data, axis=(1))) / 2
                edge_data = ed(field=self.y, edge=edge)
                self.edge_centers[:, edge, 1] = np.min(edge_data, axis=(1)) + (np.max(edge_data, axis=(1)) - np.min(edge_data, axis=(1))) / 2
                edge_data = ed(field=self.z, edge=edge)
                self.edge_centers[:, edge, 2] = np.min(edge_data, axis=(1)) + (np.max(edge_data, axis=(1)) - np.min(edge_data, axis=(1))) / 2            
                
        elif self.gdim == 3:
            self.edge_centers = np.zeros((self.nelv, 12, 3), dtype=self.x.dtype) # 4 vertices, 3 coords
            
            for edge in range(0, 12):
                edge_data = ed(field=self.x, edge=edge)
                self.edge_centers[:, edge, 0] = np.min(edge_data, axis=(1)) + (np.max(edge_data, axis=(1)) - np.min(edge_data, axis=(1))) / 2
                edge_data = ed(field=self.y, edge=edge)
                self.edge_centers[:, edge, 1] = np.min(edge_data, axis=(1)) + (np.max(edge_data, axis=(1)) - np.min(edge_data, axis=(1))) / 2
                edge_data = ed(field=self.z, edge=edge)
                self.edge_centers[:, edge, 2] = np.min(edge_data, axis=(1)) + (np.max(edge_data, axis=(1)) - np.min(edge_data, axis=(1))) / 2            

    def get_facet_centers(self):
        '''
        Get the centroid of each facet
        
        Find the "centroid of each facet. This is used to find the shared facets between elements.

        Notes
        -----

        This is not really the centroid, as we also find a coordinate in the dimension perpendicular to the facet.
        This means that these values can be outside or inside the element. However the same behaviour should be seen in the matching elements. 
        '''

        if self.gdim == 2:
            self.log.write("info", "Facet centers not available for 2D")

        elif self.gdim == 3:
            self.log.write("info", "Getting facet centers")

            self.facet_centers = np.zeros((self.nelv, 6, 3), dtype=self.x.dtype) # 6 facets, 3 coordinates

            # Facet 1 - 6 -> 0 - 5
            for facet in range(0, 6):
                facet_data = fd(field=self.x, facet=facet)
                self.facet_centers[:, facet, 0] = np.min(facet_data, axis=(1,2)) + (np.max(facet_data, axis=(1,2)) - np.min(facet_data, axis=(1,2))) / 2
                facet_data = fd(field=self.y, facet=facet)
                self.facet_centers[:, facet, 1] = np.min(facet_data, axis=(1,2)) + (np.max(facet_data, axis=(1,2)) - np.min(facet_data, axis=(1,2))) / 2
                facet_data = fd(field=self.z, facet=facet)
                self.facet_centers[:, facet, 2] = np.min(facet_data, axis=(1,2)) + (np.max(facet_data, axis=(1,2)) - np.min(facet_data, axis=(1,2))) / 2            


    def create_connectivity(self):

        if self.create_connectivity_bool:

            self.log.write("info", "Creating connectivity")

            if self.lz > 1:
                z_ind = [0, self.lz - 1]
            else:
                z_ind = [0]
            self.coord_hash_to_shared_map = dict()
            for e in range(0, self.nelv):

                # loop through all faces (3 loops required)

                for k in z_ind:
                    for j in range(0, self.ly):
                        for i in range(0, self.lx):
                            point = (
                                self.x[e, k, j, i],
                                self.y[e, k, j, i],
                                self.z[e, k, j, i],
                            )
                            point = hash(point)
                            if point in self.coord_hash_to_shared_map:
                                # self.coord_hash_to_shared_map[point].append((e, k, j, i))
                                self.coord_hash_to_shared_map[point].append(
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                )
                            else:
                                # self.coord_hash_to_shared_map[point] = [(e, k, j, i)]
                                self.coord_hash_to_shared_map[point] = [
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                ]

                for j in [0, self.ly - 1]:
                    for k in range(self.lz):
                        for i in range(self.lx):
                            point = (
                                self.x[e, k, j, i],
                                self.y[e, k, j, i],
                                self.z[e, k, j, i],
                            )
                            point = hash(point)
                            if point in self.coord_hash_to_shared_map:
                                # self.coord_hash_to_shared_map[point].append((e, k, j, i))
                                self.coord_hash_to_shared_map[point].append(
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                )
                            else:
                                # self.coord_hash_to_shared_map[point] = [(e, k, j, i)]
                                self.coord_hash_to_shared_map[point] = [
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                ]

                for i in [0, self.lx - 1]:
                    for k in range(self.lz):
                        for j in range(self.ly):
                            point = (
                                self.x[e, k, j, i],
                                self.y[e, k, j, i],
                                self.z[e, k, j, i],
                            )
                            point = hash(point)
                            if point in self.coord_hash_to_shared_map:
                                # self.coord_hash_to_shared_map[point].append((e, k, j, i))
                                self.coord_hash_to_shared_map[point].append(
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                )
                            else:
                                # self.coord_hash_to_shared_map[point] = [(e, k, j, i)]
                                self.coord_hash_to_shared_map[point] = [
                                    linear_index(i, j, k, e, self.lx, self.ly, self.lz)
                                ]


def linear_index(i, j, k, l, lx, ly, lz):
    """
    Map 4d index to 1d.

    This is used to represent the domain as a list that can be used in search trees.

    Parameters
    ----------
    i : int
        Index in x direction.
    j : int
        Index in y direction.
    k : int
        Index in z direction.
    l : int
        Index of the element.
    lx : int
        Polynomial degree in x direction.
    ly : int
        Polynomial degree in y direction.
    lz : int
        Polynomial degree in z direction.

    Returns
    -------
    int
        1d index of the 4d index.
    """
    return i + lx * ((j - 0) + ly * ((k - 0) + lz * ((l - 0))))


def nonlinear_index(linear_index_, lx, ly, lz):
    """
    Map 1d index to 4d

    This is an inverse of linear index.

    Parameters
    ----------
    linear_index_ : list
        List of 1d linear indices.
    lx : int
        Polynomial degree in x direction.
    ly : int
        Polynomial degree in y direction.
    lz : int
        Polynomial degree in z direction.
    Returns
    -------
    list
        List of 4d non linear indices correspoinf to the linear indices.
    """
    indices = []
    for list_ in linear_index_:
        index = np.zeros(4, dtype=int)
        lin_idx = list_
        index[3] = lin_idx / (lx * ly * lz)
        index[2] = (lin_idx - (lx * ly * lz) * index[3]) / (lx * ly)
        index[1] = (lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2]) / lx
        index[0] = (
            lin_idx - (lx * ly * lz) * index[3] - (lx * ly) * index[2] - lx * index[1]
        )
        ind = (index[3], index[2], index[1], index[0])
        indices.append(ind)

    return indices


def get_coordinates_from_hexadata(data):
    """
    Get the coordinates from a hexadata object in mesh format.

    Used to go from a hexadata object to a ndarray that can be used for operations.

    Parameters
    ----------
    data : HexaData
        HexaData object that contains the coordinates of the domain.

    Returns
    -------
    x : ndarray
        X coordinates of the domain. shape is (nelv, lz, ly, lx).
    y : ndarray
        Y coordinates of the domain. shape is (nelv, lz, ly, lx).
    z : ndarray
        Z coordinates of the domain. shape is (nelv, lz, ly, lx).
    """

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    x = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].pos.dtype)
    y = np.zeros_like(x)
    z = np.zeros_like(x)

    for e in range(0, nelv):
        x[e, :, :, :] = data.elem[e].pos[0, :, :, :]
        y[e, :, :, :] = data.elem[e].pos[1, :, :, :]
        z[e, :, :, :] = data.elem[e].pos[2, :, :, :]

    return x, y, z
