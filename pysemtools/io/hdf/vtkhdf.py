""" Module that defines the hdf5 object to be used in pysemtools"""

import os
import h5py
import numpy as np
from mpi4py import MPI
from ...monitoring.logger import Logger
from .hdf5 import HDF5File

class VTKHDF(HDF5File):
    """
    Class to write and read vtkhdf files in parallel using h5py.
    """

    def __init__(self, comm : MPI.Comm, fname: str, mode: str, parallel: bool):
        """ Initialize the hdf5 file object
        
        Open an hdf5 file based on inputs.

        Parameters
        ----------
        comm : MPI.Comm
            MPI communicator to use for parallel I/O. If parallel is False, this can be set
            to None.
        fname : str
            Name of the hdf5 file to read or write.
        mode : str
            Mode to open the file. Should be "read" for reading or "write" for
            writing.
        parallel : bool
            Whether to use parallel I/O or not. 
        mesh : dict
            Dictionary containing the mesh information. Should contain the keys "x", "y", and "z" for the coordinates of the mesh points.
        """
        super().__init__(comm, fname, mode, parallel)

        self.shape = None
        
        # Write the headers if we are writing
        self.set_active_group("/VTKHDF")
        if self.mode == "w":
            write_headers(self.active_group)

    def write_mesh_data(self, mesh: dict, distributed_axis: int = 0):
        """ Write the mesh data to the hdf5 file
        
        Parameters
        ----------
        mesh : dict
            Dictionary containing the mesh information. Should contain the keys "x", "y", and "z" for the coordinates of the mesh points.
        """

        if distributed_axis != 0:
            raise NotImplementedError("Distributed axis other than 0 is not implemented for the write_mesh_data function")

        # =================
        # Serial vtk set up
        # =================
        if not self.parallel:
            vtk_data = set_up_hex_vtk_mesh_serial(mesh["x"], mesh["y"], mesh["z"])
        else:
            vtk_data = set_up_hex_vtk_mesh_parallel(self.comm, mesh["x"], mesh["y"], mesh["z"], distributed_axis)

        # Write the mesh metadata
        self.set_active_group("/VTKHDF")
        self.active_group.create_dataset("NumberOfPoints", data=vtk_data.pop("NumberOfPoints"), dtype="i8")
        self.active_group.create_dataset("NumberOfCells", data=vtk_data.pop("NumberOfCells"), dtype="i8")
        nconectivity_ids = vtk_data["NumberOfConnectivityIds"][0]
        self.active_group.create_dataset("NumberOfConnectivityIds", data=vtk_data.pop("NumberOfConnectivityIds"), dtype="i8")

        # Save the shape of the mesh for later use
        self.shape = vtk_data.pop("shape")

        # Write the mesh data
        self.write_dataset("Points", vtk_data.pop("Points"), distributed_axis=distributed_axis)
        self.write_dataset("Connectivity", vtk_data.pop("Connectivity"), distributed_axis=distributed_axis)
        self.write_dataset("Types", vtk_data.pop("Types"), distributed_axis=distributed_axis)
        extra_global_entries = [1] if self.parallel else [0]
        self.write_dataset("Offsets", vtk_data.pop("Offsets"), distributed_axis=distributed_axis, extra_global_entries=extra_global_entries)
        self.active_group["Offsets"][-1] = nconectivity_ids

    def write_point_data(self, dataset_name : str, data: np.ndarray, distributed_axis: int = 0):
        """ Write point data to the hdf5 file
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to write. This will be used as the name of the dataset in the hdf5 file.
        data : np.ndarray
            Data to write. Should have the same number of points as the mesh.
        distributed_axis : int
            Axis along which the data is distributed in parallel. Should be 0 for now.
        """
        if self.shape is None:
            raise ValueError("Mesh data must be written before writing point data")

        if distributed_axis != 0:
            raise NotImplementedError("Distributed axis other than 0 is not implemented for the write_point_data function")

        # Write the point data
        self.set_active_group("/VTKHDF/PointData")
        self.write_dataset(dataset_name, data.flatten(), distributed_axis=distributed_axis, shape_in_ram=self.shape)

def set_up_hex_vtk_mesh_serial(X : np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """Set up the mesh in serial"""

    if X.ndim != 3 or Y.ndim != 3 or Z.ndim != 3:
        raise ValueError("X, Y, and Z should be 3D arrays")

    # Get the point list        
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    n_pts_x = X.shape[0]
    n_pts_y = X.shape[1]
    n_pts_z = X.shape[2]
    n_cell_x = n_pts_x - 1
    n_cell_y = n_pts_y - 1
    n_cell_z = n_pts_z - 1
    n_cell_local = n_cell_x * n_cell_y * n_cell_z

    VTK_HEXAHEDRON = 12
    VTK_CELL_POINTS = 8
    types = np.full(n_cell_local, VTK_HEXAHEDRON, dtype=np.uint8)

    connectivity = []
    for i in range(n_cell_x):
        for j in range(n_cell_y):
            for k in range(n_cell_z):
                v0 = ijk_to_id(i,   j,   k,   n_pts_y, n_pts_z)
                v1 = ijk_to_id(i+1, j,   k,   n_pts_y, n_pts_z)
                v2 = ijk_to_id(i+1, j+1, k,   n_pts_y, n_pts_z)
                v3 = ijk_to_id(i,   j+1, k,   n_pts_y, n_pts_z)
                v4 = ijk_to_id(i,   j,   k+1, n_pts_y, n_pts_z)
                v5 = ijk_to_id(i+1, j,   k+1, n_pts_y, n_pts_z)
                v6 = ijk_to_id(i+1, j+1, k+1, n_pts_y, n_pts_z)
                v7 = ijk_to_id(i,   j+1, k+1, n_pts_y, n_pts_z)
                connectivity.extend([v0, v1, v2, v3, v4, v5, v6, v7])

    connectivity = np.asarray(connectivity, dtype=np.int64)   # length 8*ncells
    offsets = np.arange(0, VTK_CELL_POINTS*n_cell_local + 1, VTK_CELL_POINTS, dtype=np.int64)   # length ncells+1

    NumberOfPoints = (points.shape[0],)
    NumberOfCells = (n_cell_local,)
    NumberOfConnectivityIds = (connectivity.size,)

    return {"points": points, "connectivity": connectivity, 
            "offsets": offsets, "types": types, 
            "NumberOfPoints": NumberOfPoints, 
            "NumberOfCells": NumberOfCells, 
            "NumberOfConnectivityIds": NumberOfConnectivityIds,
            "shape": (n_pts_x, n_pts_y, n_pts_z)}

def set_up_hex_vtk_mesh_parallel(comm: MPI.Comm, X : np.ndarray, Y: np.ndarray, Z: np.ndarray, distributed_axis: int):
    """Set up the mesh in parallel"""
    
    if X.ndim != 3 or Y.ndim != 3 or Z.ndim != 3:
        raise ValueError("X, Y, and Z should be 3D arrays")
    
    if distributed_axis != 0:
        raise NotImplementedError("Distributed axis other than 0 is not implemented for the set_up_hex_vtk_mesh_parallel function")

    # Get the point list           
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    # Determine the number of points and cells locally 
    n_pts_x = X.shape[0]
    n_pts_y = X.shape[1]
    n_pts_z = X.shape[2]
    
    # For the distributed axis, the number of cells is n_pts_total - 1. The last rank is the one that "loses" one point
    # This happenes because we do not have overlapping boundaries across ranks. That would make it easier.
    n_cell_x = n_pts_x - 1 if comm.Get_rank() == comm.Get_size() - 1 else n_pts_x
    n_cell_y = n_pts_y - 1
    n_cell_z = n_pts_z - 1
    n_cell_local = n_cell_x * n_cell_y * n_cell_z

    # Determine global and offsets for nells
    n_pts_global = comm.allreduce(points.shape[0], op=MPI.SUM)
    n_pts_offset = comm.scan(n_pts_x) - n_pts_x
    n_pts_x_global = comm.allreduce(n_pts_x, op=MPI.SUM)
    n_cell_global = comm.allreduce(n_cell_local, op=MPI.SUM)

    # Set up array with types of cells
    VTK_HEXAHEDRON = 12
    VTK_CELL_POINTS = 8
    types = np.full(n_cell_local, VTK_HEXAHEDRON, dtype=np.uint8)

    # Set up connectivity and offsets 
    connectivity = []
    for i in range(n_cell_x):
        for j in range(n_cell_y):
            for k in range(n_cell_z):
                v0 = ijk_to_id_mpi(i,   j,   k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v1 = ijk_to_id_mpi(i+1, j,   k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v2 = ijk_to_id_mpi(i+1, j+1, k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v3 = ijk_to_id_mpi(i,   j+1, k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v4 = ijk_to_id_mpi(i,   j,   k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v5 = ijk_to_id_mpi(i+1, j,   k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v6 = ijk_to_id_mpi(i+1, j+1, k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v7 = ijk_to_id_mpi(i,   j+1, k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                connectivity.extend([v0, v1, v2, v3, v4, v5, v6, v7])

    connectivity = np.asarray(connectivity, dtype=np.int64)   # length 8*ncells
    conn_start = comm.scan(connectivity.size) - connectivity.size
    offsets_local = (np.arange(0, VTK_CELL_POINTS*n_cell_local + 1, VTK_CELL_POINTS, dtype=np.int64) + conn_start)
    offsets = offsets_local[:-1]   # length = n_cell_total - Total offsets need to be n_cell_total + 1, but we fix later

    NumberOfPoints = (n_pts_global,)
    NumberOfCells = (n_cell_global,)
    NumberOfConnectivityIds = (comm.allreduce(connectivity.size, op=MPI.SUM),)

    return {"points": points, "connectivity": connectivity, 
            "offsets": offsets, "types": types, 
            "NumberOfPoints": NumberOfPoints, 
            "NumberOfCells": NumberOfCells, 
            "NumberOfConnectivityIds": NumberOfConnectivityIds,
            "shape": (n_pts_x_global, n_pts_y, n_pts_z)}
    
def ijk_to_id(i, j, k, ny, nz):
    return i*(ny*nz) + j*nz + k

def ijk_to_id_mpi(i, j, k, ny, nz, parallel_axis, offset):
    if parallel_axis == 0:
        i = i + offset
        return i*(ny*nz) + j*nz + k
    else:
        raise NotImplementedError("Parallel axis other than 0 is not implemented for the ijk_to_id_mpi function")

def write_headers(root):
    root.attrs["Version"] = (2, 3)
    root.attrs["Type"] = "UnstructuredGrid"