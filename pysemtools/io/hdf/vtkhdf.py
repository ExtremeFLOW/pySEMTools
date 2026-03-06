""" Module that defines the hdf5 object to be used in pysemtools"""

import os
import h5py
import numpy as np
from mpi4py import MPI
from ...monitoring.logger import Logger
from .hdf5 import HDF5File
from ...datatypes import VTKMesh

class VTKHDFFile(HDF5File):
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
            Mode to open the file. Should be "r" for reading or "w" for
            writing.
        parallel : bool
            Whether to use parallel I/O or not. 
        """
        super().__init__(comm, fname, mode, parallel)

        self.shape = None
        self.mesh_fname = None
        
        # Write the headers if we are writing
        self.set_active_group("/VTKHDF")
        if self.mode == "w":
            write_headers(self.active_group)
    
    def close(self, clean: bool = True):
        """ Close the hdf5 file object 
        
        Parameters
        ----------
        clean : bool
            Whether to clean up the file after closing. This will delete the file from disk. Should only be used for testing.
        """
        super().close(clean)
        self.shape = None
        self.mesh_fname = None

    def read_mesh_data(self, dtype : np.dtype = np.double, distributed_axis: int = 0):
        """ Read the mesh data from the hdf5 file
        
        Parameters
        ----------
        dtype : np.dtype
            Data type to read the mesh data as. Should be a floating point type.
        distributed_axis : int
            Axis along which the data is distributed in parallel. Should be 0 for now.
        
        Returns
        -------
        x : np.ndarray
            The x coordinates of the mesh points.
        y : np.ndarray
            The y coordinates of the mesh points.
        z : np.ndarray
            The z coordinates of the mesh points.
        """

        if distributed_axis != 0:
            raise NotImplementedError("Distributed axis other than 0 is not implemented for the read_mesh_data function")

        # Read the mesh data
        self.set_active_group("/VTKHDF")
        points = self.read_dataset("Points", dtype=dtype, distributed_axis=distributed_axis, as_array_list_in_file=True)
        
        return points[...,0], points[...,1], points[...,2]

    def write_mesh_data(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, distributed_axis: int = 0):
        """ Write the mesh data to the hdf5 file
        
        Parameters
        ----------
        x : np.ndarray
            The x coordinates of the mesh points.
        y : np.ndarray
            The y coordinates of the mesh points.
        z : np.ndarray
            The z coordinates of the mesh points.
        distributed_axis : int
            Axis along which the data is distributed in parallel. Should be 0 for now.
        """

        if distributed_axis != 0:
            raise NotImplementedError("Distributed axis other than 0 is not implemented for the write_mesh_data function")

        # ========================
        # Set up the vtk mesh data
        # ========================
        vtk = VTKMesh(self.comm, x, y, z, cell_type="hex", global_connectivity=self.parallel, distributed_axis=distributed_axis)

        # Write the mesh metadata
        self.set_active_group("/VTKHDF")
        self.active_group.create_dataset("NumberOfPoints", data=vtk.data.pop("NumberOfPoints"), dtype="i8")
        self.active_group.create_dataset("NumberOfCells", data=vtk.data.pop("NumberOfCells"), dtype="i8")
        nconectivity_ids = vtk.data["NumberOfConnectivityIds"][0]
        self.active_group.create_dataset("NumberOfConnectivityIds", data=vtk.data.pop("NumberOfConnectivityIds"), dtype="i8")

        # Save the shape of the mesh for later use
        self.shape = vtk.data.pop("shape")

        # Write the mesh data
        mesh_shape = [axs for axs in self.shape] + [3]
        self.write_dataset("Points", vtk.data.pop("Points"), distributed_axis=distributed_axis, shape_in_ram=mesh_shape)
        self.write_dataset("Connectivity", vtk.data.pop("Connectivity"), distributed_axis=distributed_axis)
        self.write_dataset("Types", vtk.data.pop("Types"), distributed_axis=distributed_axis)
        extra_global_entries = [1] if self.parallel else [0]
        self.write_dataset("Offsets", vtk.data.pop("Offsets"), distributed_axis=distributed_axis, extra_global_entries=extra_global_entries)
        self.active_group["Offsets"][-1] = nconectivity_ids

        self.mesh_fname = self.fname

    def link_to_existing_mesh(self, mesh_name: str):
        """ Link to an existing mesh
        
        Avoid rewriting mesh data if not necessary. It can be quite costly in storage.

        Parameters
        ----------
        mesh_name : str
            Name of the hdf5 file to link to. 
        """

        if mesh_name == "" and self.mesh_fname is None:
            raise ValueError("Mesh name must be provided if mesh name is not set in the file")
        elif mesh_name == "":
            mesh_name = self.mesh_fname

        # Get the field global shape from the mesh
        if self.shape is None:
            temp = HDF5File(self.comm, mesh_name, "r", self.parallel)
            mesh_shape = temp.file["/VTKHDF/Points"].attrs["shape"]
            self.shape = tuple(mesh_shape[:-1]) # Remove the last dimension which is the coordinate dimension
            temp.close()

        self.set_active_group("/VTKHDF")
        self.active_group["NumberOfPoints"] = h5py.ExternalLink(mesh_name, "/VTKHDF/NumberOfPoints")
        self.active_group["NumberOfCells"] = h5py.ExternalLink(mesh_name, "/VTKHDF/NumberOfCells")
        self.active_group["NumberOfConnectivityIds"] = h5py.ExternalLink(mesh_name, "/VTKHDF/NumberOfConnectivityIds")
        self.active_group["Points"] = h5py.ExternalLink(mesh_name, "/VTKHDF/Points")
        self.active_group["Connectivity"] = h5py.ExternalLink(mesh_name, "/VTKHDF/Connectivity")
        self.active_group["Types"] = h5py.ExternalLink(mesh_name, "/VTKHDF/Types")
        self.active_group["Offsets"] = h5py.ExternalLink(mesh_name, "/VTKHDF/Offsets")


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

    def read_point_data(self, dataset_name : str, dtype : np.dtype = np.double, distributed_axis: int = 0):
        """ Read point data from the hdf5 file
        
        Parameters
        ----------
        dataset_name : str
            Name of the dataset to read. This should be the name of the dataset in the hdf5 file.
        dtype : np.dtype
            Data type to read the dataset as. Should be a floating point type.
        distributed_axis : int
            Axis along which the data is distributed in parallel. Should be 0 for now.
        
        Returns
        -------
        np.ndarray
            The point data read from the hdf5 file. Will have the same shape as the mesh points.
        """

        if distributed_axis != 0:
            raise NotImplementedError("Distributed axis other than 0 is not implemented for the read_point_data function")

        # Read the point data
        self.set_active_group("/VTKHDF/PointData")
        data = self.read_dataset(dataset_name, dtype=dtype, distributed_axis=distributed_axis, as_array_list_in_file=True)

        return data


def write_headers(root):
    root.attrs["Version"] = (2, 3)
    root.attrs["Type"] = "UnstructuredGrid"