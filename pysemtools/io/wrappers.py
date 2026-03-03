"""Wrappers to ease IO"""

import sys
from typing import Union
import os
import h5py

from ..datatypes.msh import Mesh
from ..datatypes.field import FieldRegistry
from .ppymech.neksuite import pynekread, pynekwrite
import numpy as np
from mpi4py import MPI
from ..monitoring.logger import Logger
from .hdf.hdf5 import HDF5File
from .hdf.vtkhdf import VTKHDFFile

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

def partition_read_data(comm, fname: str = None, distributed_axis: int = 0):
    """
    Generate partition information for hdf5 files. Useful if needing to read or write multiple files with the same partitioning, such that the
    read/write functions does not need to do the same every time.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator
    fname : str
        The name of the file to read
    distributed_axis : int, optional
        The axis along which the data is distributed, by default 0. This is used to determine how many elements to read from the file in parallel.
    
    Returns
    -------
    list
        A list of slices corresponding to the local data to be read by each process
    """

    #raise NotImplementedError("Parallel IO is not implemented for hdf5 files")
    with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
        for key in f.keys()[0]:
            # Get the global array shape and sizes
            global_array_shape = f[key].shape

            # Determine how many axis zero elements to get locally
            # This corresponds to a linearly load balanced partitioning
            i_rank = comm.Get_rank()
            m = global_array_shape[distributed_axis]
            pe_rank = i_rank
            pe_size = comm.Get_size()
            ip = np.floor(
                (
                    np.double(m)
                    + np.double(pe_size)
                    - np.double(pe_rank)
                    - np.double(1)
                )
                / np.double(pe_size)
            )
            local_axis_0_shape = int(ip)
            #determine the offset
            offset = comm.scan(local_axis_0_shape) - local_axis_0_shape

            # Determine the local array shape
            temp = list(global_array_shape)
            temp[distributed_axis] = local_axis_0_shape
            local_array_shape = tuple(temp)
            
            # Get the slice of the data that should be read
            slices = [slice(None) for i in range(len(global_array_shape))]
            slices[distributed_axis] = slice(offset, offset + local_array_shape[distributed_axis])     

    return slices

def read_data(comm, fname: str, keys: list[str], parallel_io: bool = False, dtype = np.single, distributed_axis: int = 0, slices: list = None):
    """
    Read data from a file and return a dictionary with the names of the files and keys

    Parameters
    ----------
    comm, MPI.Comm
        The MPI communicator
    fname : str
        The name of the file to read
    keys : list[str]
        The keys to read from the file
    parallel_io : bool, optional
        If True, read the file in parallel, by default False. This is aimed for hdf5 files, and currently it does not work if True
    dtype : np.dtype, optional
        The data type of the data to read, by default np.single
    distributed_axis : int, optional
        The axis along which the data is distributed, by default 0. This is used to determine how many elements to read from the file in parallel.
    slices : list, optional
        A list of slices to read from the file. If None, the local data will be read based on the distributed_axis and the communicator. If provided, it should match the number of dimensions in the data.
        Note that if you are reading in parallel, the slices should be provided in such a way that they correspond to the local data on each process, otherwise the data will be replicated.
        If in doubt, do not provide slices, and the local data will be determined automatically.

    Returns
    -------
    dict
        A dictionary with the keys and the data read from the file
    """

    # Check the file extension
    path = os.path.dirname(fname)
    prefix = os.path.basename(fname).split('.')[0]
    extension = os.path.basename(fname).split('.')[1]

    # Read the data
    if extension == 'hdf5':
        
        file = HDF5File(comm, fname, "r", parallel_io)
        data = {}
        for key in keys:
            data[key] = file.read_dataset(key, dtype=dtype, distributed_axis=distributed_axis)
        file.close()
 
    elif extension[0] == 'f':
        
        data = {}
        msh = None
        fld = FieldRegistry(comm)

        # Go through the keys
        for key in keys:
            # If the mesh must be read, create a mesh object
            if key in ["x", "y", "z"]:
                # Read the mesh only once
                if msh is None:
                    msh = Mesh(comm)
                    pynekread(fname, comm, data_dtype=dtype, msh=msh)
                if key == "x":
                    data[key] = np.copy(msh.x)
                elif key == "y":
                    data[key] = np.copy(msh.y)
                elif key == "z":
                    data[key] = np.copy(msh.z)
            else:
                # Read the field
                fld.add_field(comm, field_name=key, file_type="fld", file_name=fname, file_key=key, dtype=dtype)
                data[key] = np.copy(fld.registry[key])
                fld.clear() 

    elif extension == 'vtkhdf':        
        
        file = VTKHDFFile(comm, fname, "r", parallel_io)
        data = {}
        for key in keys:
            data[key] = file.read_point_data(key, dtype=dtype, distributed_axis=distributed_axis)
            print(data[key].shape)
        file.close()


    return data 

def write_data(comm, fname: str, data_dict: dict[str, np.ndarray], parallel_io: bool = False, dtype = np.single, msh: Union[Mesh, list[np.ndarray]] = None, write_mesh:bool=False, distributed_axis: int = 0, uniform_shape: bool = False):
    """
    Write data to a file

    Parameters
    ----------
    comm, MPI.Comm
        The MPI communicator
    fname : str
        The name of the file to write
    data_dict : dict
        The data to write to the file
    parallel_io : bool, optional
        If True, write the file in parallel, by default False. This is aimed for hdf5 files, and currently it does not work if True
    dtype : np.dtype, optional
    msh : Mesh, optional
        The mesh object to write to a fld file, by default None
    write_mesh : bool, optional
        Only valid for writing fld files
    distributed_axis : int, optional
        The axis along which the data is distributed, by default 0
    uniform_shape : bool, optional
        If True, the global shape of the data is assumed to be uniform, by default False
    """

    if uniform_shape:
        raise NotImplementedError("Uniform shape was deprecated for now.")

    # Check the file extension
    path = os.path.dirname(fname)
    prefix = os.path.basename(fname).split('.')[0]
    extension = os.path.basename(fname).split('.')[1]

    # Write the data
    if (extension == 'hdf5') or (extension == 'h5'):

        file = HDF5File(comm, fname, "w", parallel_io)

        if write_mesh:
            file.write_dataset("x", msh[0], dtype=dtype, distributed_axis=distributed_axis)
            file.write_dataset("y", msh[1], dtype=dtype, distributed_axis=distributed_axis)
            file.write_dataset("z", msh[2], dtype=dtype, distributed_axis=distributed_axis)

        for key in data_dict.keys():
            file.write_dataset(key, data_dict[key].astype(dtype), distributed_axis=distributed_axis)
        
        file.close()

    elif extension[0] == 'f':
        if msh is None:
            raise ValueError("A mesh object must be provided to write a fld file")
        elif isinstance(msh, list):
            msh = Mesh(comm, x = msh[0], y = msh[1], z = msh[2], create_connectivity=False)
        
        # Write the data to a fld file
        fld = FieldRegistry(comm)
        for key in data_dict.keys():
            fld.add_field(comm, field_name=key, field=data_dict[key], dtype=dtype)

        if dtype == np.single:
            wdsz = 4
        elif dtype == np.double:
            wdsz = 8

        pynekwrite(fname, comm, msh=msh, fld=fld, wdsz=wdsz, write_mesh=write_mesh)

    # Write the data
    elif (extension == 'vtkhdf'):

        if msh is None:
            raise ValueError("A mesh object must be provided to write a vtkhdf file")
        
        file = VTKHDFFile(comm, fname, "w", parallel_io)
        file.write_mesh_data({"x": msh[0], "y": msh[1], "z": msh[2]}, distributed_axis=distributed_axis)

        for key in data_dict.keys():
            file.write_point_data(key, data_dict[key].astype(dtype), distributed_axis=distributed_axis)

        file.close()

    else:
        raise ValueError("The file extension is not supported")

    return