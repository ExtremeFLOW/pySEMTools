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
        if parallel_io:
            if distributed_axis != 0:
                raise NotImplementedError("Parallel axis other than 0 is not implemented for reading vtkhdf files in parallel")
            
            with h5py.File(fname, 'r', driver='mpio', comm=comm) as f:
                data = {}
                for key in keys:

                    global_array_shape = tuple(f["VTKHDF"]["PointData"][key].attrs["global_shape"])
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

                    # The offset is now in number of rows (or first axis). But it needs to be into the flattened array
                    if distributed_axis == 0:
                        offset = offset * (global_array_shape[1] * global_array_shape[2])
                    else:
                        raise NotImplementedError("Parallel axis other than 0 is not implemented for reading vtkhdf files in parallel")

                    # Determine the local array shape
                    temp = list(global_array_shape)
                    temp[distributed_axis] = local_axis_0_shape
                    local_array_shape = tuple(temp)
                    
                    # Get the slice of the data that should be read
                    slices = [slice(None)]
                    if distributed_axis == 0:
                        slices[distributed_axis] = slice(offset, offset + local_array_shape[distributed_axis]*(global_array_shape[1] * global_array_shape[2]))
                    else:
                        raise NotImplementedError("Parallel axis other than 0 is not implemented for reading vtkhdf files in parallel")

                    
                    # Allocate the local data array
                    local_data = np.empty(local_array_shape, dtype=dtype)
                    local_data = local_data.reshape(-1)

                    # Read the data
                    local_data[:] = f["VTKHDF"]["PointData"][key][tuple(slices)] 
                    
                    # Store the data
                    data[key] = local_data.reshape(local_array_shape)

        else:
            with h5py.File(fname, 'r') as f:
                data = {}
                for key in keys:
                    temp_data = f["VTKHDF"]["PointData"][key][:]
                    if "global_shape" in f["VTKHDF"]["PointData"][key].attrs:
                        original_shape = f["VTKHDF"]["PointData"][key].attrs["global_shape"]
                        temp_data = temp_data.reshape(original_shape)

                    data[key] = temp_data


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

        if parallel_io:
            
            # ===================
            # Set up vtk
            # ===================

            # Set up points list 
            if msh is None:
                raise ValueError("A mesh object must be provided to write a vtkhdf file in parallel")
            else:  
                X = msh[0]
                Y = msh[1]
                Z = msh[2]
                
                if len(X.shape) != 3 or len(Y.shape) != 3 or len(Z.shape) != 3:
                    raise ValueError("The mesh provided must be a structured mesh with 3D coordinates to write a vtkhdf file in parallel")
                
                points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

            # Determine the number of points and cells locally 
            n_pts_x = X.shape[0]
            n_pts_y = X.shape[1]
            n_pts_z = X.shape[2]
            # For the distributed axis, the number of cells is n_pts_total - 1. The last rank is the one that "loses" one point
            # This happenes because we do not have overlapping boundaries across ranks. That would make it easier.
            if distributed_axis == 0:
                n_cell_x = n_pts_x - 1 if comm.Get_rank() == comm.Get_size() - 1 else n_pts_x
            else:
                raise NotImplementedError("Parallel axis other than 0 is not implemented for writing vtkhdf files")
            n_cell_y = n_pts_y - 1
            n_cell_z = n_pts_z - 1
            n_cell_local = n_cell_x * n_cell_y * n_cell_z

            # Determine global and offsets for nells
            if distributed_axis == 0: # Only for distirbuted axis 0 we get the same behaviour that flattens() do in the global array
                n_pts_global = comm.allreduce(points.shape[0], op=MPI.SUM)
                n_pts_offset = comm.scan(n_pts_x) - n_pts_x
                n_pts_x_global = comm.allreduce(n_pts_x, op=MPI.SUM)
            else:
                raise NotImplementedError("Parallel axis other than 0 is not implemented for writing vtkhdf files")
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

            # ================
            # Write data
            # ================
            with h5py.File(fname, 'w', driver='mpio', comm=comm) as f:
                root = f.create_group("VTKHDF")
                write_headers(root)

                # Number of points, cells and conecitivity ids.                
                root.create_dataset("NumberOfPoints", data=(n_pts_global,), dtype="i8")
                root.create_dataset("NumberOfCells", data=(n_cell_global,), dtype="i8")
                root.create_dataset("NumberOfConnectivityIds", data=(comm.allreduce(connectivity.size, op=MPI.SUM),), dtype="i8")

                # Data to write is the actual data, plus vtk related data
                keys = ["Connectivity", "Offsets", "Types", "Points"] + [key for key in data_dict.keys()]
                dsets = [connectivity, offsets, types, points] + [data_dict[key].flatten() for key in data_dict.keys()]


                point_data = root.create_group("PointData")                
                # When writing in this format, we always allocate a list of (npts, 3) so we always distribute along axis 0
                # Even if the data itself had a different distribution
                distributed_axis = 0  
                # Write
                for key, data in zip(keys, dsets):
                    
                    # Determine local sizes
                    local_array_shape = data.shape
                    local_array_size = data.size

                    # Obtain the total number of entries in the array    
                    global_axis_0_shape = np.array(comm.allreduce(local_array_shape[distributed_axis], op=MPI.SUM))
                    # Obtain the offset in the file
                    offset = comm.scan(local_array_shape[distributed_axis]) - local_array_shape[distributed_axis]
                    
                    # Set the global size
                    temp = list(local_array_shape)
                    if key == "Offsets": 
                        global_axis_0_shape += 1 # For offsets we need one more entry than the number of cells
                    temp[distributed_axis] = global_axis_0_shape
                    global_array_shape = tuple(temp)

                    # Get the slice where the data should be written
                    slices = [slice(None) for i in range(len(global_array_shape))]
                    slices[distributed_axis] = slice(offset, offset + local_array_shape[distributed_axis])
 
                    # Create the data set and assign the data
                    if key in ["Connectivity", "Offsets", "Types", "Points"]:
                        dset = root.create_dataset(key, global_array_shape, dtype=data.dtype)
                        dset[tuple(slices)] = data
                        if key == "Offsets": # Offsets is n_cell_total + 1. The last rank writes the last entry (here we let them all write it)
                            root["Offsets"][n_cell_global] = comm.allreduce(connectivity.size, op=MPI.SUM)
                    else:
                        dset = point_data.create_dataset(key, global_array_shape, dtype=data.dtype)
                        dset[tuple(slices)] = data

                        # Write out the original shape of the array    
                        global_shape = []
                        for i in range((len(data_dict[key].shape))):
                            if i == distributed_axis:
                                global_shape.append(-1)
                            else:
                                global_shape.append(data_dict[key].shape[i])
                        dset.attrs["global_shape"] = (n_pts_x_global, n_pts_y, n_pts_z)
                     
        else:

            X = msh[0]
            Y = msh[1]
            Z = msh[2]
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

            with h5py.File(fname, "w") as file:
                root = file.create_group("VTKHDF")
                write_headers(root)

                root.create_dataset("NumberOfPoints", data=(points.shape[0],), dtype="i8")
                root.create_dataset("Points", data=points, dtype="f8")

                root.create_dataset("NumberOfCells", data=(n_cell_local,), dtype="i8")
                root.create_dataset("Types", data=types, dtype="uint8")

                root.create_dataset("NumberOfConnectivityIds", data=(connectivity.size,), dtype="i8")
                root.create_dataset("Connectivity", data=connectivity, dtype="i8")
                root.create_dataset("Offsets", data=offsets, dtype="i8")

                point_data = root.create_group("PointData")
                for key in data_dict.keys():
                    dset = point_data.create_dataset(key, data=(data_dict[key].flatten()), dtype=data_dict[key].dtype)    
                    dset.attrs["global_shape"] = (n_pts_x, n_pts_y, n_pts_z)

    else:
        raise ValueError("The file extension is not supported")

    return