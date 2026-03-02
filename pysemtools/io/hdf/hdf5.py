""" Module that defines the hdf5 object to be used in pysemtools"""

import os
import h5py
import numpy as np
from mpi4py import MPI

class HDF5File:
    """
    Class to write and read hdf5 files in parallel using h5py.
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
        """
        
        self.comm = comm
        self.fname = fname
        if mode not in ["r", "w"]:
            raise ValueError("Mode should be 'read' or 'write'")
        else:
            self.mode = mode
        self.parallel = parallel

        if self.parallel:
            self.file = h5py.File(self.fname, self.mode, driver='mpio', comm=self.comm)
        else:
            self.file = h5py.File(self.fname, self.mode)

        # Some temporary variables to store
        self.global_shape = None
        self.local_shape = None
        self.offset = None
        self.count = None
        self.slices = None
        self.local_alloc_shape = None

    def close(self, clean: bool = False):
        """ Close the hdf5 file object """
        self.file.close()

        if clean:
            self.global_shape = None
            self.local_shape = None
            self.offset = None
            self.count = None
            self.slices = None        
            self.local_alloc_shape = None


    def read_dataset(self, dataset_name: str, dtype : np.dtype = np.double, distributed_axis: int = None, slices: list = None):
        """ Read a dataset from the hdf5 file object 
        """

        if self.mode != "r":
            raise ValueError("File is not opened in read mode")

        if self.parallel and distributed_axis is None:
            raise ValueError("Distributed axis must be specified for parallel reading")
                
        if slices is not None and len(slices) != self.file[dataset_name].ndim:
            raise ValueError("Number of slices must match the number of dimensions of the dataset")

        # ===========
        # Serial read 
        # ===========
        if not self.parallel:
            if slices is None:
                data = self.file[dataset_name][:]
            else:
                data = self.file[dataset_name][tuple(slices)]
        
        # =============
        # Parallel read 
        # =============
        else:
            if slices is None:
                self.set_read_slices(global_shape=self.file[dataset_name].shape, distributed_axis=distributed_axis, explicit_strides=False)
            else:
                # If the slices are explicitly provided, determine the local shape to allocate
                self.global_shape = self.file[dataset_name].shape
                
                # Local shape from slices
                local_array_shape = []
                for dim, slc in zip(self.global_shape, slices):
                    if isinstance(slc, slice):
                        start = 0 if slc.start is None else slc.start
                        stop = dim if slc.stop is None else slc.stop
                        local_array_shape.append(stop - start)

                # Set the attributes
                self.local_shape = tuple(local_array_shape)
                self.local_alloc_shape = self.local_shape
                self.slices = tuple(slices)

            local_data = np.empty(self.local_alloc_shape, dtype=dtype)
            local_data[:] = self.file[dataset_name][self.slices].reshape(self.local_shape)

        return local_data 
    
    def set_read_slices(self, global_shape: tuple, distributed_axis: int, explicit_strides: bool = False):
        """Set the slices that should be read from the file."""

        # Perform a load balanced distribution
        i_rank = self.comm.Get_rank()
        m = global_shape[distributed_axis]
        pe_rank = i_rank
        pe_size = self.comm.Get_size()
        ip = np.floor(
            (
                np.double(m)
                + np.double(pe_size)
                - np.double(pe_rank)
                - np.double(1)
            )
            / np.double(pe_size)
        )
        local_distributed_axis_shape = int(ip)
        #determine the offset and count to read
        offset = self.comm.scan(local_distributed_axis_shape) - local_distributed_axis_shape
        count = local_distributed_axis_shape

        # Update the offset and count to traverse the non distributed axes if explicit strides are used
        if explicit_strides:
            stride = 1
            for i in range(len(global_shape)):
                if i  != distributed_axis:
                    stride = stride * global_shape[i]
            offset = offset * stride
            count = count * stride

        # Determine the local shape of the array to be read
        local_shape = list(global_shape)
        local_shape[distributed_axis] = local_distributed_axis_shape
        local_shape = tuple(local_shape)

        # build the slices to read
        if explicit_strides:
            slices = [slice(None)]
            slices[distributed_axis] = slice(offset, offset + count)
            local_alloc_shape = (count,)
        else:
            slices = [slice(None)] * len(global_shape)
            slices[distributed_axis] = slice(offset, offset + count)
            local_alloc_shape = local_shape

        # Store the global shape in case this info can be reused
        self.global_shape = global_shape
        self.offset = offset
        self.count = count
        self.slices = tuple(slices)
        self.local_shape = local_shape
        self.local_alloc_shape = local_alloc_shape
