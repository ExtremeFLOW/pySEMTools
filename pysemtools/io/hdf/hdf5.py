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

        # Assign the communicator and assign empty attributes        
        self.comm = comm
        
        # Set attributes that are assigned when opening a file
        self.mode = None
        self.parallel = None
        self.file = None
        self.fname = None

        # Some temporary variables to store
        self.global_shape = None
        self.local_shape = None
        self.offset = None
        self.count = None
        self.slices = None
        self.local_alloc_shape = None

        # Open a file
        self.open(fname, mode, parallel)

    def open(self, fname: str, mode: str, parallel: bool):
        """ Open an hdf5 file based on inputs. This can be used to open a new file after closing the previous one."""
        
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
            # Set slices
            if slices is None:
                self.set_read_slices_linear_lb(global_shape=self.file[dataset_name].shape, distributed_axis=distributed_axis, explicit_strides=False)
            else:
                self.set_read_slices_external(global_shape=self.file[dataset_name].shape, slices=slices)
            
            local_data = self.read_slices(dataset_name, dtype=dtype)

        return local_data
    
    def set_read_slices_linear_lb(self, global_shape: tuple, distributed_axis: int, explicit_strides: bool = False):
        """Set the slices that should be read from the file.

        Data is distributed in a linear load balanced way.
        """
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
    
    def set_read_slices_external(self, global_shape: tuple, slices: list):
        """Set the slices that should be read from the file based on external input."""

        # Local shape from slices
        local_array_shape = []
        for dim, slc in zip(global_shape, slices):
            if isinstance(slc, slice):
                start = 0 if slc.start is None else slc.start
                stop = dim if slc.stop is None else slc.stop
                local_array_shape.append(stop - start)

        # Set the attributes
        self.global_shape = global_shape
        self.offset = None
        self.count = None
        self.slices = tuple(slices)
        self.local_shape = tuple(local_array_shape)
        self.local_alloc_shape = self.local_shape 

    def read_slices(self, dataset_name: str, dtype : np.dtype = np.double):
        """Read the slices that should be read from the file. This is useful if the slices have been set using set_read_slices and we want to read the same slices again."""
        if self.slices is None:
            raise ValueError("Slices have not been set")
        if self.local_alloc_shape is None:
            raise ValueError("Local allocation shape is not set")
        if self.local_shape is None:
            raise ValueError("Local shape is not set")

        local_data = np.empty(self.local_alloc_shape, dtype=dtype)
        local_data[:] = self.file[dataset_name][self.slices]

        return local_data.reshape(self.local_shape)

    def write_dataset(self, dataset_name: str, data: np.ndarray, distributed_axis: int = None):
        """ Write a dataset to the hdf5 file object 
        """

        if self.mode != "w":
            raise ValueError("File is not opened in write mode")

        if self.parallel and distributed_axis is None:
            raise ValueError("Distributed axis must be specified for parallel writing")

        # ============
        # Serial write 
        # ============
        if not self.parallel:
            self.file.create_dataset(dataset_name, data=data, dtype=data.dtype)

        # ==============
        # Parallel write 
        # ==============
        else:
            # Set slices 
            self.set_write_slices(local_shape=data.shape, distributed_axis=distributed_axis, extra_global_entries=None)
    
            # Write the slices
            self.write_slices(dataset_name, data)
 
    def set_write_slices(self, local_shape: tuple, distributed_axis: int, extra_global_entries: list[int] = None):
        """Set the slices that should be written to the file."""

        # Set the local shape
        local_distributed_axis_shape = local_shape[distributed_axis]

        # Determine offset and count to write
        offset = self.comm.scan(local_distributed_axis_shape) - local_distributed_axis_shape
        count = local_distributed_axis_shape 
        
        # Determine the global shape of the array
        global_distributed_axis_shape = self.comm.allreduce(local_distributed_axis_shape, op=MPI.SUM)
        global_shape = list(local_shape)
        global_shape[distributed_axis] = global_distributed_axis_shape
        if extra_global_entries is not None:
            for i, extra in enumerate(extra_global_entries):
                global_shape[i] += extra
        global_shape = tuple(global_shape)
        
        # Determine the slices where to write
        slices = [slice(None)] * len(local_shape)
        slices[distributed_axis] = slice(offset, offset + count)

        # Store the info in the attributes
        self.global_shape = global_shape
        self.offset = offset
        self.count = count
        self.slices = tuple(slices)
        self.local_shape = local_shape
        self.local_alloc_shape = None
        
    def write_slices(self, dataset_name: str, data: np.ndarray):
        """Write the slices that should be written to the file. This is useful if the slices have been set using set_write_slices and we want to write the same slices again."""
        if self.slices is None:
            raise ValueError("Slices have not been set")
        if self.global_shape is None:
            raise ValueError("Global shape is not set")

        dset = self.file.create_dataset(dataset_name, shape=self.global_shape, dtype=data.dtype)
        dset[self.slices] = data
            