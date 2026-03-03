""" Module that defines the hdf5 object to be used in pysemtools"""

import os
import h5py
import numpy as np
from mpi4py import MPI
from ...monitoring.logger import Logger

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
        self.log = Logger(comm=comm, module_name="HDF5File")
        
        # Set attributes that are assigned when opening a file
        self.mode = None
        self.parallel = None
        self.file = None
        self.active_group = None
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
 
    def set_active_group(self, group_name: str):
        """ Set the active group to read or write data from. This is useful to avoid having to specify the group every time a dataset is read or written."""
        
        group_path = group_name.split("/")
        group_path = [name for name in group_path if name != ""] # Remove empty strings
        root = self.file["/"]
                
        for depth, name in enumerate(group_path):
            if name not in root:
                if self.mode == "w":
                    root = root.create_group(name)
                else:
                    raise ValueError(f"Group {group_name} does not exist in the file")
            else:
                root = root[name]
                
        self.active_group = root

    def open(self, fname: str, mode: str, parallel: bool):
        """ Open an hdf5 file based on inputs. This can be used to open a new file after closing the previous one."""

        self.log.tic() 
        self.fname = fname
        if mode not in ["r", "w"]:
            raise ValueError("Mode should be 'read' or 'write'")
        else:
            self.mode = mode
 
        self.parallel = parallel
        if self.comm.Get_size() < 2: 
            self.parallel = False # Overwrite to serial if only one rank is used

        if self.parallel:
            self.file = h5py.File(self.fname, self.mode, driver='mpio', comm=self.comm)
        else:
            self.file = h5py.File(self.fname, self.mode)

        parallel_str = "parallel" if self.parallel else "serial"
        self.log.write("info", f"{self.fname} opened - mode {self.mode} - {parallel_str}")

        # Set the active group to the root group
        self.set_active_group("/")

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

        self.log.toc(message=f"{self.fname} closed")

    def read_dataset(self, dataset_name: str, dtype : np.dtype = np.double, distributed_axis: int = None, slices: list = None, as_1d_in_file: bool = False):
        """ Read a dataset from the hdf5 file object

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to read. Can include the group path, e.g. "/group1/group2/dataset".
        dtype : np.dtype
            Data type to read the dataset in. Default is np.double.
        distributed_axis : int
            Axis along which the data is distributed in parallel. This is required for parallel reading. Default is None.
        slices : list
            Optional. List of slices to read from the dataset. In case it is known
        as_1d_in_file : bool
            Optional. default is False. Whether the data is stored as 1D in the file. This is useful if originally the data had a different shape
            but was flattened to 1d before writing. This will use the shape attribute stored in the file to do the partioning
            but will keep in mind that the data is stored as a 1d array to read properly.
        """

        if self.mode != "r":
            raise ValueError("File is not opened in read mode")

        if self.parallel and distributed_axis is None:
            raise ValueError("Distributed axis must be specified for parallel reading")
                
        if slices is not None and len(slices) != self.active_group[dataset_name].ndim:
            raise ValueError("Number of slices must match the number of dimensions of the dataset")

        self.log.write("debug", f"Reading dataset {dataset_name} - dtype {dtype} - distributed_axis {distributed_axis}")

        # Set the active group based on the data set name
        if len(dataset_name.split("/")) > 1:
            group_name = "/".join(dataset_name.split("/")[:-1]) # Exclude the dataset name from the full path
            self.set_active_group(group_name)
            dataset_name = dataset_name.split("/")[-1]

        # Query the shape
        if "shape" in self.active_group[dataset_name].attrs:
            global_shape = self.active_group[dataset_name].attrs["shape"]
        else:
            global_shape = self.active_group[dataset_name].shape

        # ===========
        # Serial read 
        # ===========
        if not self.parallel:
            if slices is None:
                local_data = self.active_group[dataset_name][:].reshape(global_shape).astype(dtype)
            else:
                local_data = self.active_group[dataset_name][tuple(slices)]
        
        # =============
        # Parallel read 
        # =============
        else:

            # Set slices
            if slices is None:
                self.set_read_slices_linear_lb(global_shape=global_shape, distributed_axis=distributed_axis, explicit_strides=as_1d_in_file)
            else:
                self.set_read_slices_external(global_shape=global_shape, slices=slices)
            
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
        local_data[:] = self.active_group[dataset_name][self.slices]

        return local_data.reshape(self.local_shape)

    def write_dataset(self, dataset_name: str, data: np.ndarray, distributed_axis: int = None, extra_global_entries: list[int] = None, shape_in_ram: tuple = None):
        """ Write a dataset to the hdf5 file object 

        Parameters
        ----------
        dataset_name : str
            Name of the dataset to write. Can include the group path, e.g. "/group1/group2/dataset".
        data : np.ndarray
            Data to write to the file.
        distributed_axis : int
            Axis along which the data is distributed in parallel. This is required for parallel writing. Default is None.
        extra_global_entries : list[int]
            Optional. List of extra entries to add to the global shape of the dataset. This is useful
            if the ranks are writing a certain amount of data but the global array should be bigger than 
            what they collectively write.
        shape_in_ram : tuple
            Optional. Shape of the data in RAM. This is useful if the data is stored in a different shape that
            it originally had, for example, if it is stored in a 1d array but originally it had a different shape.
            this will be the shape that is stored in the file in the attribute "shape" and can be used to reshape the data when reading it.
        """

        if self.mode != "w":
            raise ValueError("File is not opened in write mode")

        if self.parallel and distributed_axis is None:
            raise ValueError("Distributed axis must be specified for parallel writing")
        
        self.log.write("debug", f"Writing dataset {dataset_name} - dtype {data.dtype} - distributed_axis {distributed_axis}")
        
        # Set the active group based on the data set name
        if len(dataset_name.split("/")) > 1:
            group_name = "/".join(dataset_name.split("/")[:-1]) # Exclude the dataset name from the full path
            self.set_active_group(group_name)
            dataset_name = dataset_name.split("/")[-1]

        # ============
        # Serial write 
        # ============
        if not self.parallel:
            dset = self.active_group.create_dataset(dataset_name, data=data, dtype=data.dtype)
            if shape_in_ram is not None:
                dset.attrs["shape"] = shape_in_ram
            else:
                dset.attrs["shape"] = data.shape

        # ==============
        # Parallel write 
        # ==============
        else:
            # Set slices 
            self.set_write_slices(local_shape=data.shape, distributed_axis=distributed_axis, extra_global_entries=extra_global_entries)
    
            # Write the slices
            self.write_slices(dataset_name, data, shape_in_file=shape_in_ram)
 
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
        
    def write_slices(self, dataset_name: str, data: np.ndarray, shape_in_file: tuple = None):
        """Write the slices that should be written to the file. This is useful if the slices have been set using set_write_slices and we want to write the same slices again."""
        if self.slices is None:
            raise ValueError("Slices have not been set")
        if self.global_shape is None:
            raise ValueError("Global shape is not set")

        dset = self.active_group.create_dataset(dataset_name, shape=self.global_shape, dtype=data.dtype)
        dset[self.slices] = data
        if shape_in_file is not None:
            dset.attrs["shape"] = shape_in_file
        else:
            dset.attrs["shape"] = self.global_shape
            