""" Module that contains the class and methods to perform direct sampling on a field """

import random
from mpi4py import MPI
from ..monitoring.logger import Logger
from ..datatypes.msh import Mesh
from ..datatypes.coef import Coef
from ..datatypes.coef import get_transform_matrix, get_derivative_matrix
import numpy as np
import bz2
import sys
import h5py
import os
import torch
import math

class DirectSampler:

    """ 
    Class to perform direct sampling on a field in the SEM format
    """

    def __init__(self, comm: MPI.Comm = None, dtype: np.dtype = np.double,  msh: Mesh = None, filename: str = None, max_elements_to_process: int = 256, bckend: str = "numpy", mass_matrix = None):
        
        self.log = Logger(comm=comm, module_name="DirectSampler")
        
        self.b = mass_matrix
        
        if msh is not None:
            self.init_from_msh(msh, dtype=dtype, max_elements_to_process=max_elements_to_process)
        elif filename is not None:
            self.init_from_file(comm, filename, max_elements_to_process=max_elements_to_process)
        else:
            self.log.write("info", "No mesh provided. Please provide a mesh to initialize the DirectSampler")

        # Init bckend
        self.bckend = bckend
        if bckend == "torch": 
            # Find the device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Set the device dtype
            if dtype == np.float32:
                self.dtype_d = torch.float32
            elif dtype == np.float64:
                self.dtype_d = torch.float64
            # Transfer needed data
            self.v_d = torch.tensor(self.v, dtype=self.dtype_d, device = self.device, requires_grad=False)
            self.vinv_d = torch.tensor(self.vinv, dtype=self.dtype_d, device = self.device, requires_grad=False)
            self.dr_d = torch.tensor(self.dr, dtype=self.dtype_d, device = self.device, requires_grad=False)
            self.ds_d = torch.tensor(self.ds, dtype=self.dtype_d, device = self.device, requires_grad=False)
            if self.gdim > 2:
                self.dt_d = torch.tensor(self.dt, dtype=self.dtype_d, device = self.device, requires_grad=False)
            self.dn_d = torch.tensor(self.dn, dtype=self.dtype_d, device = self.device, requires_grad=False)
            self.dv_dr_d = torch.matmul(self.dr_d, self.v_d)
            self.dv_ds_d = torch.matmul(self.ds_d, self.v_d)
            if self.gdim > 2:
                self.dv_dt_d = torch.matmul(self.dt_d, self.v_d)

            if not isinstance(self.b, type(None)):
                self.b_d = torch.tensor(self.b, dtype=self.dtype_d, device = self.device, requires_grad=False)
            else:
                self.b_d = None

            # If the data was initialized from file, put it in a torch tensor
            if hasattr(self, "uncompressed_data"):
                for field in self.uncompressed_data.keys():
                    for data in self.uncompressed_data[field].keys():
                        self.uncompressed_data[field][data] = torch.tensor(self.uncompressed_data[field][data], dtype=self.dtype_d, device = self.device, requires_grad=False)

        self.supporting_data = {}

    def init_from_file(self, comm: MPI.Comm, filename: str, max_elements_to_process: int = 256):
        """
        """

        self.log.write("info", f"Initializing the DirectSampler from file: {filename}")

        self.settings, self.compressed_data = self.read_compressed_samples(comm = comm, filename=filename)

        self.init_common(max_elements_to_process)

        self.uncompressed_data = self.decompress_samples(self.settings, self.compressed_data)

        self.kw_diag = self.settings["covariance"]["kw_diag"]


    def init_from_msh(self, msh: Mesh, dtype: np.dtype = np.double, max_elements_to_process: int = 256):

        self.log.write("info", "Initializing the DirectSampler from a Mesh object")
        
        # Geometrical parameters for this mesh
        nelv = msh.nelv
        lz = msh.lz
        ly = msh.ly
        lx = msh.lx
        gdim = msh.gdim
        
        # Dictionary to store the settings as they are added
        self.settings = {}
        if dtype == np.float32:
            self.settings["dtype"] = "single"
        elif dtype == np.float64:
            self.settings["dtype"] = "double"
        self.settings["mesh_information"] = {"lx": lx, "ly": ly, "lz": lz, "nelv": nelv, "gdim": gdim}

        # Create a dictionary that will have the data that needs to be compressed later
        self.uncompressed_data = {}

        # Create a dictionary that will hold the data after compressed
        self.compressed_data = {}

        # Initialize the common parameters
        self.init_common(max_elements_to_process)

    def init_common(self, max_elements_to_process: int = 256):

        self.max_elements_to_process = max_elements_to_process

        # Mesh information
        self.lx = self.settings["mesh_information"]["lx"]
        self.ly = self.settings["mesh_information"]["ly"]
        self.lz = self.settings["mesh_information"]["lz"]
        self.gdim = self.settings["mesh_information"]["gdim"]
        self.nelv = self.settings["mesh_information"]["nelv"]

        # dtype
        if self.settings["dtype"] == "single":
            self.dtype = np.float32
        elif self.settings["dtype"] == "double":
            self.dtype = np.float64
        
        # Get transformation matrices for this mesh
        self.v, self.vinv, self.w3, self.x, self.w = get_transform_matrix(
            self.lx, self.gdim, apply_1d_operators=False, dtype=self.dtype
        )

        self.dr, self.ds, self.dt, self.dn = get_derivative_matrix(
            self.lx, self.gdim, dtype=self.dtype, apply_1d_operators=False
        )

        self.dv_dr = np.matmul(self.dr, self.v)
        self.dv_ds = np.matmul(self.ds, self.v)
        if self.gdim > 2:
            self.dv_dt = np.matmul(self.dt, self.v) 


    def clear(self):

        # Clear the data that has been sampled. This is necesary to avoid mixing things up when sampling new fields.
        self.settings = {}
        self.uncompressed_data = {}
        self.compressed_data = {}
    
    def sample_field(self, field: np.ndarray = None, field_name: str = "field", covariance_method: str = "average", covariance_elements_to_average: int = 1, covariance_keep_modes: int=1,
                    compression_method: str = "fixed_bitrate", bitrate: float = 1/2, max_samples_per_it: int = 1, update_noise = False):
        
        self.log.write("info", "Sampling the field with options: covariance_method: {covariance_method}, compression_method: {compression_method}")

        # Copy the field into device if needed
        if self.bckend == "torch":
            field = torch.tensor(field, dtype=self.dtype_d, device = self.device, requires_grad=False)

        self.log.write("info", "Estimating the covariance matrix")
        self._estimate_field_covariance(field=field, field_name=field_name, method=covariance_method, elements_to_average=covariance_elements_to_average, keep_modes=covariance_keep_modes)

        if compression_method == "fixed_bitrate":
            self.settings["compression"] =  {"method": compression_method,
                                             "bitrate": bitrate,
                                             "n_samples" : int(self.lx*self.ly*self.lz * bitrate),
                                             "update_noise": update_noise}
            
            if self.bckend == "numpy":
                self.log.write("info", f"Sampling the field using the fixed bitrate method. using settings: {self.settings['compression']}")
                if update_noise:
                    raise ValueError("The update_noise option is not supported for the numpy backend")
                field_sampled = self._sample_fixed_bitrate(field, field_name, self.settings, max_samples_per_it)
            elif self.bckend == "torch":
                self.log.write("info", f"Sampling the field using the fixed bitrate method. using settings: {self.settings['compression']}")
                self.log.write("info", f"Using backend: {self.bckend} on device: {self.device}")
                if update_noise and (covariance_method != "average"):
                    self.uncompressed_data[f"{field_name}"]["noise"] = 1e-3 * torch.ones((field.shape[0], 1), dtype=self.dtype_d, device = self.device, requires_grad=False)
                field_sampled = self._sample_fixed_bitrate_torch(field, field_name, self.settings, max_samples_per_it, update_noise)

            self.uncompressed_data[f"{field_name}"]["field"] = field_sampled
            self.log.write("info", f"Sampled_field saved in field uncompressed_data[\"{field_name}\"][\"field\"]")

        else:
            raise ValueError("Invalid method to sample the field")
        
    def compress_samples(self, lossless_compressor: str = "bzip2"):
        """
        """

        self.log.write("info", f"Compressing the data using the lossless compressor: {lossless_compressor}")
        self.log.write("info", "Compressing data in uncompressed_data")
        for field in self.uncompressed_data.keys():
            self.log.write("info", f"Compressing data for field [\"{field}\"]:")
            self.compressed_data[field] = {}
            for data in self.uncompressed_data[field].keys():
                self.log.write("info", f"Compressing [\"{data}\"] for field [\"{field}\"]")
                if self.bckend == "numpy":
                    self.compressed_data[field][data] = bz2.compress(self.uncompressed_data[field][data].tobytes())
                elif self.bckend == "torch":
                    self.compressed_data[field][data] = bz2.compress(self.uncompressed_data[field][data].cpu().numpy().tobytes())


    def write_compressed_samples(self, comm = None,  filename="compressed_samples.h5"):
        """
        Writes compressed data to an HDF5 file in a hierarchical format, with separate
        groups for each MPI rank. If parallel HDF5 is supported, all ranks write to a single file
        using the 'mpio' driver. Otherwise, a folder is created to hold separate files for each rank,
        and a log message is generated to indicate this behavior.
        
        Parameters:
            compressed_data (dict): A dictionary structured as { field: { data_key: compressed_bytes } }
            filename (str): Base filename for the HDF5 file.
        """
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        try:
            # Check if h5py was built with MPI support.
            if h5py.get_config().mpi:
                # Open a single file for parallel writing.
                f = h5py.File(filename, "w", driver="mpio", comm=comm)
            else:
                raise RuntimeError("Parallel HDF5 not supported in this h5py build.")
        except Exception:
            # Log that parallel HDF5 is not available and a folder will be created.
            self.log.write("info", "Parallel HDF5 not available; creating folder to store rank files.")
            path_name = os.path.dirname(filename)
            base_name = os.path.basename(filename)
            folder_name = f"{base_name}_comp"
            if rank == 0:
                os.makedirs(os.path.join(path_name, folder_name), exist_ok=True)
            # Ensure all ranks wait until the folder has been created.
            comm.Barrier()
            file_path = os.path.join(path_name, folder_name, f"{base_name}_rank_{rank}.h5")
            f = h5py.File(file_path, "w")

        # Indicate that the data are bytes
        binary_dtype = h5py.vlen_dtype(np.uint8)

        with f:

            # If settings exist, add them as metadata in a top-level group.
            if hasattr(self, "settings") and self.settings is not None:
                # In parallel mode, have rank 0 create the settings group.
                if comm.Get_rank() == 0:
                    settings_group = f.create_group("settings")
                    settings_dict = {key: self.settings[key] for key in self.settings.keys() if key != "mesh_information"}
                    add_settings_to_hdf5(settings_group, settings_dict)
                                         

            # Ensure all ranks wait until settings are written.
            comm.Barrier()

            # Create a top-level group for this rank.
            rank_group = f.create_group(f"rank_{rank}")

            # Add the mesh information of the rank
            mesh_info_group = rank_group.create_group("mesh_information")
            add_settings_to_hdf5(mesh_info_group, self.settings["mesh_information"])

            for field, data_dict in self.compressed_data.items():
                # Create a subgroup for each field.
                field_group = rank_group.create_group(field)
                for data_key, compressed_bytes in data_dict.items():

                    # This step is necessary to convert the bytes to a numpy array. to store in HDF5 ...
                    # ... It produced problems until I did that.                    
                    data_array = np.frombuffer(compressed_bytes, dtype=np.uint8)
                    dset = field_group.create_dataset(data_key, (1,), dtype=binary_dtype)
                    dset[0] = data_array

    def read_compressed_samples(self, comm=None, filename="compressed_samples.h5"):
        """
        Reads an HDF5 file (or folder of files if non-parallel mode was used) created by write_compressed_samples.
        Assumes that the same number of ranks is used for reading as for writing and that each rank reads only its own data.
        
        Returns a tuple:
            (global_settings, local_data)
        where:
            - global_settings is a dictionary from the top-level "settings" group (e.g., with keys "covariance" and "compression")
            and augmented with the rank-specific "mesh_information".
            - local_data is a dictionary structured as { "compressed_data": { field: { data_key: compressed_bytes } } }
        """
        
        rank = comm.Get_rank()

        # Open the file in parallel mode if available; otherwise open the per-rank file.
        try:
            if h5py.get_config().mpi:
                f = h5py.File(filename, "r", driver="mpio", comm=comm)
                mode = "parallel"
            else:
                raise RuntimeError("Parallel HDF5 not supported")
        except Exception:
            path_name = os.path.dirname(filename)
            base_name = os.path.basename(filename)
            folder_name = f"{base_name}_comp"
            file_path = os.path.join(path_name, folder_name, f"{base_name}_rank_{rank}.h5")
            f = h5py.File(file_path, "r")
            mode = "non_parallel"

        # Read global settings (from the top-level "settings" group, written by rank 0).
        global_settings = {}
        if rank == 0:
            global_settings = load_hdf5_settings(f["settings"])
        global_settings = comm.bcast(global_settings, root=0)

        # Read rank-specific data from the "rank_{rank}" group.
        rank_group = f[f"rank_{rank}"]

        # Read the rank-specific mesh information from the "mesh_information" subgroup.
        mesh_information = {}
        if "mesh_information" in rank_group:
            mesh_information = load_hdf5_settings(rank_group["mesh_information"])

        # Add mesh_information to global_settings.
        global_settings["mesh_information"] = mesh_information

        # Read compressed data from the remaining groups (fields).
        compressed_data = {}
        for field_key in rank_group:
            # Skip the mesh_information subgroup.
            if field_key == "mesh_information":
                continue
            field_group = rank_group[field_key]
            field_dict = {}
            for data_key in field_group:
                dset = field_group[data_key]
                # Each dataset is stored as an array of shape (1,) containing a variable-length uint8 array.
                field_dict[data_key] = dset[0].tobytes()
            compressed_data[field_key] = field_dict

        f.close()

        return global_settings, compressed_data
    
    def decompress_samples(self, settings, compressed_data=None):
        """
        Decompresses the compressed data in the compressed_data dictionary.
        """

        uncompressed_data = {}
        for field, data_dict in compressed_data.items():
            uncompressed_data[field] = {}
            for data_key, compressed_bytes in data_dict.items():

                dtype = settings["dtype"]

                # Select the shape based on the name of the data
                nelv = settings["mesh_information"]["nelv"]
                lz = settings["mesh_information"]["lz"]
                ly = settings["mesh_information"]["ly"]
                lx = settings["mesh_information"]["lx"]
                average = settings["covariance"]["averages"]
                elements_to_average = settings["covariance"]["elements_to_average"]
                keep_modes = settings["covariance"].get("keep_modes", 1)
                kw_diag = settings["covariance"]["kw_diag"]
                
                if dtype == "single":
                    d_dtype = np.float32
                elif dtype == "double":
                    d_dtype = np.float64

                if data_key == "field":    
                    shape = (nelv, lz, ly, lx)
                elif data_key == "kw":
                    if kw_diag:
                        shape = (average, lx*ly*lz)
                    else:
                        shape = (average, lx*ly*lz, lx*ly*lz)
                elif data_key == "U":
                    shape = (lx*ly*lz, lx*ly*lz)
                elif data_key == "s":
                    shape = (keep_modes)
                elif data_key == "Vt":
                    shape = (keep_modes, lx*ly*lz)
                elif data_key == "f_hat":    
                    shape = (nelv, lz, ly, lx)
                elif data_key == "noise":
                    shape = (nelv, 1)
                elif data_key == "f_hat_packed":
                    packed_size = int(np.ceil(lx*ly*lz/8))
                    shape = (average*elements_to_average, keep_modes, packed_size)
                    d_dtype = np.uint8
                elif data_key == "f_hat_max":
                    shape = (average*elements_to_average, 1)
                elif data_key == "f_hat_min":
                    shape = (average*elements_to_average, 1)
                elif data_key == "f_hat_shape":
                    shape = (2,)
                    d_dtype = np.int64
                else:
                    raise ValueError("Invalid data key")

                temp = np.frombuffer(bz2.decompress(compressed_bytes), dtype=d_dtype)

                uncompressed_data[field][data_key] = temp.reshape(shape)

        return uncompressed_data



    def _estimate_field_covariance(self, field: np.ndarray = None, field_name: str = "field", method="average", elements_to_average: int = 1, keep_modes: int = 1):
        """
        """

        # Create a dictionary to store the data that will be compressed
        self.uncompressed_data[f"{field_name}"] = {}


        self.log.write("info", "Transforming the field into to legendre space")
        field_hat = self.transform_field(field, to="legendre")

        if method == "average":

            # In this case, the kw should be taken as already the diagonal form
            self.kw_diag = True

            self.settings["covariance"] = {"method": "average",
                                           "elements_to_average": elements_to_average,
                                           "averages": int(np.ceil(self.nelv/elements_to_average)),
                                           "kw_diag": self.kw_diag}

            self.log.write("info", f"Estimating the covariance matrix using the averaging method method. Averaging over {elements_to_average} elements at a time")
            kw = self._estimate_covariance_average(field_hat, self.settings["covariance"])

            # Store the covariances in the data to be compressed:
            self.uncompressed_data[f"{field_name}"]["kw"] = kw
            self.log.write("info", f"Covariance saved in field uncompressed_data[\"{field_name}\"][\"kw\"]")

        elif method == "svd":
            # In this case, the kw will not be only the diagonal of the stored data but an approximation of the actual covariance
            self.kw_diag = False
            
            self.settings["covariance"] = {"method": "svd",
                                           "averages" : int(1),
                                           "elements_to_average": self.nelv,
                                           "keep_modes": keep_modes,
                                           "kw_diag": self.kw_diag}
            
            self.log.write("info", f"Estimating the covariance matrix using the SVD method. Keeping {keep_modes} modes")
            U, kw = self._estimate_covariance_svd(field, self.settings["covariance"])

            # Store the covariances in the data to be compressed:
            self.uncompressed_data[f"{field_name}"]["U"] = U
            self.uncompressed_data[f"{field_name}"]["kw"] = kw

            self.log.write("info", f"U saved in field uncompressed_data[\"{field_name}\"][\"U\"]")
            self.log.write("info", f"s saved in field uncompressed_data[\"{field_name}\"][\"kw\"]")
        
        elif method == "dlt":
            # In this case, the kw will not be only the diagonal of the stored data but an approximation of the actual covariance
            self.kw_diag = False
            
            self.settings["covariance"] = {"method": "dlt",
                                           "averages" : self.nelv,
                                           "elements_to_average": int(1),
                                           "keep_modes": keep_modes,
                                           "kw_diag": self.kw_diag}
            
            self.log.write("info", f"Estimating the covariance matrix using the DLT method. Keeping {keep_modes} modes")
            fld_hat_truncated = self._estimate_covariance_dlt(field_hat, self.settings["covariance"])

            # Store the covariances in the data to be compressed:
            self.uncompressed_data[f"{field_name}"]["f_hat"] = fld_hat_truncated

            self.log.write("info", f"f_hat saved in field uncompressed_data[\"{field_name}\"][\"f_hat\"]")
        
        elif method.split("_")[0] == "ad":
            # In this case, the kw will not be only the diagonal of the stored data but an approximation of the actual covariance
            self.kw_diag = False
            
            try:
                estimation = method.split("_")[1]
            except:
                estimation = ""

            self.settings["covariance"] = {"method": "ad",
                                           "estimation": estimation,
                                           "averages" : self.nelv,
                                           "elements_to_average": int(1),
                                           "keep_modes": keep_modes,
                                           "kw_diag": self.kw_diag}
            
            self.log.write("info", f"Estimating the covariance matrix using the AD method. Keeping {keep_modes} modes")
            if (estimation == "rmse") or (estimation == ""):
                self.field = field
                fld_hat_truncated = self._estimate_covariance_ad_rmse(field_hat, self.settings["covariance"])
            elif estimation == "ml":
                self.field = field
                fld_hat_truncated = self._estimate_covariance_ad_ml(field_hat, self.settings["covariance"])
            else:
                raise ValueError("Invalid estimation method for the AD method")

            # Store the covariances in the data to be compressed:
            self.uncompressed_data[f"{field_name}"]["f_hat"] = fld_hat_truncated

            self.log.write("info", f"f_hat saved in field uncompressed_data[\"{field_name}\"][\"f_hat\"]")
        
        elif method == "bitplane":
            # In this case, the kw will not be only the diagonal of the stored data but an approximation of the actual covariance
            self.kw_diag = False
            
            self.settings["covariance"] = {"method": "bitplane",
                                           "averages" : self.nelv,
                                           "elements_to_average": int(1),
                                           "keep_modes": keep_modes,
                                           "kw_diag": self.kw_diag}
            
            self.log.write("info", f"Estimating the covariance matrix using the DLT method. Keeping {keep_modes} modes")
            y_packed, y_max, y_min, y_shape = self._estimate_covariance_bitplane(field_hat, self.settings["covariance"])

            # Store the covariances in the data to be compressed:
            self.uncompressed_data[f"{field_name}"]["f_hat_packed"] = y_packed
            self.uncompressed_data[f"{field_name}"]["f_hat_max"] = y_max
            self.uncompressed_data[f"{field_name}"]["f_hat_min"] = y_min
            self.uncompressed_data[f"{field_name}"]["f_hat_shape"] = np.array(list(y_shape), dtype = np.int64)

            self.log.write("info", f"f_hat saved in field uncompressed_data[\"{field_name}\"][\"f_hat\"]")


        else:
            raise ValueError("Invalid method to estimate the covariance matrix")
        
        self.log.write("info", f"Covariance matrix estimated with settings: {self.settings['covariance']}")
    
    def _sample_fixed_bitrate(self, field: np.ndarray, field_name: str, settings: dict, max_samples_per_it : int = 1):
        """
        """
        if max_samples_per_it != 1:
            self.log.write("warning", "The max_samples_per_it parameter is not supported for the numpy backend. It will be ignored (Set to 1)")

        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]

        # Retrieve the number of samples
        n_samples = settings["compression"]["n_samples"]

        # Reshape the fields into the KW supported shapes
        y = field.reshape(averages, elements_to_average, field.shape[1], field.shape[2], field.shape[3])
        V = self.v
        numfreq = n_samples

        # Now reshape the x, y elements into column vectors
        y = field.reshape(averages, elements_to_average, field.shape[1] * field.shape[2] * field.shape[3], 1)

        #allocation the truncated field
        y_truncated = np.ones_like(y) * -50

        # Create an array that contains the indices of the elements that have been sampled
        # The first indext to store is always index 0
        ind_train = np.zeros((averages, elements_to_average, n_samples), dtype=int)

        # Set up some help for the selections
        avg_idx = np.arange(averages)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
        elem_idx = np.arange(elements_to_average)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

        chunk_size_e = self.max_elements_to_process
        n_chunks_e = int(np.ceil(elements_to_average / chunk_size_e))
        chunk_size_a = self.max_elements_to_process
        n_chunks_a = int(np.ceil(averages / chunk_size_a))

        for chunk_id_a in range(n_chunks_a):
            start_a = chunk_id_a * chunk_size_a
            end_a = (chunk_id_a + 1) * chunk_size_a
            if end_a > averages:
                end_a = averages

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = (chunk_id_e + 1) * chunk_size_e
                if end_e > elements_to_average:
                    end_e = elements_to_average

                avg_idx = np.arange(start_a, end_a)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
                elem_idx = np.arange(start_e, end_e)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

                self.log.write("info",f"Proccesing up to {(avg_idx.flatten()[-1] + 1) * (elem_idx.flatten()[-1]+1)}/{self.nelv} element")

                avg_idx2 = avg_idx.reshape(avg_idx.shape[0], 1)
                elem_idx2 = elem_idx.reshape(1, elem_idx.shape[1])
                
                # Set the initial index to be added
                imax = np.zeros((avg_idx.shape[0], elem_idx.shape[1]), dtype=int)

                # Get covariance matrix
                kw = self._get_covariance_matrix(settings, field_name, avg_idx2, elem_idx2)

                for freq in range(0,numfreq):

                    self.log.write("debug", f"Obtaining sample {freq+1}/{numfreq}")

                    # Sort the indices for each average and element
                    ind_train[avg_idx2, elem_idx2, :freq+1] = np.sort(ind_train[avg_idx2, elem_idx2, :freq+1], axis=2)

                    # Get the prediction and the standard deviation
                    y_21, y_21_std = self.gaussian_process_regression(y, V, kw, ind_train, avg_idx, elem_idx, avg_idx2, elem_idx2, freq, predict_mean=False, predict_std=True) 

                    # Set the variance as zero for the samples that have already been selected
                    I, J, _ = np.ix_(np.arange(y_21_std.shape[0]), np.arange(y_21_std.shape[1]), np.arange(y_21_std.shape[2]))
                    I2, J2, K2 = np.ix_(avg_idx2.flatten(), elem_idx2.flatten(), np.arange(freq+1))
                    y_21_std[I, J, ind_train[I2,J2,K2]] = 0

                    # Get the index of the sample with the highest standardd deviation
                    imax = np.argmax(y_21_std, axis=2)
                    if np.any(imax == 0):
                        # Replace 0 values in imax with random integers not in ind_train
                        zero_indices = np.argwhere(imax == 0)
                        for i, j in zero_indices:
                            possible_indices = np.setdiff1d(np.arange(y.shape[2]), ind_train[avg_idx2[i, 0], elem_idx2[0, j], :freq+1])
                            if len(possible_indices) > 0:
                                imax[i, j] = np.random.choice(possible_indices)
 
                    # Assign the index to be added
                    if freq < numfreq-1:
                        ind_train[avg_idx2, elem_idx2,freq+1] = imax

                # Get the finally selected samples
                y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2,elem_idx2,:],:]

                # This is still with column vectors at the end. We need to reshape it.
                y_truncated[avg_idx, elem_idx, ind_train[avg_idx2, elem_idx2, :],:] = y_11

        # Reshape the field back to its original shape
        return y_truncated.reshape(field.shape)

    def _sample_fixed_bitrate_torch(self, field: torch.Tensor, field_name: str, settings: dict, max_samples_per_it: int = 1, update_noise : bool = False):
        """
        """

        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]
        n_samples = settings["compression"]["n_samples"]

        # Set the number of samples to be taken individually before grouping them
        lx = self.lx

        # For KW routines, we first reshape into a five-dimensional tensor.
        V = self.v_d
        if settings["covariance"]["method"] == "svd":
            V = self.uncompressed_data[field_name]["U"]
        numfreq = n_samples

        # Reshape the spatial dimensions into column vectors:
        # New shape: (averages, elements_to_average, spatial_elements, 1)
        y = field.reshape(averages, elements_to_average, field.shape[1] * field.shape[2] * field.shape[3], 1)

        if settings["covariance"]["method"] == "average":
            update_noise = False
        if update_noise:
            noise = self.uncompressed_data[field_name]["noise"].view(averages, elements_to_average, 1, 1)
        else:
            noise = None


        # Allocate the truncated field with the same type and device as y, filled with -50.
        y_truncated = torch.ones_like(y) * -50

        # Create an array to hold indices of the sampled elements.
        ind_train = torch.zeros((averages, elements_to_average, n_samples), dtype=torch.int64, device=y.device)

        # Define index helpers (for the full range, though we will use slicing for subchunks)
        # (Not used in all places below since slicing simplifies contiguous chunks.)
        full_avg_idx = torch.arange(averages, device=y.device).view(averages, 1, 1)
        full_elem_idx = torch.arange(elements_to_average, device=y.device).view(1, elements_to_average, 1)

        # Set up chunking parameters to avoid processing too many elements at once.
        chunk_size_e = self.max_elements_to_process
        n_chunks_e = math.ceil(elements_to_average / chunk_size_e)
        chunk_size_a = self.max_elements_to_process
        n_chunks_a = math.ceil(averages / chunk_size_a)

        # Loop over chunks along the "averages" dimension.
        for chunk_id_a in range(n_chunks_a):
            start_a = chunk_id_a * chunk_size_a
            end_a = min((chunk_id_a + 1) * chunk_size_a, averages)

            # Loop over chunks along the "elements_to_average" dimension.
            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, elements_to_average)

                # Create chunk-specific index helpers.
                # avg_idx: shape (chunk_a, 1, 1); elem_idx: shape (1, chunk_e, 1)
                avg_idx = torch.arange(start_a, end_a, device=y.device).view(-1, 1, 1)
                elem_idx = torch.arange(start_e, end_e, device=y.device).view(1, -1, 1)
                # Also define flattened versions for later use.
                avg_idx2 = avg_idx.reshape(avg_idx.shape[0], 1)  # shape: (chunk_a, 1)
                elem_idx2 = elem_idx.reshape(1, elem_idx.shape[1])  # shape: (1, chunk_e)

                # Log processing info.
                last_avg = avg_idx.flatten()[-1].item() + 1
                last_elem = elem_idx.flatten()[-1].item() + 1
                self.log.write("info", f"Processing up to {last_avg * last_elem}/{self.nelv} element")

                # Initialize imax (for the indices with maximum standard deviation)
                imax = torch.zeros((avg_idx.shape[0], elem_idx.shape[1]), dtype=torch.int64, device=y.device)

                # Get the covariance matrix for the current chunk.
                kw = self._get_covariance_matrix(settings, field_name, avg_idx2, elem_idx2)

                # Loop over frequency/sample iterations.
                # Loop over frequency/sample iterations.
                freq = 0  # Tracks total number of samples collected so far
                iteration = 1  # Tracks the number of iterations performed
                self.log.write("debug", f"Sample 1 is index 0")
                while freq < numfreq - 1:

                    self.log.write("debug", f"Obtaining sample {freq+1+1}/{numfreq}")

                    # For the current chunk, sort the sampled indices along the frequency axis.
                    # Using slicing over the contiguous block.
                    sub_ind_train = ind_train[start_a:end_a, start_e:end_e, :freq+1]
                    sorted_sub_ind, _ = torch.sort(sub_ind_train, dim=2)
                    ind_train[start_a:end_a, start_e:end_e, :freq+1] = sorted_sub_ind

                    #Perform gaussian process regression
                    y_21, y_21_std = self.gaussian_process_regression_torch(
                        y, V, kw, ind_train,
                        avg_idx, elem_idx, avg_idx2, elem_idx2,
                        freq, predict_mean=False, predict_std=True, noise=noise
                    )

                    # Set the standard deviation to zero at indices that have already been sampled.
                    # For each (i, j) in the chunk and each k in 0..freq, set:
                    #    y_21_std[i, j, ind_train[i, j, k]] = 0
                    sub_y_std = y_21_std  # shape: (chunk_a, chunk_e, some_dim)
                    for k in range(freq + 1):
                        # Extract the sample indices for this frequency.
                        indices = ind_train[start_a:end_a, start_e:end_e, k]  # shape: (chunk_a, chunk_e)
                        # Create meshgrid for the first two dimensions.
                        a_idx, e_idx = torch.meshgrid(
                            torch.arange(end_a - start_a, device=y.device),
                            torch.arange(end_e - start_e, device=y.device),
                            indexing='ij'
                        )
                        sub_y_std[a_idx, e_idx, indices] = 0
                    
                    # Determine how many new samples to select this iteration
                    num_samples_this_it = 1 if iteration < lx else min(max_samples_per_it, (numfreq-1) - freq)


                    # Change the strategy depending on the number of samples to be selected.
                    # This one is always off, kept here for debugging in the close future
                    if max_samples_per_it == 0:
                        
                        # Get the index (along the third dimension) of the maximum standard deviation.
                        imax = torch.argmax(sub_y_std, dim=2)

                        # For any location where imax == 0, replace it with a random index not already sampled.
                        if (imax == 0).any():
                            zero_indices = (imax == 0).nonzero(as_tuple=False)
                            for idx in zero_indices:
                                i, j = idx[0].item(), idx[1].item()
                                # Get already selected indices for this (i,j) in the current chunk.
                                already_selected = ind_train[start_a:end_a, start_e:end_e, :freq+1][i, j].tolist()
                                all_indices = set(range(y.shape[2]))
                                possible_indices = list(all_indices - set(already_selected))
                                if possible_indices:
                                    imax[i, j] = random.choice(possible_indices)

                        # Save the newly selected indices if more samples are needed.
                        ind_train[start_a:end_a, start_e:end_e, freq+1] = imax
                    
                    else:

                        sorted_indices = torch.argsort(sub_y_std, dim=2, descending=True)  

                        final_selected = torch.zeros(
                            (sub_y_std.shape[0], sub_y_std.shape[1], num_samples_this_it),
                            dtype=torch.long,
                            device=sub_y_std.device
                        )

                        for i in range(sub_y_std.shape[0]):       # chunk_a
                            for j in range(sub_y_std.shape[1]):   # chunk_e

                                i_global = start_a + i
                                j_global = start_e + j

                                already_selected = set(ind_train[i_global, j_global, :freq+1].tolist())

                                candidate_list = sorted_indices[i, j]  # shape = [n_points]

                                # Filter out duplicates
                                valid_indices = []
                                for c in candidate_list:
                                    idx_val = c.item()
                                    if idx_val not in already_selected:
                                        valid_indices.append(idx_val)
                                    # Stop once we have enough new indices
                                    if len(valid_indices) == num_samples_this_it:
                                        break

                                final_selected[i, j, :] = torch.tensor(
                                    valid_indices,
                                    dtype=torch.long,
                                    device=sub_y_std.device
                                )

                        # Store the newly selected indices in `ind_train`.
                        for i in range(sub_y_std.shape[0]):
                            for j in range(sub_y_std.shape[1]):
                                i_global = start_a + i
                                j_global = start_e + j
                                # Write into the correct slice
                                ind_train[i_global, j_global, freq+1 : freq+1+num_samples_this_it] = final_selected[i, j]

                    # Update the noise if pertinent
                    if update_noise:

                        if not isinstance(self.b_d, type(None)):                            
                            
                            bb_ = self.b_d.view(averages, elements_to_average, field.shape[1] * field.shape[2] * field.shape[3], 1) 
                            bb = bb_[avg_idx2, elem_idx2]
                            y_ = y[avg_idx2, elem_idx2]
                            residuals = y_ - y_21
                            residuals2 = residuals ** 2

                            # Get_std
                            residuals = torch.sum(residuals * bb, dim=2, keepdim=True) / torch.sum(bb, dim=2, keepdim=True)  # per-batch sum across samples
                            residuals2 = torch.sum(residuals2 *bb, dim=2, keepdim=True) / torch.sum(bb, dim=2, keepdim=True)  # per-batch sum across samples
                            sigma_n = torch.sqrt(abs(residuals2 - residuals ** 2))  # per-batch std across samples

                            # Update the noise
                            noise[start_a:end_a, start_e:end_e] = sigma_n
                            
                        else:
                            y_ = y[avg_idx2, elem_idx2]
                            residuals = y_ - y_21
                            sigma_n = residuals.std(dim=2, keepdim=True)  # per-batch std across samples
                            noise[start_a:end_a, start_e:end_e] = sigma_n

                    # Update counters
                    freq += num_samples_this_it
                    iteration += 1

                # After finishing frequency iterations for this chunk, assign the final selected samples.
                # For each frequency k, we extract the corresponding sample from y and place it into y_truncated.
                # We work on the current chunk only.
                chunk_a = end_a - start_a
                chunk_e = end_e - start_e
                # Obtain sub-tensors for convenience.
                y_chunk = y[start_a:end_a, start_e:end_e, :, :]         # shape: (chunk_a, chunk_e, spatial_elements, 1)
                y_trunc_chunk = y_truncated[start_a:end_a, start_e:end_e, :, :]  # same shape
                # Create meshgrid for the two leading dimensions.
                a_idx, e_idx = torch.meshgrid(
                    torch.arange(chunk_a, device=y.device),
                    torch.arange(chunk_e, device=y.device),
                    indexing='ij'
                )
                for k in range(n_samples):
                    # For each (i,j) in the chunk, get the index from ind_train and assign that column.
                    indices = ind_train[start_a:end_a, start_e:end_e, k]  # shape: (chunk_a, chunk_e)
                    # Use advanced indexing to assign the corresponding values.
                    y_trunc_chunk[a_idx, e_idx, indices, :] = y_chunk[a_idx, e_idx, indices, :]
                # Place the updated chunk back into y_truncated.
                y_truncated[start_a:end_a, start_e:end_e, :, :] = y_trunc_chunk

        # Reshape the truncated field back to the original shape of "field"
        return y_truncated.reshape(field.shape)

    def reconstruct_field(self, field_name: str = None, get_mean: bool = True, get_std: bool = False, mean_op = None, std_op = None, use_coefficients: bool = False):
            if self.bckend == "numpy":
                return self.reconstruct_field_numpy(field_name, get_mean, get_std)
            elif self.bckend == "torch":
                if not use_coefficients:
                    return self.reconstruct_field_torch(field_name, get_mean, get_std, mean_op = mean_op, std_op = std_op)
                else:
                    return self.reconstruct_field_torch_coefficients(field_name)

    def reconstruct_field_numpy(self, field_name: str = None, get_mean: bool = True, get_std: bool = False):
        """
        """

        # Retrieve the sampled field
        sampled_field = self.uncompressed_data[field_name]["field"]
        settings = self.settings

        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]

        # Retrieve the number of samples
        n_samples = settings["compression"]["n_samples"]

        # Reshape the fields into the KW supported shapes
        y = sampled_field.reshape(averages, elements_to_average, sampled_field.shape[1], sampled_field.shape[2], sampled_field.shape[3])
        V = self.v
        numfreq = n_samples

        # Now reshape the x, y elements into column vectors
        y = sampled_field.reshape(averages, elements_to_average, sampled_field.shape[1] * sampled_field.shape[2] * sampled_field.shape[3], 1)

        #allocation the truncated field
        y_reconstructed = None
        y_reconstructed_std = None
        if get_mean:
            y_reconstructed = np.ones_like(y) * -50
        if get_std:
            y_reconstructed_std = np.ones_like(y) * -50

        # Create an array that contains the indices of the elements that have been sampled
        # The first indext to store is always index 0
        ind_train = np.zeros((averages, elements_to_average, n_samples), dtype=int)
        ## Get the ind train from the sampled field
        for e in range(averages):
            for i in range(elements_to_average):
                temp = np.where(y[e,i] != -50)
                ind_train[e,i, :len(temp[0])] = temp[0]
        ind_train = np.sort(ind_train, axis=2)

        # Set up some help for the selections
        avg_idx = np.arange(averages)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
        elem_idx = np.arange(elements_to_average)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

        chunk_size_e = self.max_elements_to_process
        n_chunks_e = int(np.ceil(elements_to_average / chunk_size_e))
        chunk_size_a = self.max_elements_to_process
        n_chunks_a = int(np.ceil(averages / chunk_size_a))

        for chunk_id_a in range(n_chunks_a):
            start_a = chunk_id_a * chunk_size_a
            end_a = (chunk_id_a + 1) * chunk_size_a
            if end_a > averages:
                end_a = averages

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = (chunk_id_e + 1) * chunk_size_e
                if end_e > elements_to_average:
                    end_e = elements_to_average

                avg_idx = np.arange(start_a, end_a)[:, np.newaxis, np.newaxis]        # shape: (averages, 1, 1)
                elem_idx = np.arange(start_e, end_e)[np.newaxis, :, np.newaxis]  # shape: (1, elements_to_average, 1)

                self.log.write("info",f"Proccesing up to {(avg_idx.flatten()[-1] + 1) * (elem_idx.flatten()[-1]+1)}/{self.nelv} element")

                avg_idx2 = avg_idx.reshape(avg_idx.shape[0], 1)
                elem_idx2 = elem_idx.reshape(1, elem_idx.shape[1])
                
                # Get covariance matrix
                kw = self._get_covariance_matrix(settings, field_name, avg_idx2, elem_idx2)


                # Get the prediction and the standard deviation
                y_21, y_21_std = self.gaussian_process_regression(y, V, kw, ind_train, avg_idx, elem_idx, avg_idx2, elem_idx2, predict_mean=get_mean, predict_std=get_std) 

                # This is still with column vectors at the end. We need to reshape it.
                if get_mean:
                    y_reconstructed[avg_idx2, elem_idx2] = y_21
                if get_std:
                    y_reconstructed_std[avg_idx2, elem_idx2] = y_21_std.reshape(y_21_std.shape[0], y_21_std.shape[1], y_21_std.shape[2], 1)

        if get_mean:
            y_reconstructed = y_reconstructed.reshape(sampled_field.shape)
            y_reconstructed[sampled_field != -50] = sampled_field[sampled_field != -50]
        if get_std:
            y_reconstructed_std = y_reconstructed_std.reshape(sampled_field.shape)

        # Reshape the field back to its original shape
        return y_reconstructed, y_reconstructed_std

    def reconstruct_field_torch(self, field_name: str = None, get_mean: bool = True, get_std: bool = False, mean_op = None, std_op = None):
        """
        Reconstructs the field using Gaussian Process Regression in PyTorch.
        """

        # Retrieve the sampled field
        sampled_field = self.uncompressed_data[field_name]["field"]
        settings = self.settings

        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]

        # Retrieve the number of samples
        n_samples = settings["compression"]["n_samples"]

        # Move data to the same device as sampled_field
        device = sampled_field.device

        # Reshape the fields into the KW-supported shapes
        y = sampled_field.reshape(averages, elements_to_average, -1, 1)  # Shape: (averages, elements_to_average, all_dim, 1)
        V = self.v_d
        if settings["covariance"]["method"] == "svd":
            V = self.uncompressed_data[field_name]["U"]
        numfreq = n_samples
         
        noise_ = self.uncompressed_data[field_name].get("noise", None)
        if not isinstance(noise_, type(None)):
            noise = noise_.view(averages, elements_to_average, 1, 1)
        else:
            noise = None

        # Allocate storage for reconstructed fields
        y_reconstructed = None
        y_reconstructed_std = None
        if get_mean:
            y_reconstructed = torch.full_like(y, -50, device=device)
        if get_std:
            y_reconstructed_std = torch.full_like(y, -50, device=device)

        # Create an array that contains the indices of the sampled elements
        ind_train = torch.zeros((averages, elements_to_average, n_samples), dtype=torch.long, device=device)

        # Populate ind_train using PyTorch indexing
        for e in range(averages):
            for i in range(elements_to_average):
                temp = torch.where(y[e, i] != -50)
                ind_train[e, i, :temp[0].numel()] = temp[0]

        ind_train, _ = torch.sort(ind_train, dim=2)  # Ensure indices are sorted

        # Set up indexing tensors for selections
        chunk_size_e = self.max_elements_to_process
        n_chunks_e = int(torch.ceil(torch.tensor(elements_to_average / chunk_size_e)).item())
        chunk_size_a = self.max_elements_to_process
        n_chunks_a = int(torch.ceil(torch.tensor(averages / chunk_size_a)).item())

        for chunk_id_a in range(n_chunks_a):
            start_a = chunk_id_a * chunk_size_a
            end_a = min((chunk_id_a + 1) * chunk_size_a, averages)

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, elements_to_average)

                avg_idx = torch.arange(start_a, end_a, device=device).view(-1, 1, 1)  # Shape: (chunk_a, 1, 1)
                elem_idx = torch.arange(start_e, end_e, device=device).view(1, -1, 1)  # Shape: (1, chunk_e, 1)

                self.log.write("info", f"Processing up to {(avg_idx.flatten()[-1] + 1) * (elem_idx.flatten()[-1] + 1)}/{self.nelv} elements")

                avg_idx2 = avg_idx.view(avg_idx.shape[0], 1)  # Shape: (chunk_a, 1)
                elem_idx2 = elem_idx.view(1, elem_idx.shape[1])  # Shape: (1, chunk_e)

                # Get covariance matrix using the PyTorch version
                kw = self._get_covariance_matrix(settings, field_name, avg_idx2, elem_idx2)

                # Get the prediction and the standard deviation
                y_21, y_21_std = self.gaussian_process_regression_torch(
                    y, V, kw, ind_train, avg_idx, elem_idx, avg_idx2, elem_idx2,
                    predict_mean=get_mean, predict_std=get_std, mean_op = mean_op, std_op = std_op,
                    noise=noise
                )

                # Store reconstructed values
                if get_mean:
                    y_reconstructed[avg_idx2, elem_idx2] = y_21
                if get_std:
                    y_reconstructed_std[avg_idx2, elem_idx2] = y_21_std.view(*y_21_std.shape[:3], 1)

        # Final reshaping back to original shape
        mask = sampled_field != -50
        if get_mean:
            y_reconstructed = y_reconstructed.reshape(sampled_field.shape)
            y_reconstructed[mask] = sampled_field[mask]
        if get_std:
            y_reconstructed_std = y_reconstructed_std.reshape(sampled_field.shape)
            #y_reconstructed_std[mask] = 0

        return y_reconstructed, y_reconstructed_std
    
    def reconstruct_field_torch_coefficients(self, field_name: str = None):
        """
        Reconstructs the field using Gaussian Process Regression in PyTorch.
        """

        settings = self.settings
        # Set the reshaping parameters
        averages = settings["covariance"]["averages"]
        elements_to_average = settings["covariance"]["elements_to_average"]

        try:
            y = self.uncompressed_data[field_name]["f_hat"]
        except KeyError:
            raise KeyError(f"Field {field_name} not found in uncompressed_data - make sure you used DLT when compressing, otherwise remove the use_coefficients option")
        
        # Allocate storage for reconstructed fields
        y_reconstructed_std = None

        y_reconstructed = self.transform_field(field = y, to="physical")

        return y_reconstructed, y_reconstructed_std
 
    def _estimate_covariance_average(self, field_hat : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            # Retrieve the settings
            averages=settings["averages"]
            elements_to_average=settings["elements_to_average"]

            # Create an average of field_hat over the elements
            temp_field = field_hat.reshape(averages, elements_to_average, field_hat.shape[1], field_hat.shape[2], field_hat.shape[3])
            field_hat_mean = np.mean(temp_field, axis=1)        
            
            ### This block was to average with weights, but the coefficients do not really have that sort of mass matrix.
            ##temp_mass = self.B.reshape(averages, elements_to_average, self.B.shape[1], self.B.shape[2], self.B.shape[3])
            #
            ## Perform a weighted average with the mass matrix
            ##field_hat_mean = np.sum(temp_field * temp_mass, axis=1) / np.sum(temp_mass, axis=1)
            ###

            # This is the way in which I calculate the covariance here and then get the diagonals
            if self.kw_diag == True:
                # Get the covariances
                kw = np.einsum("eik,ekj->eij", field_hat_mean.reshape(averages,-1,1), field_hat_mean.reshape(averages,-1,1).transpose(0,2,1))

                # Extract only the diagonals
                kw = np.einsum("...ii->...i", kw)
            else:
                # But I can leave the calculation of the covariance itself for later and store here the average of field_hat
                kw = field_hat_mean.reshape(averages,-1,1)

        elif self.bckend == "torch":
            # Retrieve the settings
            averages=settings["averages"]
            elements_to_average=settings["elements_to_average"]

            # Create an average of field_hat over the elements
            fh = field_hat.reshape(averages, elements_to_average, -1, 1)
            fh_bar = torch.mean(fh, dim=1, keepdim=True)
            fh_centered = fh - fh_bar
            fh_star = torch.matmul(fh_centered, fh_centered.transpose(-1,-2))
            kw = torch.sum(fh_star, dim=1) / (fh_star.shape[1] - 1)
            # This is the way in which I calculate the covariance here and then get the diagonals
            if self.kw_diag == True:
                
                # Extract only the diagonals
                kw = torch.einsum("...ii->...i", kw)

            
        return kw
    
    def _estimate_covariance_svd(self, field : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            raise NotImplementedError("SVD is not implemented in NumPy")

        elif self.bckend == "torch":

            # Retrieve the settings
            averages = settings["averages"]  # In the case of SVD, this is 1
            elements_to_average = settings["elements_to_average"]  # In the case of SVD, this is the number of elements in the rank.

            # Create a snapshot matrix S = (Data in the element, element)
            S = field.reshape(averages * elements_to_average,
                                field.shape[1] * field.shape[2] * field.shape[3]).clone()
            S = S.permute(1, 0).clone()

            # Perform the SVD using torch.linalg.svd (set full_matrices=False to match NumPy's behavior)
            U, s, Vt = torch.linalg.svd(S, full_matrices=False)

            # Keep only the first keep_modes

            fh = torch.diag(s)@Vt
            fh = fh.reshape(fh.shape[0], 1, fh.shape[1])
            fh = fh.permute(2,0,1)
            fh_bar = torch.mean(fh, dim=0, keepdim=True)
            
            fh_centered = fh - fh_bar
            fh_star = torch.matmul(fh_centered, fh_centered.transpose(-1,-2))
            kw = torch.sum(fh_star, dim=0) / (fh_star.shape[1] - 1)
            
            # Give it a shape conformant with averages
            kw = kw.reshape(1, kw.shape[0], kw.shape[1])

            # This is the way in which I calculate the covariance here and then get the diagonals
            if self.kw_diag == True:
                
                # Extract only the diagonals
                kw = torch.einsum("...ii->...i", kw)


        return U, kw
    
    def _estimate_covariance_dlt(self, field_hat : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            nelv = int(settings['averages'] * settings['elements_to_average'])
            n_samples = settings["keep_modes"]

            # Get needed information
            V = self.v
            numfreq = n_samples

            # Now reshape the x, y elements into column vectors
            y = field_hat.reshape(field_hat.shape[0], -1)

            #allocation the truncated field
            y_truncated = np.copy(y)

            # Set up chunking parameters to avoid processing too many elements at once.
            chunk_size_e = self.max_elements_to_process
            n_chunks_e = math.ceil(nelv / chunk_size_e)

            # Loop over chunks along the element dimension.
            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, nelv)

                # Create chunk-specific index helpers.
                elem_idx = np.arange(start_e, end_e)

                # Get the sorted coefficients in each element in descending order
                ind = np.argsort(np.abs(y[elem_idx, :]), axis=1)[:, ::-1]

                # Set the entries after the numfreq-th to zero
                y_truncated[elem_idx.reshape(-1,1), ind[:, numfreq:]] = 0

        elif self.bckend == "torch":

            nelv = int(settings['averages'] * settings['elements_to_average'])
            n_samples = settings["keep_modes"]
            
            # Reshape so that we have [nelv, -1]
            y = field_hat.reshape(field_hat.shape[0], -1)  # shape: (nelv, ?)

            # Make a copy for truncation
            y_truncated = y.clone()

            # Prepare chunking
            chunk_size_e = self.max_elements_to_process
            n_chunks_e = math.ceil(nelv / chunk_size_e)

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, nelv)

                # Get the row indices for this chunk
                elem_idx = torch.arange(start_e, end_e, device=y.device)  # shape: (chunk_size,)

                # Extract the sub-tensor for this chunk: (chunk_size, ?)
                sub_y = y[elem_idx, :]

                # Sort indices by absolute value in descending order along dim=1
                ind = torch.argsort(torch.abs(sub_y), dim=1, descending=True)

                # Keep only the top n_samples indices
                col_idx_to_zero = ind[:, n_samples:]

                # Construct row indices (broadcasted) to match the shape of col_idx_to_zero
                row_idx = elem_idx.unsqueeze(1).expand(-1, col_idx_to_zero.shape[1])

                # Set those positions to zero
                y_truncated[row_idx, col_idx_to_zero] = 0

        # Reshape back to the original shape
        return y_truncated.reshape(field_hat.shape)
    
    def _estimate_covariance_bitplane(self, field_hat : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            nelv = int(settings['averages'] * settings['elements_to_average'])
            n_samples = settings["keep_modes"]
            initial = 64 - n_samples
            packed_size = int(np.ceil((field_hat.shape[1]*field_hat.shape[2]*field_hat.shape[3]) / 8))

            # Get needed information
            V = self.v
            numfreq = n_samples

            # Now reshape the x, y elements into column vectors
            y = field_hat.reshape(field_hat.shape[0], -1)

            #allocation the truncated field
            y_packed = np.empty((y.shape[0], n_samples, packed_size), dtype=np.uint8)
            y_min = np.empty((y.shape[0], 1), dtype=np.float64)
            y_max = np.empty((y.shape[0], 1), dtype=np.float64)

            # Set up chunking parameters to avoid processing too many elements at once.
            chunk_size_e = self.max_elements_to_process
            n_chunks_e = math.ceil(nelv / chunk_size_e)
                
            # For reconstruction
            shape_y = y.shape
            y_reconstructed = np.zeros(shape_y)

            # Loop over chunks along the element dimension.
            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, nelv)

                # Create chunk-specific index helpers.
                elem_idx = np.arange(start_e, end_e)

                y_batch = y[elem_idx, :]

                y_batch_max = np.max(y_batch, axis=1)
                y_batch_min = np.min(y_batch, axis=1)
                y_batch_norm = (y_batch - y_batch_min[:, np.newaxis]) / (y_batch_max - y_batch_min)[:, np.newaxis]
                y_batch_norm = np.clip(y_batch_norm, 0, 0.9999999999999999)
                
                max_uint64 = 2**64 - 1
                y_batch_quant = np.floor(y_batch_norm * max_uint64).astype(np.uint64)
                
                y_batch_packed = np.empty((y_batch_quant.shape[0], n_samples, packed_size), dtype=np.uint8)
                for i in range(initial, 64):
                    # Extract the i-th bit plane as a uint8 array (each element is 0 or 1)
                    bit_plane = ((y_batch_quant >> i) & 1).astype(np.uint8)
                     
                    # Pack the bits so that 8 bits are stored in one byte using np.packbits.
                    packed = np.packbits(bit_plane, axis=1)
                    y_batch_packed[:, i - initial] = packed
                
                y_packed[elem_idx] = y_batch_packed
                y_min[elem_idx,0] = y_batch_min
                y_max[elem_idx,0] = y_batch_max

        elif self.bckend == "torch":

            raise NotImplementedError("The bitplane method is not implemented for the torch backend")

        # Reshape back to the original shape
        return y_packed, y_max, y_min, shape_y
    
    
    def _estimate_covariance_ad_rmse(self, field_hat : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            raise NotImplementedError("The AD method is not implemented for the numpy backend")
        elif self.bckend == "torch":

            nelv = int(settings['averages'] * settings['elements_to_average'])
            n_samples = settings["keep_modes"]
            
            # Reshape so that we have [nelv, -1]
            y = self.field.reshape(self.field.shape[0], -1) 
            y_hat = field_hat.reshape(field_hat.shape[0], -1) 
            mass = self.b_d.reshape(self.b_d.shape[0], -1)

            y_truncated = y_hat.clone().detach()

            # Prepare chunking
            chunk_size_e = self.max_elements_to_process
            n_chunks_e = math.ceil(nelv / chunk_size_e)

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, nelv)

                elem_idx = torch.arange(start_e, end_e, device=y_hat.device) 

                y_real = y[elem_idx, :]
                y_real = y_real.view(1, y_real.shape[0], y_real.shape[1], 1)
                mass_real = mass[elem_idx, :]
                mass_real = mass_real.view(1, mass_real.shape[0], mass_real.shape[1], 1)

                y_samples = y[elem_idx, :]
                y_samples = y_samples.view(1, y_samples.shape[0], y_samples.shape[1], 1)

                y_h = y_hat[elem_idx, :].clone()
                y_h = y_h.view(1, y_h.shape[0], y_h.shape[1], 1)

                lamb = 0.01
                y_ad = y_hat[elem_idx, :].clone().requires_grad_(True) * lamb
                y_ad = y_ad.view(1, y_ad.shape[0], y_ad.shape[1], 1)
                y_ad.retain_grad()  # Enable gradient storage for non-leaf tensor
                
                V = self.v_d.reshape(1, 1, self.v_d.shape[0], self.v_d.shape[1])

                y_approx = torch.matmul(V, y_h + y_ad)

                rmse = torch.sqrt(torch.sum((y_real - y_approx) ** 2 * mass_real, dim = (2,3), keepdim=False) / torch.sum(mass_real, dim=(2,3), keepdim=False))
                rmse.backward(torch.ones_like(rmse), retain_graph=True)
                grad = y_ad.grad.view(y_ad.shape[1], y_ad.shape[2])

                # Sort indices by absolute value in descending order along dim=1
                ind = torch.argsort(torch.abs(grad), dim=1, descending=True)

                # Keep only the top n_samples indices
                col_idx_to_zero = ind[:, n_samples:]

                # Construct row indices (broadcasted) to match the shape of col_idx_to_zero
                row_idx = elem_idx.unsqueeze(1).expand(-1, col_idx_to_zero.shape[1])

                # Set those positions to zero
                y_truncated[row_idx, col_idx_to_zero] = 0

        # Reshape back to the original shape
        return y_truncated.reshape(field_hat.shape)
    
    def _estimate_covariance_ad_ml(self, field_hat : np.ndarray, settings: dict):

        if self.bckend == "numpy":
            raise NotImplementedError("The AD method is not implemented for the numpy backend")
        elif self.bckend == "torch":

            nelv = int(settings['averages'] * settings['elements_to_average'])
            interval = 1
            n_coefficients = settings["keep_modes"]
            
            # Reshape so that we have [nelv, -1]
            y = self.field.reshape(self.field.shape[0], -1) 
            y_hat = field_hat.reshape(field_hat.shape[0], -1) 
            mass = self.b_d.reshape(self.b_d.shape[0], -1)

            y_truncated = y_hat.clone().detach()

            # Prepare chunking
            chunk_size_e = self.max_elements_to_process
            n_chunks_e = math.ceil(nelv / chunk_size_e)

            for chunk_id_e in range(n_chunks_e):
                start_e = chunk_id_e * chunk_size_e
                end_e = min((chunk_id_e + 1) * chunk_size_e, nelv)

                elem_idx = torch.arange(start_e, end_e, device=y_hat.device) 

                y_real = y[elem_idx, :]
                y_real = y_real.view(1, y_real.shape[0], y_real.shape[1], 1)
                
                mass_real = mass[elem_idx, :]
                mass_real = mass_real.view(1, mass_real.shape[0], mass_real.shape[1], 1)

                y_samples = y[elem_idx, :-1:interval]
                y_samples = y_samples.view(1, y_samples.shape[0], y_samples.shape[1], 1)

                y_h = y_hat[elem_idx, :].clone().detach()
                y_h = y_h.view(1, y_h.shape[0], y_h.shape[1], 1)
                
                y_ad = y_hat[elem_idx, :].clone().requires_grad_(True)
                y_ad = y_ad.view(1, y_ad.shape[0], y_ad.shape[1], 1)
                y_ad.retain_grad()  # Enable gradient storage for non-leaf tensor

                # Create a non linear perturbation such that it does not cancel out in the regression
                ## Note: Recall that the covariance is important for the structure, not the scale.
                scaling = 0.1
                y_ad_min = y_ad.min()
                y_ad_max = y_ad.max()
                minmax = 2 * (y_ad - y_ad_min) / (y_ad_max - y_ad_min) - 1
                y_pert = torch.tanh(minmax) * scaling

                V22 = self.v_d.reshape(1, 1, self.v_d.shape[0], self.v_d.shape[1])
                V11 = V22[:, :, :-1:interval, :]

                y_cov = y_h + y_pert * y_h

                # Get the covariance matrix
                kw = torch.einsum("aeik,aekj->aeij", y_cov, y_cov.permute(0, 1, 3, 2))
                temp = torch.matmul(V11, kw)  
                K11 = torch.matmul(temp, V11.permute(0,1, 3, 2))   
                K12 = torch.matmul(temp, V22.permute(0,1, 3, 2))
                K21 = K12.transpose(-1, -2)
                eps = 1e-10 * torch.eye(K11.shape[-1], device=K11.device, dtype=K11.dtype).reshape(1, 1, K11.shape[-1], K11.shape[-1])

                # Check if K11 is symetric
                if not torch.allclose(K11, K11.permute(0, 1, 3, 2)):
                    raise ValueError("K11 is not symmetric")

                # From rasmussen 2006 - marginal likelihood
                LU, pivots = torch.linalg.lu_factor(K11 + eps)

                alpha = torch.linalg.lu_solve(LU, pivots, y_samples)

                if 1 == 1:
                    U_diag = torch.diagonal(LU, dim1=-2, dim2=-1).clone()
                    log_det = torch.sum(torch.log(torch.abs(U_diag)))
                    
                    marginal_likelihood = (
                        0.5 * torch.matmul(y_samples.transpose(-1, -2), alpha) +
                        log_det +
                        0.5 * y_ad.shape[0] * y_ad.shape[1] *
                            torch.log(torch.tensor(2 * np.pi, dtype=y_ad.dtype))
                    )
                    marginal_likelihood = torch.sum(marginal_likelihood, dim=(2, 3))
                    
                    marginal_likelihood.backward(torch.ones_like(marginal_likelihood), retain_graph=True)
                else:
                    y_approx = torch.matmul(K21, alpha)
                    rmse = torch.sqrt(
                        torch.sum((y_real - y_approx) ** 2 * mass_real, dim=(2, 3), keepdim=False) /
                        torch.sum(mass_real, dim=(2, 3), keepdim=False)
                    )
                    rmse.backward(torch.ones_like(rmse), retain_graph=True)

                grad = y_ad.grad.view(y_ad.shape[1], y_ad.shape[2])
                
                # Sort indices by absolute value in descending order along dim=1
                ind = torch.argsort(torch.abs(grad), dim=1, descending=True)

                # Keep only the top n_samples indices
                col_idx_to_zero = ind[:, n_coefficients:]

                # Construct row indices (broadcasted) to match the shape of col_idx_to_zero
                row_idx = elem_idx.unsqueeze(1).expand(-1, col_idx_to_zero.shape[1])

                # Set those positions to zero
                y_truncated[row_idx, col_idx_to_zero] = 0
                
                y_ad.grad.zero_()

        # Reshape back to the original shape
        return y_truncated.reshape(field_hat.shape)
 
    def transform_field(self, field: np.ndarray = None, to: str = "legendre") -> np.ndarray:
        """
        Transform the field to the desired space
        
        Args:
            field (np.ndarray): Field to be transformed
            to (str): Space to which the field will be transformed
        
        Returns:
            np.ndarray: Transformed field
        """

        if self.bckend == "numpy":
            if to == "legendre":
                return apply_operator(self.vinv, field)
            elif to == "physical":
                return apply_operator(self.v, field)
            else:
                raise ValueError("Invalid space to transform the field to")
        elif self.bckend == "torch":
            if to == "legendre":
                return torch_apply_operator(self.vinv_d, field)
            elif to == "physical":
                return torch_apply_operator(self.v_d, field)
            else:
                raise ValueError("Invalid space to transform the field to")

    def _get_covariance_matrix(self, settings: dict, field_name: str, avg_idx2: np.ndarray, elem_idx2: np.ndarray):
        """
        """

        if self.bckend == "numpy":

            averages2 = avg_idx2.shape[0]
            elements_to_average2 = elem_idx2.shape[1]
            
            if settings["covariance"]["method"] == "average":
                if self.kw_diag == True:        
                    # Retrieve the diagonal of the covariance matrix
                    kw = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:,0]]
            
                    # Transform it into an actual matrix, not simply a vector
                    # Aditionally, add one axis to make it consistent with the rest of the arrays and enable broadcasting
                    kw_ = np.einsum('...i,ij->...ij', kw, np.eye(kw.shape[-1])).reshape(averages2, 1 ,  kw.shape[-1], kw.shape[-1])
                else:
                    # Retrieve the averaged hat fields
                    f_hat = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:,0]]
                    # Calculate the covariance matrix with f_hat@f_hat^T
                    kw = np.einsum("eik,ekj->eij", f_hat, f_hat.transpose(0,2,1))
                    # Add an axis to make it consistent with the rest of the arrays and enable broadcasting
                    kw_ = kw.reshape(kw.shape[0], 1, kw.shape[1], kw.shape[2])
                
                kw = kw_
            
            elif settings["covariance"]["method"] == "svd":

                raise NotImplementedError("SVD is not implemented in NumPy")
            
            elif settings["covariance"]["method"] == "dlt":

                averages = settings["covariance"]["averages"]
                elements_to_average = settings["covariance"]["elements_to_average"]
                
                # Retrieve the truncated field
                f_hat_full = self.uncompressed_data[f"{field_name}"]["f_hat"].reshape(averages, elements_to_average, -1, 1)
                f_hat = f_hat_full[avg_idx2, elem_idx2, :, :]
                
                # Calculate the covariance matrix with f_hat @ f_hat.T
                if self.kw_diag:
                    # Reshape f_hat so that each row becomes a matrix column vector
                    f_hat_reshaped = f_hat.reshape(averages2 * elements_to_average2, -1, 1)
                    # Compute the covariance matrices for each entry: f_hat @ f_hat.T
                    kw_ = np.einsum("eik,ekj->eij", f_hat_reshaped, np.transpose(f_hat_reshaped, (0, 2, 1)))
                    # Extract only the diagonals
                    kw_diag = np.einsum("...ii->...i", kw_)
                    # Convert the diagonal vector into a full matrix by multiplying with an identity matrix
                    eye = np.eye(kw_diag.shape[-1], dtype=kw_diag.dtype)
                    kw_ = np.einsum("...i,ij->...ij", kw_diag, eye)
                    # Reshape so that the result has shape (averages2, elements_to_average2, n, n)
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[-2], kw_.shape[-1])
                else:
                    # Reshape f_hat and compute the covariance matrices
                    f_hat_reshaped = f_hat.reshape(averages2 * elements_to_average2, -1, 1)
                    kw_ = np.einsum("eik,ekj->eij", f_hat_reshaped, np.transpose(f_hat_reshaped, (0, 2, 1)))
                    # Add an axis for broadcasting and reshape accordingly
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[1], kw_.shape[2])
                
                # Ensure a copy is made (np.copy() is used in place of torch.clone())
                kw = np.copy(kw_)

            elif settings["covariance"]["method"] == "bitplane":
                
                averages = settings["covariance"]["averages"]
                elements_to_average = settings["covariance"]["elements_to_average"]
                n_samples = settings["covariance"]["keep_modes"]
                initial = 64 - n_samples

                # Reconstruct the field hat from the bitplanes
                y_packed = self.uncompressed_data[f"{field_name}"]["f_hat_packed"]
                y_max = self.uncompressed_data[f"{field_name}"]["f_hat_max"]
                y_min = self.uncompressed_data[f"{field_name}"]["f_hat_min"]
                shape_y = self.uncompressed_data[f"{field_name}"]["f_hat_shape"]

                # Inverse the process
                y_batch_packed = y_packed[avg_idx2.flatten()]
                y_batch_min = y_min[avg_idx2.flatten()]
                y_batch_max = y_max[avg_idx2.flatten()]

                # Reconstruct the field
                y_batch_quant_r = np.zeros((y_batch_packed.shape[0], shape_y[1]), dtype=np.uint64)
                for i in range(initial, 64):
                    unpacked = np.unpackbits(y_batch_packed[:, i - initial], axis=1)
                    y_batch_quant_r += unpacked.astype(np.uint64) << i

                max_trunc = 2**64 - 2**int(initial)
                y_batch_quant_r = y_batch_quant_r.astype(np.float64)/max_trunc
                
                f_hat = y_batch_quant_r * (y_batch_max - y_batch_min) + y_batch_min
 
                # Calculate the covariance matrix with f_hat @ f_hat.T
                if self.kw_diag:
                    # Reshape f_hat so that each row becomes a matrix column vector
                    f_hat_reshaped = f_hat.reshape(averages2 * elements_to_average2, -1, 1)
                    # Compute the covariance matrices for each entry: f_hat @ f_hat.T
                    kw_ = np.einsum("eik,ekj->eij", f_hat_reshaped, np.transpose(f_hat_reshaped, (0, 2, 1)))
                    # Extract only the diagonals
                    kw_diag = np.einsum("...ii->...i", kw_)
                    # Convert the diagonal vector into a full matrix by multiplying with an identity matrix
                    eye = np.eye(kw_diag.shape[-1], dtype=kw_diag.dtype)
                    kw_ = np.einsum("...i,ij->...ij", kw_diag, eye)
                    # Reshape so that the result has shape (averages2, elements_to_average2, n, n)
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[-2], kw_.shape[-1])
                else:
                    # Reshape f_hat and compute the covariance matrices
                    f_hat_reshaped = f_hat.reshape(averages2 * elements_to_average2, -1, 1)
                    kw_ = np.einsum("eik,ekj->eij", f_hat_reshaped, np.transpose(f_hat_reshaped, (0, 2, 1)))
                    # Add an axis for broadcasting and reshape accordingly
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[1], kw_.shape[2])
                
                # Ensure a copy is made (np.copy() is used in place of torch.clone())
                kw = np.copy(kw_)
            
     
        elif self.bckend == 'torch':

            averages2 = avg_idx2.shape[0]
            elements_to_average2 = elem_idx2.shape[1]

            if settings["covariance"]["method"] == "average" or settings["covariance"]["method"] == "svd":
                if self.kw_diag:
                    # Retrieve the diagonal of the covariance matrix
                    kw = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:, 0]]
                    # Transform it into an actual matrix (not just a vector) and add an extra axis
                    eye = torch.eye(kw.shape[-1], device=kw.device, dtype=kw.dtype)
                    kw_ = torch.einsum("...i,ij->...ij", kw, eye).reshape(averages2, 1, kw.shape[-1], kw.shape[-1])
                else:
                    # Retrieve the full covariance matrix
                    kw = self.uncompressed_data[f"{field_name}"]["kw"][avg_idx2[:, 0]]
                    # Reshape it to expected shape
                    kw_ = kw.view(kw.shape[0], 1, kw.shape[1], kw.shape[2])
                
                kw = kw_
 
            elif (settings["covariance"]["method"] == "dlt") or (settings["covariance"]["method"] == "ad"):

                self.log.write("debug", f"Obtaining the covariance matrix for the current chunk")

                averages = settings["covariance"]["averages"]
                elements_to_average = settings["covariance"]["elements_to_average"]

                # Retrieve the truncated field
                f_hat_full = self.uncompressed_data[f"{field_name}"]["f_hat"].view(averages, elements_to_average, -1, 1)
                f_hat = f_hat_full[avg_idx2, elem_idx2, :, :]

                # Calculate the covariance matrix with f_hat@f_hat^T
                if self.kw_diag:
                    # Reshape f_hat so that each row becomes a matrix column vector
                    f_hat_reshaped = f_hat.view(averages2 * elements_to_average2, -1, 1)
                    # Compute the covariance matrices for each entry: f_hat @ f_hat^T
                    kw_ = torch.einsum("eik,ekj->eij", f_hat_reshaped, f_hat_reshaped.permute(0, 2, 1))
                    # Extract only the diagonals
                    kw_diag = torch.einsum("...ii->...i", kw_)
                    # Convert the diagonal vector into a full matrix by multiplying with an identity matrix
                    eye = torch.eye(kw_diag.shape[-1], device=kw_diag.device, dtype=kw_diag.dtype)
                    kw_ = torch.einsum("...i,ij->...ij", kw_diag, eye)
                    # Reshape so that the result has shape (averages2, elements_to_average2, n, n)
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[-2], kw_.shape[-1])
                else:
                    # Reshape f_hat and compute the covariance matrices
                    f_hat_reshaped = f_hat.view(averages2 * elements_to_average2, -1, 1)
                    kw_ = torch.einsum("eik,ekj->eij", f_hat_reshaped, f_hat_reshaped.permute(0, 2, 1))
                    # Add an axis for broadcasting and reshape accordingly
                    kw_ = kw_.reshape(averages2, elements_to_average2, kw_.shape[1], kw_.shape[2])
                
                # Ensure a copy is made if needed (torch.clone() is used in place of np.copy)
                kw = kw_.clone() 

        return kw
    
    def gaussian_process_regression(self, y: np.ndarray, V: np.ndarray, kw: np.ndarray, 
                                    ind_train: np.ndarray, avg_idx: np.ndarray, elem_idx: np.ndarray,
                                    avg_idx2: np.ndarray, elem_idx2: np.ndarray, freq: int = None, predict_mean: bool = True, predict_std: bool = True):

        # select the correct freq index:
        if freq is None:
            freq_idex = slice(None)
        else:
            freq_idex = slice(freq+1)

        # Select the current samples            
        y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2,elem_idx2,freq_idex],:]

        V_11 = V[ind_train[avg_idx2,elem_idx2,freq_idex], :]
        V_22 = V.reshape(1,1,V.shape[0],V.shape[1])

        ## Get covariance matrices
        ## This approach was faster than using einsum. Potentially due to the size of the matrices
        ### Covariance of the sampled entries
        temp = np.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
        k11 = np.matmul(temp, np.swapaxes(V_11, -1, -2))  # results in shape: (averages, elements_to_average, freq+1, freq+1)
        ### Covariance of the predictions
        temp = np.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
        k12 = np.matmul(temp, np.swapaxes(V_22, -1, -2))  # if V_22 is shaped appropriately
        k21 = k12.transpose(0, 1, 3, 2)

        temp = np.matmul(V_22, kw)  # shape: (averages, elements_to_average, n, n)
        k22 = np.matmul(temp, np.swapaxes(V_22, -1, -2))  # results in shape: (averages, elements_to_average, n, n)

        # Make predictions to sample
        ## Create some matrices to stabilize the inversions
        eps = 1e-10*np.eye(k11.shape[-1]).reshape(1,1,k11.shape[-1],k11.shape[-1])
        ## Predict the mean and covariance matrix of all entires (data set 2) given the known samples (data set 1)

        y_21 = None
        if predict_mean:
            y_21= k21@np.linalg.inv(k11+eps)@(y_11)

        y_21_std = None
        if predict_std:    
            sigma21 = k22 - (k21@np.linalg.inv(k11+eps)@k12)           
            ## Predict the standard deviation of all entires (data set 2) given the known samples (data set 1)
            y_21_std = np.sqrt(np.abs(np.einsum("...ii->...i", sigma21)))

        return y_21, y_21_std


    def gaussian_process_regression_torch(self, y: torch.Tensor, V: torch.Tensor, kw: torch.Tensor, 
                                    ind_train: torch.Tensor, avg_idx: torch.Tensor, elem_idx: torch.Tensor,
                                    avg_idx2: torch.Tensor, elem_idx2: torch.Tensor, 
                                    freq: int = None, predict_mean: bool = True, predict_std: bool = True,
                                    method = 'lu', mean_op = None, std_op = None, noise = None):
        
        # Select the correct freq index:
        if freq is None:
            freq_idex = slice(None)
        else:
            freq_idex = slice(freq + 1)

        # If updating the noise, remove the unsampled field
        if not isinstance(noise, type(None)):
            predict_mean = True 

        # Select the current samples (using advanced indexing)
        y_11 = y[avg_idx, elem_idx, ind_train[avg_idx2, elem_idx2, freq_idex], :]

        V_11 = V[ind_train[avg_idx2, elem_idx2, freq_idex], :]
        V_22 = V.reshape(1, 1, V.shape[0], V.shape[1])

        # Get covariance matrices
        if self.kw_diag:
            # Extract the diagonal elements of kw (shape: (b1, b2, n))
            kw_diag = torch.diagonal(kw, dim1=-2, dim2=-1)  # Only the diagonal elements

            # Compute covariance matrices efficiently
            # Covariance of the sampled entries
            temp = V_11 * kw_diag.unsqueeze(-2)  # Broadcasting (shape: (b1, b2, freq+1, n))
            k11 = torch.matmul(temp, V_11.transpose(-1, -2))  # Shape: (b1, b2, freq+1, freq+1)

            # Covariance for predictions
            k12 = torch.matmul(temp, V_22.transpose(-1, -2))  # Shape: (b1, b2, freq+1, n)
            k21 = k12.transpose(-2, -1)  # Shape: (b1, b2, n, freq+1)

            # Covariance of the real data
            temp = V_22 * kw_diag.unsqueeze(-2)  # Broadcasting (shape: (b1, b2, n, n))
            k22 = torch.matmul(temp, V_22.transpose(-1, -2))  # Shape: (b1, b2, n, n)

        else:

            # Covariance of the sampled entries
            temp = torch.matmul(V_11, kw)  # shape: (averages, elements_to_average, freq+1, n)
            k11 = torch.matmul(temp, V_11.transpose(-1, -2))  # shape: (averages, elements_to_average, freq+1, freq+1)
            
            # Covariance for predictions 
            k12 = torch.matmul(temp, V_22.transpose(-1, -2))  # shape: (averages, elements_to_average, freq+1, n)
            k21 = k12.permute(0, 1, 3, 2)  # shape: (averages, elements_to_average, n, freq+1)

            # Covariance of the real data
            temp = torch.matmul(V_22, kw)  # shape: (averages, elements_to_average, n, n)
            k22 = torch.matmul(temp, V_22.transpose(-1, -2))  # shape: (averages, elements_to_average, n, n)
 
        # Create a small epsilon for numerical stability in inversion.
        if isinstance(noise, type(None)):
            eps = 1e-10 * torch.eye(k11.shape[-1], device=k11.device, dtype=k11.dtype).reshape(1, 1, k11.shape[-1], k11.shape[-1])
        else:
            eps = noise[avg_idx2, elem_idx2]**2 * torch.eye(k11.shape[-1], device=k11.device).reshape(1, 1, k11.shape[-1], k11.shape[-1])
        

        if method == 'naive':
            y_21 = None
            if predict_mean:
                # Predict the mean: y_21 = k21 * inv(k11 + eps) * y_11
                y_21 = torch.matmul(k21, torch.linalg.solve(k11 + eps, y_11))
                
                if mean_op is not None:
                    y_21 = torch.matmul(mean_op.view(1,1,V.shape[0],V.shape[1]), y_21)

            y_21_std = None
            if predict_std:
                # Predict the covariance of predictions
                sigma21 = k22 - torch.matmul(k21, torch.linalg.solve(k11 + eps, k12))

                if std_op is not None:
                    sigma21 = torch.matmul(torch.matmul(std_op.view(1,1,V.shape[0],V.shape[1]), sigma21), std_op.view(1,1,V.shape[0],V.shape[1]).transpose(-1,-2))

                sigma21 = torch.diagonal(sigma21, dim1=-2, dim2=-1)

                # Extract the diagonal (variance) and compute the standard deviation
                y_21_std = torch.sqrt(torch.abs(sigma21))
        
        elif method == 'cholesky':
            # Compute Cholesky factorization
            L = torch.linalg.cholesky(k11 + eps)  # L (lower triangular) such that k11 + eps = LL^T
            y_21 = None
            if predict_mean:
                
                # Solve L * u = y_11 (forward substitution)
                u = torch.linalg.solve_triangular(L, y_11, upper=False)

                # Solve L^T * x = u (backward substitution)
                x = torch.linalg.solve_triangular(L.transpose(-1, -2), u, upper=True)

                # Compute the predicted mean
                y_21 = torch.matmul(k21, x)
                
                if mean_op is not None:
                    y_21 = torch.matmul(mean_op.view(1,1,V.shape[0],V.shape[1]), y_21)

            y_21_std = None
            if predict_std:
                # Solve L * u = k12 (forward substitution)
                u = torch.linalg.solve_triangular(L, k12, upper=False)

                # Solve L^T * x = u (backward substitution)
                x = torch.linalg.solve_triangular(L.transpose(-1, -2), u, upper=True)

                # Compute only the diagonal of sigma21 efficiently
                sigma21_diag = k22 - torch.matmul(k21, x)
                
                if std_op is not None:
                    sigma21_diag = torch.matmul(torch.matmul(std_op.view(1,1,V.shape[0],V.shape[1]), sigma21_diag), std_op.view(1,1,V.shape[0],V.shape[1]).transpose(-1,-2))

                sigma21_diag = torch.diagonal(sigma21_diag, dim1=-2, dim2=-1)

                # Compute the standard deviation
                y_21_std = torch.sqrt(torch.abs(sigma21_diag))

        elif method == 'lu':
            # Compute LU factorization
            LU, piv = torch.linalg.lu_factor(k11 + eps)  # LU decomposition
            
            y_21 = None
            if predict_mean:

                # Solve the system using LU (equivalent to (k11 + eps)^-1 * y_11)
                x = torch.linalg.lu_solve(LU, piv, y_11)

                # Compute the predicted mean
                y_21 = torch.matmul(k21, x)

                if mean_op is not None:
                    y_21 = torch.matmul(mean_op.view(1,1,V.shape[0],V.shape[1]), y_21)

            y_21_std = None
            if predict_std:
                # Solve LU system for k12
                x = torch.linalg.lu_solve(LU, piv, k12)

                # Compute only the diagonal of sigma21 efficiently
                sigma21_diag = k22 - torch.matmul(k21, x)

                if std_op is not None:
                    sigma21_diag = torch.matmul(torch.matmul(std_op.view(1,1,V.shape[0],V.shape[1]), sigma21_diag), std_op.view(1,1,V.shape[0],V.shape[1]).transpose(-1,-2))

                sigma21_diag = torch.diagonal(sigma21_diag, dim1=-2, dim2=-1)

                # Compute the standard deviation
                y_21_std = torch.sqrt(torch.abs(sigma21_diag))
                #y_21_std = torch.abs(sigma21_diag)

        elif method == 'cg':

            y_21 = None
            if predict_mean:
                # Solve the system using PCG (Preconditioned Conjugate Gradient)
                x = conjugate_gradient(k11 + eps, y_11, max_iter=100)  # max_iter can be tuned

                # Compute the predicted mean
                y_21 = torch.matmul(k21, x)
                
                if mean_op is not None:
                    y_21 = torch.matmul(mean_op.view(1,1,V.shape[0],V.shape[1]), y_21)

            y_21_std = None
            if predict_std:
                # Solve PCG system for k12
                x = conjugate_gradient(k11 + eps, k12, max_iter=100)

                # Compute only the diagonal of sigma21 efficiently
                sigma21_diag = k22 - torch.matmul(k21, x)
                
                if std_op is not None:
                    sigma21_diag = torch.matmul(torch.matmul(std_op.view(1,1,V.shape[0],V.shape[1]), sigma21_diag), std_op.view(1,1,V.shape[0],V.shape[1]).transpose(-1,-2))

                sigma21_diag = torch.diagonal(sigma21_diag, dim1=-2, dim2=-1)

                # Compute the standard deviation
                y_21_std = torch.sqrt(torch.abs(sigma21_diag))

        return y_21, y_21_std

def apply_operator(dr, field):
        """
        Apply a 2D/3D operator to a field
        """

        nelv = field.shape[0]
        lx = field.shape[3]  # This is not a mistake. This is how the data is read
        ly = field.shape[2]
        lz = field.shape[1]

        # ==================================================
        # Using loops
        # dudrst = np.zeros_like(field, dtype=field.dtype)
        # for e in range(0, nelv):
        #    tmp = field[e, :, :, :].reshape(-1, 1)
        #    dtmp = dr @ tmp
        #    dudrst[e, :, :, :] = dtmp.reshape((lz, ly, lx))
        # ==================================================

        # Using einsum
        field_shape = field.shape
        operator_shape = dr.shape
        field_shape_as_columns = (
            field_shape[0],
            field_shape[1] * field_shape[2] * field_shape[3],
            1,
        )

        # Reshape the field in palce
        field.shape = field_shape_as_columns

        # apply the 2D/3D operator broadcasting with einsum
        transformed_field = np.einsum(
            "ejk, ekm -> ejm",
            dr.reshape(1, operator_shape[0], operator_shape[1]),
            field,
        )

        # Reshape the field back to its original shape
        field.shape = field_shape
        transformed_field.shape = field_shape

        return transformed_field

def torch_apply_operator(dr, field):
    """
    Apply a 2D/3D operator to a field using PyTorch.
    
    Parameters:
      dr (torch.Tensor): The operator tensor with shape (N, N) where N = lz * ly * lx.
      field (torch.Tensor): The field tensor with shape (nelv, lz, ly, lx).
    
    Returns:
      torch.Tensor: The transformed field with the same shape as the input field.
    """
    # Save the original shape: (nelv, lz, ly, lx)
    original_shape = field.shape

    # Flatten the spatial dimensions: reshape to (nelv, lz*ly*lx, 1)
    field_flat = field.reshape(original_shape[0], -1, 1)

    # Prepare the operator for broadcasting by reshaping to (1, N, N)
    dr_reshaped = dr.reshape(1, dr.shape[0], dr.shape[1])

    # Apply the operator using einsum.
    # The einsum notation "ejk,ekm->ejm" indicates:
    # - 'e' indexes over the batch (nelv),
    # - 'j' indexes the output vector dimension,
    # - 'k' indexes the common dimension,
    # - 'm' is the singleton dimension.
    transformed_field = torch.einsum("ejk,ekm->ejm", dr_reshaped, field_flat)

    # Reshape the result back to the original field shape
    transformed_field = transformed_field.reshape(original_shape)
    
    return transformed_field.contiguous()

def add_settings_to_hdf5(h5group, settings_dict):
    """
    Recursively adds the key/value pairs from a settings dictionary to an HDF5 group.
    Dictionary values that are themselves dictionaries are added as subgroups;
    other values are stored as attributes.
    """
    for key, value in settings_dict.items():
        if isinstance(value, dict):
            subgroup = h5group.create_group(key)
            add_settings_to_hdf5(subgroup, value)
        else:
            h5group.attrs[key] = value

def load_hdf5_settings(group):
    """
    Recursively loads an HDF5 group into a dictionary.
    Attributes become key/value pairs and subgroups are loaded recursively.
    """
    settings = {}
    # Load attributes
    for key, value in group.attrs.items():
        settings[key] = value
    # Recursively load subgroups
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            settings[key] = load_hdf5_settings(item)
    return settings

def conjugate_gradient(A, b, tol=1e-5, max_iter=1000):
    """
    Solves Ax = b using the Preconditioned Conjugate Gradient (PCG) method.
    A must be symmetric and positive definite (SPD).
    """
    x = torch.zeros_like(b)  # Initial guess (zero vector)
    r = b - torch.matmul(A, x)  # Initial residual
    z = r  # No preconditioner (Identity matrix)
    p = z.clone()  # Initial search direction
    rs_old = torch.sum(r * z, dim=-2, keepdim=True)  # Inner product (batch-wise)

    for i in range(max_iter):
        Ap = torch.matmul(A, p)
        alpha = rs_old / torch.sum(p * Ap, dim=-2, keepdim=True)  # Step size
        x = x + alpha * p  # Update solution
        r = r - alpha * Ap  # Update residual

        if torch.norm(r) < tol:  # Convergence check
            break

        z = r  # No preconditioning
        rs_new = torch.sum(r * z, dim=-2, keepdim=True)  # New inner product
        beta = rs_new / rs_old  # Compute beta
        p = z + beta * p  # Update search direction
        rs_old = rs_new  # Store for next iteration

    return x  # Approximate solution x
