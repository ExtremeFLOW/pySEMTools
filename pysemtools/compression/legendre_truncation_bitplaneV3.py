""" Module that contains the class and methods to perform direct sampling on a field """

import random
from mpi4py import MPI
from ..monitoring.logger import Logger
from ..datatypes.msh import Mesh
from ..datatypes.coef import Coef
from ..datatypes.coef import get_transform_matrix
import numpy as np
import bz2
import sys
import h5py
import os
import torch
import math

class DiscreetLegendreTruncationBPAdaptiveV3:

    """ 
    Class to perform direct sampling on a field in the SEM format
    """

    def __init__(self, comm: MPI.Comm = None, dtype: np.dtype = np.double,  msh: Mesh = None, filename: str = None, max_elements_to_process: int = 256, bckend: str = "numpy", coef: Coef = None):
        
        self.log = Logger(comm=comm, module_name="DirectSampler")
        
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

            # If the data was initialized from file, put it in a torch tensor
            if hasattr(self, "uncompressed_data"):
                for field in self.uncompressed_data.keys():
                    for data in self.uncompressed_data[field].keys():
                        self.uncompressed_data[field][data] = torch.tensor(self.uncompressed_data[field][data], dtype=self.dtype_d, device = self.device, requires_grad=False)

        # Jacobian for coefficient-space weighted error accumulation.
        self.coef = coef
        self.jac = self._build_jac(msh=msh, coef=coef)
        self.B = self._build_B(msh=msh, coef=coef)

    def _build_jac(self, msh: Mesh = None, coef: Coef = None):
        """
        Use coef.jac if available, otherwise use unit weights.
        """

        if coef is not None and hasattr(coef, "jac"):
            jac = coef.jac
            if hasattr(jac, "detach"):
                jac = jac.detach().cpu().numpy()
            return np.asarray(jac)

        if msh is not None:
            return np.ones_like(msh.x, dtype=self.dtype)

        return np.ones((self.nelv, self.lz, self.ly, self.lx), dtype=self.dtype)

    def _build_B(self, msh: Mesh = None, coef: Coef = None):
        """
        Use coef.B if available, otherwise use unit weights.
        """

        if coef is not None and hasattr(coef, "B"):
            B = coef.B
            if hasattr(B, "detach"):
                B = B.detach().cpu().numpy()
            return np.asarray(B)

        if msh is not None:
            return np.ones_like(msh.x, dtype=self.dtype)

        return np.ones((self.nelv, self.lz, self.ly, self.lx), dtype=self.dtype)

    def init_from_file(self, comm: MPI.Comm, filename: str, max_elements_to_process: int = 256):
        """
        """

        self.log.write("info", f"Initializing the DirectSampler from file: {filename}")

        self.settings, self.compressed_data = self.read_compressed_samples(comm = comm, filename=filename)

        self.init_common(max_elements_to_process)

        self.uncompressed_data = self.decompress_samples(self.settings, self.compressed_data)


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


    def clear(self):

        # Clear the data that has been sampled. This is necesary to avoid mixing things up when sampling new fields.
        self.settings = {}
        self.uncompressed_data = {}
        self.compressed_data = {}

    @staticmethod
    def _as_numpy(array):
        """Return a NumPy view/copy of a NumPy or torch-backed coefficient array."""
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()
        return np.asarray(array)

    def _build_error_gram(self, element, target_quantity, zigzag_order):
        """
        Build the element-local Gram matrix for field or gradient error.

        Let ``L`` map Legendre coefficients to nodal values.  For the physical
        x derivative in three dimensions,

            Ax = diag(drdx) Dr + diag(dsdx) Ds + diag(dtdx) Dt,
            Kx = Ax L,
            Gx = Kx.T W Kx.

        ``W`` contains the reference quadrature weights multiplied pointwise by
        the element Jacobian determinant.  The y and z operators are analogous.
        For ``target_quantity='gradient'``, the returned matrix is

            Ggradient = Gx + Gy (+ Gz in 3-D),

        corresponding to the weighted squared L2 error of the full gradient.
        For ``target_quantity='field'``, the returned matrix is

            Gfield = L.T W L,

        corresponding to the weighted squared L2 error of the nodal field.

        The final row/column permutation places G in the same spectral-zigzag
        coefficient order used by the adaptive bitplane allocator.
        """
        if self.coef is None:
            raise ValueError(
                "V3 scoring requires a Coef object with stored multidimensional "
                "operators"
            )

        required = ["v_xd", "w_xd", "jac"]
        if target_quantity == "gradient":
            required.extend(["dr_xd", "ds_xd"])
            if self.gdim == 3:
                required.append("dt_xd")
        missing = [name for name in required if not hasattr(self.coef, name)]
        if missing:
            raise ValueError(
                "Missing Coef data required by V3: " + ", ".join(missing)
                + ". Initialize Coef with store_multidimensional_operators=True."
            )

        L = self._as_numpy(self.coef.v_xd)

        # coef.w_xd is stored as a diagonal matrix.  Only its diagonal is needed,
        # because W K is a pointwise multiplication of every row of K.
        reference_weight_matrix = self._as_numpy(self.coef.w_xd)
        if reference_weight_matrix.ndim == 2:
            reference_weights = np.diag(reference_weight_matrix)
        else:
            reference_weights = reference_weight_matrix.reshape(-1)

        jacobian = self._as_numpy(self.coef.jac)[element].reshape(-1)
        physical_weights = reference_weights * jacobian

        # Field error requires no derivative or inverse-Jacobian operators.
        if target_quantity == "field":
            gram = L.T @ (physical_weights[:, None] * L)
            gram = 0.5 * (gram + gram.T)
            return gram[np.ix_(zigzag_order, zigzag_order)], float(
                np.sum(physical_weights)
            )

        Dr = self._as_numpy(self.coef.dr_xd)
        Ds = self._as_numpy(self.coef.ds_xd)
        Dt = self._as_numpy(self.coef.dt_xd) if self.gdim == 3 else None

        def geometry(name):
            if not hasattr(self.coef, name):
                raise ValueError(
                    f"Coef does not contain inverse-Jacobian factor '{name}'"
                )
            return self._as_numpy(getattr(self.coef, name))[element].reshape(-1)

        def physical_derivative_operator(direction):
            if direction == "x":
                operator = geometry("drdx")[:, None] * Dr
                operator += geometry("dsdx")[:, None] * Ds
                if self.gdim == 3:
                    operator += geometry("dtdx")[:, None] * Dt
                return operator

            if direction == "y":
                operator = geometry("drdy")[:, None] * Dr
                operator += geometry("dsdy")[:, None] * Ds
                if self.gdim == 3:
                    operator += geometry("dtdy")[:, None] * Dt
                return operator

            if direction == "z" and self.gdim == 3:
                operator = geometry("drdz")[:, None] * Dr
                operator += geometry("dsdz")[:, None] * Ds
                operator += geometry("dtdz")[:, None] * Dt
                return operator

            raise ValueError(f"Derivative direction '{direction}' is unavailable")

        directions = ["x", "y"] if self.gdim == 2 else ["x", "y", "z"]

        gram = np.zeros((L.shape[1], L.shape[1]), dtype=self.dtype)
        for direction in directions:
            A = physical_derivative_operator(direction)
            K = A @ L
            # K.T @ diag(physical_weights) @ K, without forming the diagonal W.
            gram += K.T @ (physical_weights[:, None] * K)

        # Numerical roundoff can introduce a tiny asymmetry.  Enforce the exact
        # symmetry expected by the quadratic error form.
        gram = 0.5 * (gram + gram.T)
        return gram[np.ix_(zigzag_order, zigzag_order)], float(np.sum(physical_weights))
    
    def sample_field(
        self,
        field: np.ndarray = None,
        field_name: str = "field",
        target_error: float = None,
        target_quantity: str = "gradient",
    ):
        
        self.log.write("info", f"Sampling field \"{field_name}\" with target_error={target_error}")

        # Create a dictionary to store the data that will be compressed
        self.uncompressed_data[f"{field_name}"] = {}

        # Copy the field into device if needed
        if self.bckend == "torch":
            field = torch.tensor(field, dtype=self.dtype_d, device = self.device, requires_grad=False)

        self.log.write("info", "Transforming the field into to legendre space")
        field_hat = self.transform_field(field, to="legendre")
            
        if target_error is None:
            raise ValueError("target_error must be provided")
        if target_error < 0:
            raise ValueError("target_error must be non-negative")
        if self.bckend != "numpy":
            raise NotImplementedError("bitplane sampling is currently implemented only for the numpy backend")
        if target_quantity not in {"field", "gradient"}:
            raise ValueError(
                "target_quantity must be either 'field' or 'gradient'"
            )

        self.settings["compression"] = {
            "method": "fixed_error_bitplane_v3",
            "target_error": target_error,
            "target_quantity": target_quantity,
        }

        self.log.write("info", f"Sampling the field using bitplane coding. using settings: {self.settings['compression']}")
        bitplane_data, sampling_stats = self._sample_fixed_error(field_hat, field_name, self.settings)
        self.uncompressed_data[f"{field_name}"].update(bitplane_data)
        self.settings["compression"].update(sampling_stats)
        self.log.write("info", f"Bitplane stream saved for field \"{field_name}\"")
        
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
            base_name, _ = os.path.splitext(filename)
            folder_name = f"{base_name}_comp"
            if rank == 0:
                os.makedirs(folder_name, exist_ok=True)
            # Ensure all ranks wait until the folder has been created.
            comm.Barrier()
            file_path = os.path.join(folder_name, f"{base_name}_rank_{rank}.h5")
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
            base_name, _ = os.path.splitext(filename)
            folder_name = f"{base_name}_comp"
            file_path = os.path.join(folder_name, f"{base_name}_rank_{rank}.h5")
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

                if data_key == "field":
                    shape = (nelv, lz, ly, lx)
                    if dtype == "single":
                        array_dtype = np.float32
                    elif dtype == "double":
                        array_dtype = np.float64
                elif data_key == "bitplane_symbols":
                    shape = (-1,)
                    array_dtype = np.uint32
                elif data_key == "bitplane_symbol_counts":
                    shape = (nelv,)
                    array_dtype = np.uint32
                elif data_key == "bitplane_exponents":
                    shape = (nelv,)
                    array_dtype = np.int16
                elif data_key == "bitplane_precision":
                    shape = (nelv, lz * ly * lx)
                    array_dtype = np.uint8
                else:
                    raise ValueError("Invalid data key")

                temp = np.frombuffer(bz2.decompress(compressed_bytes), dtype=array_dtype)

                uncompressed_data[field][data_key] = temp.reshape(shape)

        return uncompressed_data
 
    def _sample_fixed_error(self, field_hat: np.ndarray, field_name: str, settings: dict):
        """
        Allocate a separate contiguous bit prefix to every coefficient using
        either physical-field or physical-gradient error as the distortion.

        As in V2, a useful extension ends at the coefficient's next 1 bit; any
        intervening zero planes are included in its rate cost.  Bits are never
        skipped inside a coefficient.

        Candidate benefits are coupled.  If ``G`` is the selected field/gradient
        Gram matrix, ``e`` is the current coefficient-error vector, and refining
        coefficient c changes its reconstructed value by signed amount delta,
        then the exact reduction in the selected SSE is

            benefit_c = 2 * delta * (G @ e)[c] - delta**2 * G[c, c].

        After selecting c, this implementation updates

            e[c]  <- e[c] - delta,
            G @ e <- G @ e - delta * G[:, c].

        Because the off-diagonal entries of G change every coefficient's score,
        V2's persistent heap cannot be used: all currently available candidates
        are rescored after every accepted extension.

        ``bitplane_precision[e, c]`` records the number of MSB planes assigned
        to coefficient c of element e.  During plane-major stream construction,
        a coefficient emits an event only while ``plane < precision[e, c]``.
        The precision map is itself sent to bzip2, which can exploit patterns
        shared by coefficients and by the full domain.
        """

        target_error = settings["compression"]["target_error"]
        target_quantity = settings["compression"]["target_quantity"]
        nelv = settings["mesh_information"]["nelv"]

        # Flatten within each element and reorder the coefficients using the
        # same low-to-high spectral zigzag as V1.  The precision map and the
        # plane-major event stream therefore both follow this ordering.
        zigzag_order = self._spectral_zigzag_order(self.lz, self.ly, self.lx)
        y = field_hat.reshape(nelv, -1)[:, zigzag_order]
        n_coeff = y.shape[1]

        # The integer magnitude itself remains uint64.  This is different from
        # the RLE symbol type: a magnitude needs enough bits to expose many
        # precision planes, whereas a symbol only needs to hold one run length.
        # Float32 has fewer useful mantissa bits, so 31 planes are ample there.
        max_planes = 31 if self.dtype == np.float32 else 63

        symbols_per_element = []
        symbol_counts = np.zeros(nelv, dtype=np.uint32)
        exponents = np.zeros(nelv, dtype=np.int16)
        precision = np.zeros((nelv, n_coeff), dtype=np.uint8)
        achieved_error = np.zeros(nelv, dtype=self.dtype)

        for element in range(nelv):
            values = y[element]
            abs_values = np.abs(values)
            maximum = float(np.max(abs_values))

            # G is expressed in the same zigzag ordering as values.  With an
            # initially zero reconstruction, coefficient_error equals values.
            error_gram, element_volume = self._build_error_gram(
                element,
                target_quantity,
                zigzag_order,
            )
            element_volume = element_volume if element_volume > 0.0 else 1.0
            coefficient_error = values.astype(np.float64, copy=True)
            gram_times_error = error_gram @ coefficient_error
            initial_sse = float(coefficient_error @ gram_times_error)
            initial_sse = max(initial_sse, 0.0)
            initial_error = np.sqrt(initial_sse / element_volume)

            # Before transmitting anything the decoder reconstructs all
            # coefficients as zero.  If that already meets the requested
            # target RMS, this element requires no stream data.
            if maximum == 0.0:
                achieved_error[element] = initial_error
                symbols_per_element.append(np.empty(0, dtype=np.uint32))
                continue

            # Build a block-floating-point representation for this element.
            #
            # exponent locates the largest coefficient in binary.  All
            # coefficients then share quantum, the value represented by the
            # least-significant available integer bit.  This is the analogue of
            # TTHRESH's common scaling in Eq. (9).
            exponent = int(np.floor(np.log2(maximum)))
            quantum = np.ldexp(1.0, exponent - (max_planes - 1))

            # floor is intentional.  Since maximum < 2**(exponent + 1), the
            # scaled value is strictly below 2**max_planes and therefore fits
            # in the selected uint64 bit positions.  Rounding could push an
            # extreme value up to 2**max_planes and require one extra bit.
            magnitude = np.floor(abs_values / quantum).astype(np.uint64)
            exponents[element] = exponent

            reconstructed_magnitude = np.zeros(n_coeff, dtype=np.uint64)

            total_sse = initial_sse
            target_sse = float(target_error * target_error * element_volume)

            def next_extension(coefficient, old_precision):
                """
                Return the next contiguous prefix ending at a 1 bit.

                The returned signed_delta is the change in the reconstructed
                Legendre coefficient.  It is positive for a positive original
                coefficient and negative for a negative original coefficient.
                """
                for new_precision in range(old_precision + 1, max_planes + 1):
                    bit_position = max_planes - new_precision
                    if (int(magnitude[coefficient]) >> bit_position) & 1:
                        old_mag = int(reconstructed_magnitude[coefficient])
                        new_mag = old_mag | (1 << bit_position)
                        magnitude_delta = (new_mag - old_mag) * quantum
                        signed_delta = (
                            -magnitude_delta
                            if values[coefficient] < 0.0
                            else magnitude_delta
                        )
                        # Every traversed plane emits one bit.  The first 1 also
                        # emits the sign, hence one additional event.
                        cost = new_precision - old_precision
                        if old_mag == 0:
                            cost += 1
                        return new_precision, new_mag, signed_delta, cost
                return None

            if target_quantity == "gradient":
                # Preserve the constant Legendre mode explicitly for a gradient
                # target.  Its gradient is zero, so the objective cannot detect
                # an error in it and would otherwise allow the mean to drift.
                # The zigzag puts this mode at coefficient index zero.
                mean_coefficient = 0
                mean_precision = 0

                while True:
                    candidate = next_extension(mean_coefficient, mean_precision)
                    if candidate is None:
                        break

                    new_precision, new_mag, signed_delta, _ = candidate
                    precision[element, mean_coefficient] = new_precision
                    reconstructed_magnitude[mean_coefficient] = np.uint64(new_mag)

                    coefficient_error[mean_coefficient] -= signed_delta
                    gram_times_error -= signed_delta * error_gram[:, mean_coefficient]
                    mean_precision = new_precision

                # This should leave the gradient SSE unchanged mathematically,
                # but recomputing avoids relying on exact discrete cancellation.
                total_sse = float(coefficient_error @ gram_times_error)

            # Cache only the geometry-independent description of each next
            # extension.  Its benefit is deliberately recomputed every greedy
            # iteration because gram_times_error changes globally.
            candidates = [
                next_extension(c, int(precision[element, c]))
                for c in range(n_coeff)
            ]
            gram_diagonal = np.diag(error_gram)

            while total_sse > target_sse:
                best_score = -np.inf
                best_coefficient = None
                best_benefit = None

                for coefficient, candidate in enumerate(candidates):
                    if candidate is None:
                        continue

                    _, _, signed_delta, cost = candidate
                    benefit = (
                        2.0 * signed_delta * gram_times_error[coefficient]
                        - signed_delta * signed_delta * gram_diagonal[coefficient]
                    )
                    score = benefit / cost

                    # Coefficients are already in spectral-zigzag order, so the
                    # strict comparison also makes that order the deterministic
                    # tie breaker.
                    if score > best_score:
                        best_score = score
                        best_coefficient = coefficient
                        best_benefit = benefit

                if best_coefficient is None:
                    break

                new_precision, new_mag, signed_delta, _ = candidates[best_coefficient]
                precision[element, best_coefficient] = new_precision
                reconstructed_magnitude[best_coefficient] = np.uint64(new_mag)

                # Exact O(n_coeff) update of the coupled error state.
                coefficient_error[best_coefficient] -= signed_delta
                gram_times_error -= signed_delta * error_gram[:, best_coefficient]
                total_sse -= best_benefit

                candidates[best_coefficient] = next_extension(
                    best_coefficient,
                    new_precision,
                )

            # Re-evaluate the quadratic form once to remove accumulated scalar
            # update roundoff before recording the achieved target RMS.
            total_sse = float(coefficient_error @ (error_gram @ coefficient_error))
            achieved_error[element] = np.sqrt(max(total_sse, 0.0) / element_volume)

            # Construct a plane-major stream, but omit coefficients whose chosen
            # prefix has already ended.  Sign follows the first emitted 1.
            events = []
            significant = np.zeros(n_coeff, dtype=bool)
            for plane in range(int(np.max(precision[element]))):
                bit_position = max_planes - 1 - plane
                for coefficient in range(n_coeff):
                    if plane >= int(precision[element, coefficient]):
                        continue
                    bit = (int(magnitude[coefficient]) >> bit_position) & 1
                    events.append(bit)
                    if not significant[coefficient] and bit:
                        significant[coefficient] = True
                        events.append(int(values[coefficient] < 0.0))

            encoded_symbols = self._run_length_encode_bits(events)
            symbol_counts[element] = encoded_symbols.size
            symbols_per_element.append(encoded_symbols)

        if symbols_per_element:
            symbols = np.concatenate(symbols_per_element).astype(np.uint32, copy=False)
        else:
            symbols = np.empty(0, dtype=np.uint32)

        stats = {
            "bitplane_format": "adaptive_gram_precision_rle_uint32",
            "target_quantity": target_quantity,
            "avg_bitplanes": float(np.mean(precision)),
            "avg_achieved_error": float(np.mean(achieved_error)),
            "target_reached": bool(np.all(achieved_error <= target_error)),
        }

        bitplane_data = {
            "bitplane_symbols": symbols,
            "bitplane_symbol_counts": symbol_counts,
            "bitplane_exponents": exponents,
            "bitplane_precision": precision,
        }
        return bitplane_data, stats

    @staticmethod
    def _spectral_zigzag_order(lz, ly, lx):
        """
        Return C-order flat indices for a reversible 3-D spectral zigzag.

        Coefficients are grouped by total degree ``ix + iy + iz``.  Alternate
        degree shells are reversed to obtain a serpentine traversal.  The
        returned indices address arrays shaped ``(lz, ly, lx)``.
        """
        shells = [[] for _ in range((lx - 1) + (ly - 1) + (lz - 1) + 1)]
        for iz in range(lz):
            for iy in range(ly):
                for ix in range(lx):
                    shells[ix + iy + iz].append((iz, iy, ix))

        order = []
        for degree, shell in enumerate(shells):
            if degree % 2 == 1:
                shell.reverse()
            order.extend(
                np.ravel_multi_index(coord, (lz, ly, lx))
                for coord in shell
            )

        return np.asarray(order, dtype=np.intp)

    @staticmethod
    def _run_length_encode_bits(bits):
        """
        Convert a sequence of 0/1 events into uint32 run-length symbols.

        For example, the event sequence

            0, 0, 0, 1, 1, 0

        contains the runs ``(3, 0), (2, 1), (1, 0)`` and becomes

            (3 << 1) | 0, (2 << 1) | 1, (1 << 1) | 0
            = 6, 5, 2.

        Difference from the TTHRESH insignificance runs
        ------------------------------------------------
        TTHRESH Section 4.2 encodes each bitplane independently.  It counts the
        number k of zeroes before the next one and stores only k; the following
        one is implicit.  A final run may instead end at the bitplane boundary.
        The paper's example is:

            bits:       0 1 1 1 0 0 0 1
            zero runs:  1, 0, 0, 3

        Reading the first three symbols means "one zero then one, zero zeroes
        then one, zero zeroes then one".  The final 3 means "three zeroes then
        the last one".  Thus TTHRESH stores zero-run lengths, not ordinary
        ``(length, value)`` pairs.

        This function uses conventional binary RLE instead.  It turns the same
        bits into:

            runs:       (1, 0), (3, 1), (3, 0), (1, 1)
            symbols:    2, 7, 6, 3

        Here both the run length and its binary value are explicit.  Moreover,
        the caller supplies one stream containing magnitude, sign, and refinement
        events from all retained planes, so runs can cross plane boundaries.
        This is easier to decode but generally creates a different symbol
        distribution from TTHRESH.  bzip2 later compresses the uint32 symbols.

        One bit of a uint32 is reserved for the event value, leaving 31 bits for
        the run.  If a future element can exceed that limit, a long run can be
        split into multiple symbols; for now an explicit error prevents silent
        integer overflow.
        """
        if len(bits) == 0:
            return np.empty(0, dtype=np.uint32)

        symbols = []
        current = int(bits[0])
        run_length = 1
        max_run_length = np.iinfo(np.uint32).max >> 1

        for bit in bits[1:]:
            bit = int(bit)
            if bit == current:
                run_length += 1
            else:
                if run_length > max_run_length:
                    raise OverflowError(
                        "A bitplane run does not fit in a uint32 symbol; "
                        "split long runs before packing"
                    )
                symbols.append((run_length << 1) | current)
                current = bit
                run_length = 1

        if run_length > max_run_length:
            raise OverflowError(
                "A bitplane run does not fit in a uint32 symbol; "
                "split long runs before packing"
            )
        symbols.append((run_length << 1) | current)
        return np.asarray(symbols, dtype=np.uint32)

    @staticmethod
    def _run_length_decode_bits(symbols):
        """
        Yield the original binary events from packed RLE symbols.

        ``packed & 1`` extracts the event value from the least-significant bit.
        ``packed >> 1`` removes that bit and recovers the run length.  Calling
        this function as a generator avoids allocating the expanded event stream
        during decompression.
        """
        for symbol in symbols:
            packed = int(symbol)
            run_length = packed >> 1
            bit = packed & 1
            for _ in range(run_length):
                yield bit

    def _decode_fixed_error(self, data):
        """
        Decode the adaptive per-coefficient precision stream.

        The precision map tells the decoder whether each coefficient appears on
        a plane.  Significance, sign, and magnitude are otherwise reconstructed
        from the same event grammar used by the encoder.
        """
        n_coeff = self.lx * self.ly * self.lz
        max_planes = 31 if self.dtype == np.float32 else 63
        # Decode in transmitted zigzag order, then invert that permutation.
        output_zigzag = np.zeros((self.nelv, n_coeff), dtype=self.dtype)

        symbols = data["bitplane_symbols"]
        counts = data["bitplane_symbol_counts"]
        exponents = data["bitplane_exponents"]
        precision = data["bitplane_precision"]
        symbol_offset = 0

        for element in range(self.nelv):
            # Streams from all elements live in one concatenated symbol array.
            # count identifies this element's slice and advances the offset to
            # the beginning of the next one.
            count = int(counts[element])
            element_symbols = symbols[symbol_offset:symbol_offset + count]
            symbol_offset += count
            element_precision = precision[element]
            number_of_planes = int(np.max(element_precision))

            # An element that met the target as the all-zero approximation wrote
            # no events.  output was initialized to zero, so nothing is needed.
            if number_of_planes == 0:
                continue

            # Expand RLE on demand.  The significance and sign arrays are decoder
            # state reconstructed solely from the events seen so far.
            bits = self._run_length_decode_bits(element_symbols)
            significant = np.zeros(n_coeff, dtype=bool)
            negative = np.zeros(n_coeff, dtype=bool)
            magnitude = np.zeros(n_coeff, dtype=np.uint64)

            for plane in range(number_of_planes):
                bit_position = max_planes - 1 - plane
                for coefficient in range(n_coeff):
                    if plane >= int(element_precision[coefficient]):
                        continue
                    # This is either a significance event or a refinement event,
                    # depending on whether this coefficient was significant at
                    # the start of the current step.
                    bit = next(bits)
                    if significant[coefficient]:
                        if bit:
                            magnitude[coefficient] |= np.uint64(1) << np.uint64(bit_position)
                    elif bit:
                        # A newly significant coefficient has one immediately
                        # following sign event: 0=positive, 1=negative.
                        significant[coefficient] = True
                        magnitude[coefficient] |= np.uint64(1) << np.uint64(bit_position)
                        negative[coefficient] = bool(next(bits))

            # Recover the same quantum used by the encoder and map integer
            # magnitudes back to floating point before restoring the signs.
            quantum = np.ldexp(1.0, int(exponents[element]) - (max_planes - 1))
            decoded = magnitude.astype(np.float64) * quantum
            decoded[negative] *= -1.0
            output_zigzag[element] = decoded.astype(self.dtype)

        # This catches corrupt counts and format mismatches that leave complete
        # RLE symbols outside every element slice.
        if symbol_offset != symbols.size:
            raise ValueError("Bitplane stream has unused symbols; metadata are inconsistent")
        zigzag_order = self._spectral_zigzag_order(self.lz, self.ly, self.lx)
        output = np.zeros_like(output_zigzag)
        output[:, zigzag_order] = output_zigzag
        return output.reshape(self.nelv, self.lz, self.ly, self.lx)

    def reconstruct_field(self, field_name: str = None):
        data = self.uncompressed_data[field_name]
        if self.settings["compression"]["method"] == "fixed_error_bitplane_v3":
            field_hat = self._decode_fixed_error(data)
        else:
            raise ValueError("Unsupported compression method")
        return self.transform_field(field = field_hat, to="physical")
 
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
    
    return transformed_field

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