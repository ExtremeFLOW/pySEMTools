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

class DiscreetLegendreTruncationBPWeightedV4:

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
        self._gram_diagonal_cache = {}
        self._element_volume_cache = {}
        if coef is not None:
            # Gradient compression is the primary V4 use case.  Pay the
            # geometry-dependent cost once at construction, then reuse Gcc for
            # every field and snapshot compressed with this object.
            self.precompute_gram_diagonal("gradient")

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

    def _prepare_diagonal_error_operators(self):
        """Cache geometry-independent operators used by diagonal V4.

        ``L`` maps modal coefficients to nodal values.  For a gradient target
        the matrices ``Dr @ L``, ``Ds @ L`` and ``Dt @ L`` are also independent
        of the element geometry, so computing them once avoids an expensive
        dense matrix product for every element and physical direction.
        """
        if hasattr(self, "_v3_diagonal_operators"):
            return self._v3_diagonal_operators

        if self.coef is None:
            raise ValueError("V4 weighting requires a Coef object")

        required = ["v_xd", "w_xd", "jac"]
        missing = [name for name in required if not hasattr(self.coef, name)]
        if missing:
            raise ValueError("Missing Coef data required by V4: " + ", ".join(missing))

        L = self._as_numpy(self.coef.v_xd)
        weight_data = self._as_numpy(self.coef.w_xd)
        reference_weights = (
            np.diag(weight_data) if weight_data.ndim == 2
            else weight_data.reshape(-1)
        )
        operators = {"L": L, "reference_weights": reference_weights}

        if hasattr(self.coef, "dr_xd") and hasattr(self.coef, "ds_xd"):
            operators["DrL"] = self._as_numpy(self.coef.dr_xd) @ L
            operators["DsL"] = self._as_numpy(self.coef.ds_xd) @ L
            if self.gdim == 3:
                if not hasattr(self.coef, "dt_xd"):
                    raise ValueError("Missing Coef data required by V4: dt_xd")
                operators["DtL"] = self._as_numpy(self.coef.dt_xd) @ L

        self._v3_diagonal_operators = operators
        return operators

    def _build_error_diagonal(self, element, target_quantity):
        """Build one element's ``diag(G)`` in native coefficient order.

        No full Gram matrix is formed.  The diagonal is accumulated as
        ``sum_q weight[q] * K[q, c]**2``.  The returned evaluator computes the
        exact physical quadratic error from ``K @ coefficient_error`` and is
        used only for final validation.

        Let ``L`` map Legendre coefficients to nodal values.  For the physical
        x derivative in three dimensions,

            Ax = diag(drdx) Dr + diag(dsdx) Ds + diag(dtdx) Dt,
            Kx = Ax L,
            Gx = Kx.T W Kx.

        ``W`` contains the reference quadrature weights multiplied pointwise by
        the element Jacobian determinant.  The y and z operators are analogous.
        For ``target_quantity='gradient'``, the returned diagonal is

            Ggradient = Gx + Gy (+ Gz in 3-D),

        corresponding to the weighted squared L2 error of the full gradient.
        For ``target_quantity='field'``, the returned matrix is

            Gfield = L.T W L,

        corresponding to the weighted squared L2 error of the nodal field.

        This routine is mesh preprocessing.  It never runs inside the per-field
        allocation loop after the diagonal has been cached.
        """
        required = ["v_xd", "w_xd", "jac"]
        if target_quantity == "gradient":
            required.extend(["dr_xd", "ds_xd"])
            if self.gdim == 3:
                required.append("dt_xd")
        missing = [name for name in required if not hasattr(self.coef, name)]
        if missing:
            raise ValueError(
                "Missing Coef data required by V4: " + ", ".join(missing)
                + ". Initialize Coef with store_multidimensional_operators=True."
            )

        cached = self._prepare_diagonal_error_operators()
        L = cached["L"]
        reference_weights = cached["reference_weights"]

        jacobian = self._as_numpy(self.coef.jac)[element].reshape(-1)
        physical_weights = reference_weights * jacobian

        if target_quantity == "field":
            diagonal = np.einsum("q,qc,qc->c", physical_weights, L, L, optimize=True)
            return diagonal, float(np.sum(physical_weights))

        def geometry(name):
            if not hasattr(self.coef, name):
                raise ValueError(
                    f"Coef does not contain inverse-Jacobian factor '{name}'"
                )
            return self._as_numpy(getattr(self.coef, name))[element].reshape(-1)

        DrL = cached["DrL"]
        DsL = cached["DsL"]
        DtL = cached.get("DtL")

        def physical_modal_operator(direction):
            if direction == "x":
                operator = geometry("drdx")[:, None] * DrL
                operator += geometry("dsdx")[:, None] * DsL
                if self.gdim == 3:
                    operator += geometry("dtdx")[:, None] * DtL
                return operator

            if direction == "y":
                operator = geometry("drdy")[:, None] * DrL
                operator += geometry("dsdy")[:, None] * DsL
                if self.gdim == 3:
                    operator += geometry("dtdy")[:, None] * DtL
                return operator

            if direction == "z" and self.gdim == 3:
                operator = geometry("drdz")[:, None] * DrL
                operator += geometry("dsdz")[:, None] * DsL
                operator += geometry("dtdz")[:, None] * DtL
                return operator

            raise ValueError(f"Derivative direction '{direction}' is unavailable")

        directions = ["x", "y"] if self.gdim == 2 else ["x", "y", "z"]

        diagonal = np.zeros(L.shape[1], dtype=np.float64)
        for direction in directions:
            K = physical_modal_operator(direction)
            diagonal += np.einsum("q,qc,qc->c", physical_weights, K, K, optimize=True)
            del K

        return diagonal, float(np.sum(physical_weights))

    def precompute_gram_diagonal(self, target_quantity="gradient"):
        """Compute and cache element-local Gcc once for the current mesh.

        The stored array has the same element-local shape as ``B``:
        ``(nelv, lz, ly, lx)``.  It may therefore be supplied, retained and
        reused exactly like any other mesh-dependent scalar field.
        """
        if target_quantity not in {"field", "gradient"}:
            raise ValueError("target_quantity must be either 'field' or 'gradient'")
        if target_quantity in self._gram_diagonal_cache:
            return self._gram_diagonal_cache[target_quantity]

        n_coeff = self.lz * self.ly * self.lx
        diagonal = np.empty((self.nelv, n_coeff), dtype=np.float64)
        volumes = np.empty(self.nelv, dtype=np.float64)
        for element in range(self.nelv):
            diagonal[element], volumes[element] = self._build_error_diagonal(
                element, target_quantity
            )

        diagonal = diagonal.reshape(self.nelv, self.lz, self.ly, self.lx)
        self._gram_diagonal_cache[target_quantity] = diagonal
        self._element_volume_cache[target_quantity] = volumes
        # Public aliases make the common gradient diagonal easy to inspect or
        # pass alongside Coef.B and Coef.jac.
        if target_quantity == "gradient":
            self.Gcc = diagonal
            self.Gcc_volume = volumes
        return diagonal
    
    def sample_field(
        self,
        field: np.ndarray = None,
        field_name: str = "field",
        target_error: float = None,
        target_quantity: str = "gradient",
        allocation_mode: str = "weighted_planes",
    ):
        """Transform and encode one field to a requested physical RMS.

        Parameters
        ----------
        field:
            SEM nodal values with one ``(lz, ly, lx)`` block per element.
        field_name:
            Name used as the key in the stored stream dictionaries.
        target_error:
            Requested element-local RMS tolerance.
        target_quantity:
            ``"field"`` targets physical field RMS; ``"gradient"`` targets
            the RMS norm of the complete 2-D or 3-D physical gradient.
        allocation_mode:
            Kept as metadata. V4 supports only ``"weighted_planes"``.
        """
        
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
        if allocation_mode != "weighted_planes":
            raise ValueError("V4 supports only allocation_mode='weighted_planes'")

        self.settings["compression"] = {
            "method": "fixed_error_bitplane_v4",
            "target_error": target_error,
            "target_quantity": target_quantity,
            "allocation_mode": allocation_mode,
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
                # These streams are already bit-packed. Their values are
                # assumed to be close to random, so keep them raw. Structured
                # significance symbols, counts, exponents and precision maps
                # still benefit from bzip2.
                store_raw = data in {
                    "bitplane_sign_bytes", "bitplane_refinement_bytes"
                }
                if self.bckend == "numpy":
                    raw = self.uncompressed_data[field][data].tobytes()
                elif self.bckend == "torch":
                    raw = self.uncompressed_data[field][data].cpu().numpy().tobytes()
                self.compressed_data[field][data] = raw if store_raw else bz2.compress(raw)


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
            base_name, _ = os.path.splitext(os.path.basename(filename))
            folder_name = os.path.join(path_name, f"{base_name}_comp")
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
            path_name = os.path.dirname(filename)
            base_name, _ = os.path.splitext(os.path.basename(filename))
            folder_name = os.path.join(path_name, f"{base_name}_comp")
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
                elif data_key == "bitplane_sign_bytes":
                    shape = (-1,)
                    array_dtype = np.uint8
                elif data_key == "bitplane_sign_counts":
                    shape = (nelv,)
                    array_dtype = np.uint32
                elif data_key == "bitplane_refinement_bytes":
                    shape = (-1,)
                    array_dtype = np.uint8
                elif data_key == "bitplane_refinement_counts":
                    shape = (nelv,)
                    array_dtype = np.uint32
                else:
                    raise ValueError("Invalid data key")

                # Sign and refinement bits are already densely packed and are
                # treated as approximately random. Store them verbatim rather
                # than risking bzip2 header/block expansion.
                if data_key in {"bitplane_sign_bytes", "bitplane_refinement_bytes"}:
                    raw = compressed_bytes
                else:
                    raw = bz2.decompress(compressed_bytes)
                temp = np.frombuffer(raw, dtype=array_dtype)

                uncompressed_data[field][data_key] = temp.reshape(shape)

        return uncompressed_data
 
    def _sample_fixed_error(self, field_hat: np.ndarray, field_name: str, settings: dict):
        """
        V4: sensitivity-weighted, V1-style embedded bitplane coding.

        V3 greedily selected one coefficient extension at a time. V4 instead
        visits complete *weighted* planes. For coefficient c, define

            shift[c] = round(0.5 * log2(Gcc[c] / Gref)).

        Ordinary magnitude plane p is visited at weighted plane p-shift[c].
        Consequently, a coefficient with high sensitivity enters earlier, but
        every retained coefficient still receives one contiguous MSB prefix.
        The encoder stops only after completing a weighted plane, just as V1
        stops only after completing an ordinary plane.

        Gref is the median positive Gcc in the element. It removes the arbitrary
        units/common scale of Gcc: only relative coefficient sensitivities
        affect scheduling. The target test remains

            sum_c Gcc[c] * (c[c] - c_hat[c])**2 <= target_error**2 * volume.

        The precision map is retained for a simple, robust decoder. Unlike V3,
        it is not the result of independent greedy choices; one weighted-plane
        cutoff determines the whole row.
        """

        target_error = settings["compression"]["target_error"]
        target_quantity = settings["compression"]["target_quantity"]
        allocation_mode = "weighted_planes"
        nelv = settings["mesh_information"]["nelv"]

        # Flatten within each element and use V1's spectral zigzag. Zigzag is a
        # deterministic tie-break/order inside each weighted plane.
        zigzag_order = self._spectral_zigzag_order(self.lz, self.ly, self.lx)
        y = field_hat.reshape(nelv, -1)[:, zigzag_order]
        n_coeff = y.shape[1]

        # The integer magnitude itself remains uint64.  This is different from
        # the RLE symbol type: a magnitude needs enough bits to expose many
        # precision planes, whereas a symbol only needs to hold one run length.
        # Float32 has fewer useful mantissa bits, so 31 planes are ample there.
        max_planes = 31 if self.dtype == np.float32 else 63

        # Gcc is constructed differently for field and gradient targets, but
        # the weighted-plane encoder below is identical for both quantities.
        Gcc = self.precompute_gram_diagonal(target_quantity).reshape(nelv, -1)
        Gcc = Gcc[:, zigzag_order]
        element_volumes = self._element_volume_cache[target_quantity]

        symbols_per_element = []
        sign_bytes_per_element = []
        refinement_bytes_per_element = []
        symbol_counts = np.zeros(nelv, dtype=np.uint32)
        sign_counts = np.zeros(nelv, dtype=np.uint32)
        refinement_counts = np.zeros(nelv, dtype=np.uint32)
        exponents = np.zeros(nelv, dtype=np.int16)
        precision = np.zeros((nelv, n_coeff), dtype=np.uint8)
        achieved_error = np.zeros(nelv, dtype=self.dtype)

        for element in range(nelv):
            values = y[element]
            abs_values = np.abs(values)
            maximum = float(np.max(abs_values))

            gram_diagonal = Gcc[element]
            element_volume = float(element_volumes[element])
            element_volume = element_volume if element_volume > 0.0 else 1.0
            coefficient_error = values.astype(np.float64, copy=True)
            initial_sse = float(np.dot(gram_diagonal, coefficient_error**2))
            initial_sse = max(initial_sse, 0.0)
            initial_error = np.sqrt(initial_sse / element_volume)

            # Before transmitting anything the decoder reconstructs all
            # coefficients as zero.  If that already meets the requested
            # target RMS, this element requires no stream data.
            if maximum == 0.0:
                achieved_error[element] = initial_error
                symbols_per_element.append(np.empty(0, dtype=np.uint32))
                sign_bytes_per_element.append(np.empty(0, dtype=np.uint8))
                refinement_bytes_per_element.append(np.empty(0, dtype=np.uint8))
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

            target_sse = float(target_error * target_error * element_volume)

            # Integer shifts keep the schedule discrete and decoder-friendly.
            # Zero-sensitivity coefficients are scheduled last; for a gradient
            # target this includes the constant mode, which is handled below.
            positive = gram_diagonal > 0.0
            if np.any(positive):
                reference = float(np.median(gram_diagonal[positive]))
                shifts = np.full(n_coeff, -max_planes, dtype=np.int16)
                relative = gram_diagonal[positive] / reference
                shifts[positive] = np.rint(0.5 * np.log2(relative)).astype(np.int16)
                shifts = np.clip(shifts, -max_planes, max_planes)
            else:
                shifts = np.full(n_coeff, -max_planes, dtype=np.int16)

            reconstructed = np.zeros(n_coeff, dtype=np.float64)

            if target_quantity == "gradient":
                # The constant mode has zero gradient and cannot be protected
                # by a gradient objective. Preserve all of its available bits
                # so compression does not change the element mean.
                mean_magnitude = int(magnitude[0])
                if mean_magnitude:
                    lowest = (mean_magnitude & -mean_magnitude).bit_length() - 1
                    precision[element, 0] = max_planes - lowest
                    reconstructed[0] = mean_magnitude * quantum
                    if values[0] < 0.0:
                        reconstructed[0] *= -1.0

            coefficient_error = values.astype(np.float64) - reconstructed
            total_sse = float(np.dot(gram_diagonal, coefficient_error**2))

            # Complete weighted planes are the only stopping points. At a
            # weighted plane q, coefficient c contributes its ordinary plane
            # p=q+shift[c], if that plane exists and was not already forced.
            first_weighted_plane = int(np.min(-shifts))
            last_weighted_plane = int(np.max((max_planes - 1) - shifts))
            for weighted_plane in range(first_weighted_plane, last_weighted_plane + 1):
                if total_sse <= target_sse:
                    break
                for coefficient in range(n_coeff):
                    if not positive[coefficient]:
                        continue
                    ordinary_plane = weighted_plane + int(shifts[coefficient])
                    if ordinary_plane < 0 or ordinary_plane >= max_planes:
                        continue
                    if ordinary_plane < int(precision[element, coefficient]):
                        continue

                    # Contiguous-prefix rule: reaching plane p retains every
                    # original coefficient bit from the MSB through p.
                    new_precision = ordinary_plane + 1
                    old_value = reconstructed[coefficient]
                    retained_shift = max_planes - new_precision
                    retained = (
                        int(magnitude[coefficient]) >> retained_shift
                    ) << retained_shift
                    new_value = retained * quantum
                    if values[coefficient] < 0.0:
                        new_value = -new_value
                    reconstructed[coefficient] = new_value
                    precision[element, coefficient] = new_precision

                    old_error = values[coefficient] - old_value
                    new_error = values[coefficient] - new_value
                    total_sse += gram_diagonal[coefficient] * (
                        new_error * new_error - old_error * old_error
                    )

            # This is deliberately only the diagonal estimate.  Exact physical
            # validation can be performed globally after decoding, outside the
            # codec hot path.
            achieved_error[element] = np.sqrt(max(total_sse, 0.0) / element_volume)

            # Only first-significance decisions are run-length encoded.
            # Refinement bits for coefficients that are already significant are
            # expected to be much closer to 50/50, so store them densely rather
            # than expanding short runs into uint32 symbols.  Signs remain in a
            # third packed stream and follow first significance.
            significance_events, refinement_bits, sign_bits = self._build_bitplane_streams(
                magnitude, values < 0.0, precision[element], max_planes
            )

            encoded_symbols = self._run_length_encode_bits(significance_events)
            symbol_counts[element] = encoded_symbols.size
            symbols_per_element.append(encoded_symbols)
            refinement_counts[element] = len(refinement_bits)
            refinement_bytes_per_element.append(
                np.packbits(
                    np.asarray(refinement_bits, dtype=np.uint8), bitorder="little"
                )
            )
            sign_counts[element] = len(sign_bits)
            sign_bytes_per_element.append(
                np.packbits(np.asarray(sign_bits, dtype=np.uint8), bitorder="little")
            )

        if symbols_per_element:
            symbols = np.concatenate(symbols_per_element).astype(np.uint32, copy=False)
        else:
            symbols = np.empty(0, dtype=np.uint32)
        if sign_bytes_per_element:
            sign_bytes = np.concatenate(sign_bytes_per_element).astype(np.uint8, copy=False)
        else:
            sign_bytes = np.empty(0, dtype=np.uint8)
        if refinement_bytes_per_element:
            refinement_bytes = np.concatenate(refinement_bytes_per_element).astype(
                np.uint8, copy=False
            )
        else:
            refinement_bytes = np.empty(0, dtype=np.uint8)

        stats = {
            "bitplane_format": "v4_weighted_planes_rle_significance_raw_refinement_signs",
            "target_quantity": target_quantity,
            "allocation_mode": allocation_mode,
            "avg_bitplanes": float(np.mean(precision)),
            "avg_estimated_achieved_error": float(np.mean(achieved_error)),
            "estimated_target_reached": bool(np.all(achieved_error <= target_error)),
        }

        bitplane_data = {
            "bitplane_symbols": symbols,
            "bitplane_symbol_counts": symbol_counts,
            "bitplane_exponents": exponents,
            "bitplane_precision": precision,
            "bitplane_sign_bytes": sign_bytes,
            "bitplane_sign_counts": sign_counts,
            "bitplane_refinement_bytes": refinement_bytes,
            "bitplane_refinement_counts": refinement_counts,
        }
        return bitplane_data, stats

    @staticmethod
    def _build_bitplane_streams(magnitude, negative, precision, max_planes):
        """Split significance decisions, refinement bits, and signs.

        A coefficient that has not yet emitted a one contributes to the
        significance stream.  Its first one makes it significant and emits its
        sign.  Every later retained bit contributes to the refinement stream.
        The decoder reproduces this state machine from the precision map, so no
        per-bit tags are needed.
        """
        significance_events = []
        refinement_bits = []
        sign_bits = []
        significant = np.zeros(magnitude.size, dtype=bool)
        for plane in range(int(np.max(precision, initial=0))):
            active = precision > plane
            if not np.any(active):
                continue
            indices = np.flatnonzero(active)
            bit_position = max_planes - 1 - plane
            bits = ((magnitude[indices] >> np.uint64(bit_position)) & np.uint64(1)).astype(np.uint8)
            was_significant = significant[indices]
            if np.any(was_significant):
                refinement_bits.extend(bits[was_significant].tolist())

            insignificant_indices = indices[~was_significant]
            insignificant_bits = bits[~was_significant]
            significance_events.extend(insignificant_bits.tolist())
            first_one = insignificant_bits.astype(bool)
            if np.any(first_one):
                newly_significant = insignificant_indices[first_one]
                significant[newly_significant] = True
                sign_bits.extend(negative[newly_significant].astype(np.uint8).tolist())

        return significance_events, refinement_bits, sign_bits

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

        Here both the run length and its binary value are explicit.  The caller
        now supplies only first-significance decisions; refinement and sign bits
        are densely bit-packed in separate streams.  Runs may still cross plane
        boundaries.  bzip2 later compresses the uint32 symbols.

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
        Decode the V4 weighted-plane result from its retained precision map.

        The precision map tells the decoder whether each coefficient appears on
        a plane. First-significance decisions use the RLE stream. Refinements
        and signs use separate packed streams.
        """
        n_coeff = self.lx * self.ly * self.lz
        max_planes = 31 if self.dtype == np.float32 else 63
        # Decode in transmitted zigzag order, then invert that permutation.
        output_zigzag = np.zeros((self.nelv, n_coeff), dtype=self.dtype)

        symbols = data["bitplane_symbols"]
        counts = data["bitplane_symbol_counts"]
        exponents = data["bitplane_exponents"]
        precision = data["bitplane_precision"]
        sign_bytes = data["bitplane_sign_bytes"]
        sign_counts = data["bitplane_sign_counts"]
        refinement_bytes = data["bitplane_refinement_bytes"]
        refinement_counts = data["bitplane_refinement_counts"]
        symbol_offset = 0
        sign_byte_offset = 0
        refinement_byte_offset = 0

        for element in range(self.nelv):
            # Streams from all elements live in one concatenated symbol array.
            # count identifies this element's slice and advances the offset to
            # the beginning of the next one.
            count = int(counts[element])
            element_symbols = symbols[symbol_offset:symbol_offset + count]
            symbol_offset += count
            element_precision = precision[element]
            number_of_planes = int(np.max(element_precision))
            element_sign_count = int(sign_counts[element])
            element_sign_nbytes = (element_sign_count + 7) // 8
            packed_element_signs = sign_bytes[
                sign_byte_offset:sign_byte_offset + element_sign_nbytes
            ]
            sign_byte_offset += element_sign_nbytes
            element_signs = np.unpackbits(
                packed_element_signs, bitorder="little"
            )[:element_sign_count]
            sign_offset = 0
            element_refinement_count = int(refinement_counts[element])
            element_refinement_nbytes = (element_refinement_count + 7) // 8
            packed_element_refinements = refinement_bytes[
                refinement_byte_offset:
                refinement_byte_offset + element_refinement_nbytes
            ]
            refinement_byte_offset += element_refinement_nbytes
            element_refinements = np.unpackbits(
                packed_element_refinements, bitorder="little"
            )[:element_refinement_count]
            refinement_offset = 0

            # An element that met the target as the all-zero approximation wrote
            # no events.  output was initialized to zero, so nothing is needed.
            if number_of_planes == 0:
                continue

            # Expand significance RLE on demand. Refinements and signs are
            # consumed independently from their packed streams.
            significance_bits = self._run_length_decode_bits(element_symbols)
            significant = np.zeros(n_coeff, dtype=bool)
            negative = np.zeros(n_coeff, dtype=bool)
            magnitude = np.zeros(n_coeff, dtype=np.uint64)

            for plane in range(number_of_planes):
                bit_position = max_planes - 1 - plane
                for coefficient in range(n_coeff):
                    if plane >= int(element_precision[coefficient]):
                        continue
                    if significant[coefficient]:
                        if refinement_offset >= element_refinement_count:
                            raise ValueError("Packed refinement stream ended early")
                        bit = int(element_refinements[refinement_offset])
                        refinement_offset += 1
                        if bit:
                            magnitude[coefficient] |= np.uint64(1) << np.uint64(bit_position)
                    else:
                        try:
                            bit = next(significance_bits)
                        except StopIteration as exc:
                            raise ValueError("Significance RLE stream ended early") from exc
                        if not bit:
                            continue
                        # A newly significant coefficient consumes one packed
                        # sign bit: 0=positive, 1=negative.
                        significant[coefficient] = True
                        magnitude[coefficient] |= np.uint64(1) << np.uint64(bit_position)
                        if sign_offset >= element_sign_count:
                            raise ValueError("Separate sign stream ended early")
                        negative[coefficient] = bool(element_signs[sign_offset])
                        sign_offset += 1

            if sign_offset != element_sign_count:
                raise ValueError("Separate sign stream has unused sign bits")
            if refinement_offset != element_refinement_count:
                raise ValueError("Packed refinement stream has unused bits")
            try:
                next(significance_bits)
            except StopIteration:
                pass
            else:
                raise ValueError("Significance RLE stream has unused bits")

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
        if sign_byte_offset != sign_bytes.size:
            raise ValueError("Separate sign stream has unused bytes")
        if refinement_byte_offset != refinement_bytes.size:
            raise ValueError("Packed refinement stream has unused bytes")
        zigzag_order = self._spectral_zigzag_order(self.lz, self.ly, self.lx)
        output = np.zeros_like(output_zigzag)
        output[:, zigzag_order] = output_zigzag
        return output.reshape(self.nelv, self.lz, self.ly, self.lx)

    def reconstruct_field(self, field_name: str = None):
        data = self.uncompressed_data[field_name]
        if self.settings["compression"]["method"] == "fixed_error_bitplane_v4":
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