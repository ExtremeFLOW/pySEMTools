# %%
#####################################################################################
#####################################################################################
#####################################################################################
# function to read and store the interpolated fields as structured and averaged fields
#####################################################################################
def read_interpolated_scalar_stat_hdf5_fields(
    path_to_files,
    Reynolds_number,
    Prandtl_number,
    If_average,
    If_convert_to_single,
    Nstruct,
    av_func,
    output_fname="averaged_and_renamed_interpolated_scalar_fields.hdf5",
):

    # %%
    import h5py
    import numpy as np
    import time

    # %%
    # Generic function to load field by number
    def load_field(num, prefix="interpolated_scalar_fields"):
        """Load HDF5 field from file by number.

        Parameters:
        - num: field number (int)
        - prefix: filename prefix (default: 'interpolated_scalar_fields')

        Returns:
        - Flattened numpy array of field data
        """
        filename = path_to_files + "/" + f"{prefix}{num:05d}.hdf5"
        with h5py.File(filename, "r") as f:
            return np.array(f["/field_0"]).flatten()

    # %%
    print("reading xyz coordinates...")
    start_time = time.time()
    with h5py.File(
        path_to_files + "/" + "coordinates_interpolated_scalar_fields.hdf5", "r"
    ) as f:
        XYZ_vec = np.array(f["/xyz"]).T  # Transpose to match MATLAB's permute([2,1])
    XYZ_vec = XYZ_vec.T

    Npts = XYZ_vec.shape[0]
    print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %%
    print("reading the 42 statistics fields...")
    start_time = time.time()

    S_vec = load_field(0)
    UiS_vec = np.column_stack([load_field(i) for i in range(1, 4)])
    S2_vec = load_field(4)
    S3_vec = load_field(5)
    S4_vec = load_field(6)
    UiS2_vec = np.column_stack([load_field(i) for i in range(7, 10)])
    UiUjS_vec = np.column_stack([load_field(i) for i in range(10, 16)])
    PS_vec = load_field(16)
    PdSdxi_vec = np.column_stack([load_field(i) for i in range(17, 20)])
    UidSdxj_vec = np.column_stack([load_field(i) for i in range(20, 29)])
    SdUidxj_vec = np.column_stack([load_field(i) for i in range(29, 38)])
    SDiss_vec = np.column_stack([load_field(i) for i in range(38, 42)])

    print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %% make 1D arrays 2D
    S_vec = S_vec[:, np.newaxis]
    S2_vec = S2_vec[:, np.newaxis]
    S3_vec = S3_vec[:, np.newaxis]
    S4_vec = S4_vec[:, np.newaxis]
    PS_vec = PS_vec[:, np.newaxis]

    # %%
    print("reading the additional fields...")
    start_time = time.time()

    dSdxj_vec = np.column_stack([load_field(i) for i in range(42, 45)])
    d2Sdxj2_vec = np.column_stack([load_field(i) for i in range(45, 48)])
    dS2dxj_vec = np.column_stack([load_field(i) for i in range(48, 51)])
    d2S2dxj2_vec = np.column_stack([load_field(i) for i in range(51, 54)])
    dUiSdxj_vec = np.column_stack([load_field(i) for i in range(54, 63)])
    dUiS2dxj_vec = np.column_stack([load_field(i) for i in range(63, 72)])
    dUiUjSdxk_vec = np.column_stack([load_field(i) for i in range(72, 90)])
    dPSdxj_vec = np.column_stack([load_field(i) for i in range(90, 93)])

    print("Finished reading all fields.")
    print(f"Done in {time.time() - start_time:.2f} seconds.")

    # Save the variables in a directory
    vars = {}
    for name in list(locals().keys()):
        if name.endswith("_vec") and isinstance(locals()[name], np.ndarray):
            vars[name] = locals()[name]

    # %% Convert arrays to single precision if required
    if If_convert_to_single:
        print("Converting arrays into single precision...")
        start_time = time.time()
        for name in list(vars.keys()):
            if name.endswith("_vec") and isinstance(vars[name], np.ndarray):
                # print(np.shape(locals()[name]))
                vars[name] = vars[name].astype(np.float32)
        print("Conversion complete.")
        print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %% Reshape arrays based on Nstruct
    print("Reshaping into arrays...")
    start_time = time.time()
    for name in list(vars.keys()):
        if name.endswith("_vec") and isinstance(vars[name], np.ndarray):
            reshaped_name = name[:-4] + "_struct"
            vars[reshaped_name] = vars[name].reshape(
                (Nstruct[1], Nstruct[0], Nstruct[2], vars[name].shape[1]), order="F"
            )
            del vars[name]
    print("Reshaping complete.")
    print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %% Permute arrays to original shape
    print("Permuting arrays into the original shape...")
    start_time = time.time()
    for name in list(vars.keys()):
        if name.endswith("_struct") and isinstance(vars[name], np.ndarray):
            vars[name] = np.transpose(vars[name], (1, 0, 2, 3))
    print("Permutation complete.")
    print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %% Apply user-specified averaging function if required
    if If_average:
        print("Taking the user-specified average using function av_func...")
        start_time = time.time()
        for name in list(vars.keys()):
            if name.endswith("_struct") and isinstance(vars[name], np.ndarray):
                vars[name] = av_func(vars[name])
        print("Averaging complete.")
        print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %% Reynolds and Prandtl numbers needed to calculate viscous related terms later
    Rer_here = Reynolds_number
    Prt_here = Prandtl_number
    vars["Rer_here"] = Rer_here
    vars["Prt_here"] = Prt_here

    # %% Save the data in HDF5 format
    print("Saving the data in HDF5 format...")
    start_time = time.time()
    with h5py.File(path_to_files + "/" + output_fname, "w") as hf:
        global_vars = dict(vars)  # Create a copy to avoid modification issues
        for name, data in global_vars.items():
            if (name.endswith("_struct") or name.endswith("_here")) and isinstance(
                data, (np.ndarray, int, float)
            ):
                hf.create_dataset(name, data=data)
    print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %%
    print("Data saved successfully in HDF5 format.")


# %%
#####################################################################################
#####################################################################################
#####################################################################################
# function to read the raw but averaged fields and calcuate the scalar budgets in Cartesian coordinates
# the fluid statistics are also needed to calculate some terms in the budgets
#####################################################################################
def calculate_scalar_budgets_in_Cartesian(
    path_to_files="./",
    input_scalar_filename="averaged_and_renamed_interpolated_scalar_fields.hdf5",
    input_fluid_filename="averaged_and_renamed_interpolated_fields.hdf5",
    output_filename="sstat3d_format.hdf5",
):

    # %%
    import h5py
    import numpy as np
    import time
    import sys

    # %%
    with (
        h5py.File(
            path_to_files + "/" + input_scalar_filename, "r"
        ) as input_scalar_file,
        h5py.File(path_to_files + "/" + input_fluid_filename, "r") as input_fluid_file,
        h5py.File(path_to_files + "/" + output_filename, "w") as output_file,
    ):

        # %%
        Rer_here = np.float64(input_scalar_file["Rer_here"])
        print("Reynolds number = ", Rer_here)

        output_file.create_dataset("Rer_here", data=Rer_here, compression=None)

        # %%
        Prt_here = np.float64(input_scalar_file["Prt_here"])
        print("Prandtl number = ", Prt_here)

        output_file.create_dataset("Prt_here", data=Prt_here, compression=None)

        # %% XYZ coordinates
        print("--------------working on XYZ coordinates...")
        start_time = time.time()

        XYZ = np.array(input_scalar_file["XYZ_struct"])
        output_file.create_dataset("XYZ", data=XYZ, compression=None)
        Nxyz = XYZ.shape[:3]
        print("Number of points: ", Nxyz)

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Check if scalar XYZ coordinates match fluid XYZ coordinates
        print(
            "--------------checking if scalar XYZ coordinates match fluid XYZ coordinates..."
        )
        start_time = time.time()

        XYZ_fluid = np.array(input_fluid_file["XYZ_struct"])
        Nxyz_fluid = XYZ_fluid.shape[:3]

        # Check if number of points match
        if Nxyz != Nxyz_fluid:
            sys.exit(
                "Number of points in scalar and fluid files do not match! "
                f"Scalar: {Nxyz}, Fluid: {Nxyz_fluid}"
            )

        # Check if coordinates match
        if not np.allclose(XYZ, XYZ_fluid):
            sys.exit("EXYZ coordinates in scalar and fluid files do not match!")
        else:
            print("XYZ coordinates match between scalar and fluid files.")

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Load required fluid statistics for calculating scalar budgets
        # %% Mean velocities
        print("--------------loading mean velocities...")
        start_time = time.time()

        Ui = np.array(input_fluid_file["UVW_struct"])

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Mean velocity derivatives
        print("--------------loading mean velocity derivatives...")
        start_time = time.time()

        dUidXj = np.array(input_fluid_file["dUidxj_struct"])

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Pressure
        print("--------------loading pressure...")
        start_time = time.time()

        P = np.array(input_fluid_file["P_struct"])

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Reynolds stresses
        print("--------------loading Reynolds stresses...")
        start_time = time.time()

        UiUj = np.array(input_fluid_file["UiUj_struct"])
        UiUj = UiUj - (Ui[..., [0, 1, 2, 0, 0, 1]] * Ui[..., [0, 1, 2, 1, 2, 2]])

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Reynolds stress gradients
        print("--------------loading Reynolds stress gradients...")
        start_time = time.time()

        dUiUjdXk = np.array(input_fluid_file["UiUjdx_struct"])
        dUiUjdXk = (
            dUiUjdXk
            - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
            * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]]
            - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]]
            * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5]]
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Start processing scalar statistics

        # %% Scalar and Its Moments
        print("--------------working on scalar and its moments...")
        start_time = time.time()

        S = np.array(input_scalar_file["S_struct"])
        S2 = np.array(input_scalar_file["S2_struct"])
        S3 = np.array(input_scalar_file["S3_struct"])
        S4 = np.array(input_scalar_file["S4_struct"])

        S2 = S2 - S**2
        S3 = S3 - S**3 - 3 * S * S2
        S4 = S4 - S**4 - 6 * S**2 * S2 - 4 * S * S3

        output_file.create_dataset("S", data=S, compression=None)
        output_file.create_dataset("S2", data=S2, compression=None)
        output_file.create_dataset("S3", data=S3, compression=None)
        output_file.create_dataset("S4", data=S4, compression=None)

        del S3, S4
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent Scalar Fluxes
        print("--------------working on scalar fluxes...")
        start_time = time.time()

        UiS = np.array(input_fluid_file["UiS_struct"])

        UiS = UiS - Ui * S

        output_file.create_dataset("UiS", data=UiS, compression=None)

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar variance fluxes
        print("--------------working on scalar variance fluxes...")
        start_time = time.time()

        UiS2 = np.array(input_fluid_file["UiS2_struct"])

        UiS2 = UiS2 - Ui * S2 - 2 * S * UiS - Ui * S**2

        output_file.create_dataset("UiS2", data=UiS2, compression=None)
        del UiS2
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux transport
        print("--------------working on scalar flux transport...")
        start_time = time.time()

        UiUjS = np.array(input_fluid_file["UiUjS_struct"])

        UiUjS = (
            UiUjS
            - Ui[..., [0, 1, 2, 0, 0, 1]] * UiS[..., [0, 1, 2, 1, 2, 2]]
            - Ui[..., [0, 1, 2, 1, 2, 2]] * UiS[..., [0, 1, 2, 0, 0, 1]]
            - S * UiUj[..., [0, 1, 2, 3, 4, 5]]
            - S * Ui[..., [0, 1, 2, 0, 0, 1]] * Ui[..., [0, 1, 2, 1, 2, 2]]
        )

        output_file.create_dataset("UiUjS", data=UiUjS, compression=None)
        del UiUjS
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Pressure-scalar correlation
        print("--------------working on pressure-scalar correlation...")
        start_time = time.time()

        PS = np.array(input_scalar_file["PS_struct"])

        PS = PS - P * S

        output_file.create_dataset("PS", data=PS, compression=None)

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Mean scalar gradients
        print("--------------working on mean scalar gradients...")
        start_time = time.time()

        dSdXj = np.array(input_scalar_file["dSdxj_struct"])

        output_file.create_dataset("dSdXj", data=dSdXj, compression=None)
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Mean scalar eqn: convection
        print("--------------working on mean scalar eqn: convection...")

        start_time = time.time()

        S_convection = np.sum(Ui * dSdXj, axis=-1)

        output_file.create_dataset("S_convection", data=S_convection, compression=None)
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Mean scalar eqn: turbulent diffusion
        print("--------------working on mean scalar eqn: turbulent diffusion...")
        start_time = time.time()

        dUiSdXj = np.array(input_scalar_file["dUiSdxj_struct"])
        dUiSdXj = (
            dUiSdXj
            - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]]
            * dSdXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
            - S * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        )

        S_turb_diffusion = -np.sum(dUiSdXj[:, 0:3], axis=-1)

        output_file.create_dataset(
            "S_turb_diffusion", data=S_turb_diffusion, compression=None
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Mean scalar eqn: molecular diffusion
        print("--------------working on mean scalar eqn: molecular diffusion...")
        start_time = time.time()

        d2SdXj2 = np.array(input_scalar_file["d2Sdxj2_struct"])

        S_molecular_diffusion = (1 / (Rer_here * Prt_here)) * np.sum(d2SdXj2, axis=-1)

        output_file.create_dataset(
            "S_molecular_diffusion", data=S_molecular_diffusion, compression=None
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Mean scalar eqn: residual
        print("--------------working on mean scalar eqn: residual...")
        start_time = time.time()

        S_residual = S_turb_diffusion + S_molecular_diffusion - S_convection

        output_file.create_dataset("S_residual", data=S_residual, compression=None)
        del (
            S_convection,
            S_turb_diffusion,
            S_molecular_diffusion,
            S_residual,
        )
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Scalar variance eqn: convection
        print("--------------working on scalar variance eqn: convection...")
        start_time = time.time()

        dS2dXj = np.array(input_scalar_file["dS2dxj_struct"])
        S2_convection = np.sum(Ui * dS2dXj, axis=-1)

        output_file.create_dataset(
            "S2_convection", data=S2_convection, compression=None
        )
        del dS2dXj
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Scalar variance eqn: production
        print("--------------working on scalar variance eqn: production...")
        start_time = time.time()

        S2_production = -2 * np.sum(UiS * dSdXj, axis=-1)

        output_file.create_dataset(
            "S2_production", data=S2_production, compression=None
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Scalar variance eqn: turbulent diffusion
        print("--------------working on scalar variance eqn: turbulent diffusion...")
        start_time = time.time()

        dUiS2dXj = np.array(input_scalar_file["dUiS2dxj_struct"])
        dUiS2dXj = (
            dUiS2dXj
            - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]]
            * dS2dXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
            - S2 * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8]]
            - 2 * UiS * dSdXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
            - 2 * S * dUiSdXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8]]
            - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]]
            * dS2dXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
            - S2 * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        )

        S2_turb_diffusion = -np.sum(dUiS2dXj[:, 0:3], axis=-1)

        output_file.create_dataset(
            "S2_turb_diffusion", data=S2_turb_diffusion, compression=None
        )
        del dUiS2dXj
        print(f"Done in {time.time() - 0:.2f} seconds.")

        # %% Scalar variance eqn: molecular diffusion
        print("--------------working on scalar variance eqn: molecular diffusion...")
        start_time = time.time()

        d2S2dXi2 = np.array(input_scalar_file["d2S2dxj2_struct"])
        d2S2dXi2 = d2S2dXi2 - 2 * dSdXj**2 - 2 * S * d2SdXj2

        S2_molecular_diffusion = 1 / (Rer_here * Prt_here) * np.sum(d2S2dXi2, axis=-1)

        output_file.create_dataset(
            "S2_molecular_diffusion", data=S2_molecular_diffusion, compression=None
        )
        del d2S2dXi2, d2SdXj2
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Scalar variance eqn: dissipation
        print("--------------working on scalar variance eqn: dissipation...")
        start_time = time.time()

        S2Diss = np.array(input_scalar_file["SDiss_struct"])[..., 0]
        S2Diss = S2Diss - np.sum(dSdXj**2, axis=-1)

        S2_dissipation = -2 * (1 / (Rer_here * Prt_here)) * S2Diss

        output_file.create_dataset(
            "S2_dissipation", data=S2_dissipation, compression=None
        )
        del S2Diss
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Scalar variance eqn: residual
        print("--------------working on scalar variance eqn: residual...")
        start_time = time.time()

        S2_residual = (
            S2_production
            + S2_turb_diffusion
            + S2_molecular_diffusion
            + S2_dissipation
            - S2_convection
        )

        output_file.create_dataset("S2_residual", data=S2_residual, compression=None)
        del (
            S2_production,
            S2_turb_diffusion,
            S2_molecular_diffusion,
            S2_dissipation,
            S2_convection,
            S2_residual,
        )
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: convection
        print("--------------working on turbulent scalar flux eqn: convection...")
        start_time = time.time()

        UiS_convection = np.zeros((*Nxyz, 3))
        UiS_convection[..., 0] = np.sum(Ui * dUiSdXj[..., :3], axis=-1)
        UiS_convection[..., 1] = np.sum(Ui * dUiSdXj[..., 3:6], axis=-1)
        UiS_convection[..., 2] = np.sum(Ui * dUiSdXj[..., 6:9], axis=-1)

        output_file.create_dataset(
            "UiS_convection", data=UiS_convection, compression=None
        )

        # %% Turbulent scalar flux eqn: velocity gradient production
        print(
            "--------------working on turbulent scalar flux eqn: velocity gradient production..."
        )
        start_time = time.time()

        UiS_velocity_gradient_production = np.zeros((*Nxyz, 3))
        UiS_velocity_gradient_production[..., 0] = -np.sum(
            UiS * dUidXj[..., [0, 3, 6]], axis=-1
        )
        UiS_velocity_gradient_production[..., 1] = -np.sum(
            UiS * dUidXj[..., [1, 4, 7]], axis=-1
        )
        UiS_velocity_gradient_production[..., 2] = -np.sum(
            UiS * dUidXj[..., [2, 5, 8]], axis=-1
        )

        output_file.create_dataset(
            "UiS_velocity_gradient_production",
            data=UiS_velocity_gradient_production,
            compression=None,
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: scalar gradient production
        print(
            "--------------working on turbulent scalar flux eqn: scalar gradient production..."
        )
        start_time = time.time()

        UiS_scalar_gradient_production = np.zeros((*Nxyz, 3))
        UiS_scalar_gradient_production[..., 0] = -np.sum(
            UiUj[..., [0, 3, 4]] * dSdXj, axis=-1
        )
        UiS_scalar_gradient_production[..., 1] = -np.sum(
            UiUj[..., [3, 1, 5]] * dSdXj, axis=-1
        )
        UiS_scalar_gradient_production[..., 2] = -np.sum(
            UiUj[..., [4, 5, 2]] * dSdXj, axis=-1
        )

        output_file.create_dataset(
            "UiS_scalar_gradient_production", data=UiS_scalar_gradient_production
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: turbulent diffusion
        print(
            "--------------working on turbulent scalar flux eqn: turbulent diffusion..."
        )
        start_time = time.time()

        dUiUjSdXk = np.array(input_scalar_file["dUiUjSdxk_struct"])

        inds_i = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        inds_j = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        inds_k = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        inds_didk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5])
        inds_djdk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8])
        inds_ij = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        dUiUjSdXk = (
            dUiUjSdXk
            - Ui[..., inds_i] * Ui[..., inds_j] * dSdXj[..., inds_k]
            - Ui[..., inds_j] * S * dUidXj[..., inds_didk]
            - Ui[..., inds_i] * S * dUidXj[..., inds_djdk]
            - Ui[..., inds_j] * dUiSdXj[..., inds_didk]
            - UiS[..., inds_i] * dUidXj[..., inds_djdk]
            - Ui[..., inds_i] * dUiSdXj[..., inds_djdk]
            - UiS[..., inds_j] * dUidXj[..., inds_didk]
            - S * dUiUjdXk
            - UiUj[..., inds_ij] * dSdXj[..., inds_k]
        )

        UiS_turb_diffusion = np.zeros((*Nxyz, 3))
        for i in range(3):
            inds = np.where((inds_i == i) & (inds_j == inds_k))[0]
            UiS_turb_diffusion[..., i] = -np.sum(dUiUjSdXk[..., inds], axis=-1)

        output_file.create_dataset(
            "UiS_turb_diffusion", data=UiS_turb_diffusion, compression=None
        )
        del dUiUjSdXk
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: viscous diffusion
        print(
            "--------------working on turbulent scalar flux eqn: viscous diffusion..."
        )
        start_time = time.time()

        # %% Turbulent scalar flux eqn: molecular diffusion

        # %% Turbulent scalar flux eqn: scalar-pressure correlation gradient

        # %% Turbulent scalar flux eqn: pressure-scalar gradient correlation

        # %% Turbulent scalar flux eqn: dissipation

        # %% Turbulent scalar flux eqn: residual

        # %% Pressure-scalar gradient correlation
        print("--------------working on pressure-scalar gradient correlation...")
        start_time = time.time()

        PdSdXi = np.array(input_scalar_file["PdSdxi_struct"])

        PdSdXi = PdSdXi - P * dSdXj

        output_file.create_dataset("PdSdXi", data=PdSdXi, compression=None)
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Velocity-scalar gradient correlation
        print("--------------working on velocity-scalar gradient correlation...")
        start_time = time.time()

        UidSdxj = np.array(input_scalar_file["UidSdxj_struct"])

        UidSdxj = (
            UidSdxj
            - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]]
            * dSdXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        )

        output_file.create_dataset("UidSdxj", data=UidSdxj, compression=None)
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Scalar-velocity gradient correlation
        print("--------------working on scalar-velocity gradient correlation...")
        start_time = time.time()

        SdUidXj = np.array(input_scalar_file["SdUidxj_struct"])

        SdUidXj = SdUidXj - S * dUidXj

        # %% Scalar dissipation

        # SDiss_vec = np.column_stack([load_field(i) for i in range(38, 42)])
        # d2Sdx2_vec = np.column_stack([load_field(i) for i in range(45, 48)])
        # dUiSdxj_vec = np.column_stack([load_field(i) for i in range(54, 57)])
        # dUiS2dxj_vec = np.column_stack([load_field(i) for i in range(57, 60)])
        # dPSdxj_vec = np.column_stack([load_field(i) for i in range(66, 69)])

        # %%
        print("--------------working on velocity gradients...")
        start_time = time.time()

        # Read velocity gradient tensor from input file
        dUidXj = np.array(input_file["dUidxj_struct"][()])

        # Save to output file
        output_file.create_dataset("dUidXj", data=dUidXj)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on momentum convection terms...")
        start_time = time.time()

        # Reshape dUidXj
        dUidXj_reshaped = dUidXj.reshape(Nxyz[0], Nxyz[1], Nxyz[2], 3, 3, order="F")

        # Compute Momentum convection
        Momentum_convection = np.sum(UVW[..., np.newaxis] * dUidXj_reshaped, axis=3)

        # Save to output file
        output_file.create_dataset("Momentum_convection", data=Momentum_convection)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on the residual momentum convection terms...")
        start_time = time.time()

        # Compute Residual Momentum convection terms
        Momentum_convectionRes = np.zeros((*Nxyz, 3))  # Shape: (N1, N2, N3, 3)
        Momentum_convectionRes[..., 0] = np.sum(
            UVW[..., [1, 2]] * dUidXj[..., [1, 2]], axis=-1
        )
        Momentum_convectionRes[..., 1] = np.sum(
            UVW[..., [0, 2]] * dUidXj[..., [3, 5]], axis=-1
        )
        Momentum_convectionRes[..., 2] = np.sum(
            UVW[..., [0, 1]] * dUidXj[..., [6, 7]], axis=-1
        )

        # Save to output file
        output_file.create_dataset(
            "Momentum_convectionRes", data=Momentum_convectionRes
        )
        del Momentum_convectionRes
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on momentum pressure terms...")
        print("Warning: assuming rho=1 everywhere!")
        start_time = time.time()

        # Compute Momentum pressure
        Momentum_pressure = -np.array(input_file["dPdx_struct"][()])

        # Save to output file
        output_file.create_dataset("Momentum_pressure", data=Momentum_pressure)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on production terms...")
        start_time = time.time()

        Prod_ij = np.zeros((*dUidXj.shape[:3], 6))

        Prod_ij[..., 0] = -2 * np.sum(Rij[..., [0, 3, 4]] * dUidXj[..., :3], axis=3)
        Prod_ij[..., 1] = -2 * np.sum(Rij[..., [3, 1, 5]] * dUidXj[..., 3:6], axis=3)
        Prod_ij[..., 2] = -2 * np.sum(Rij[..., [4, 5, 2]] * dUidXj[..., 6:9], axis=3)
        Prod_ij[..., 3] = -np.sum(
            Rij[..., [0, 3, 4]] * dUidXj[..., 3:6]
            + Rij[..., [3, 1, 5]] * dUidXj[..., :3],
            axis=3,
        )
        Prod_ij[..., 4] = -np.sum(
            Rij[..., [0, 3, 4]] * dUidXj[..., 6:9]
            + Rij[..., [4, 5, 2]] * dUidXj[..., :3],
            axis=3,
        )
        Prod_ij[..., 5] = -np.sum(
            Rij[..., [3, 1, 5]] * dUidXj[..., 6:9]
            + Rij[..., [4, 5, 2]] * dUidXj[..., 3:6],
            axis=3,
        )

        TKE_prod = np.sum(Prod_ij[..., :3], axis=3) / 2

        output_file.create_dataset("Prod_ij", data=Prod_ij)
        output_file.create_dataset("TKE_prod", data=TKE_prod)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on convection terms...")
        start_time = time.time()

        dRij_dxk = np.array(input_file["dUiUjdx_struct"][:])

        dRij_dxk = (
            dRij_dxk
            - UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
            * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]]
            - UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]]
            * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5]]
        )

        Conv_ij = np.zeros((*UVW.shape[:3], 6))

        Conv_ij[..., 0] = np.sum(UVW[..., :3] * dRij_dxk[..., :3], axis=3)
        Conv_ij[..., 1] = np.sum(UVW[..., :3] * dRij_dxk[..., 3:6], axis=3)
        Conv_ij[..., 2] = np.sum(UVW[..., :3] * dRij_dxk[..., 6:9], axis=3)
        Conv_ij[..., 3] = np.sum(UVW[..., :3] * dRij_dxk[..., 9:12], axis=3)
        Conv_ij[..., 4] = np.sum(UVW[..., :3] * dRij_dxk[..., 12:15], axis=3)
        Conv_ij[..., 5] = np.sum(UVW[..., :3] * dRij_dxk[..., 15:18], axis=3)

        Conv_ij = -Conv_ij
        TKE_conv = np.sum(Conv_ij[..., :3], axis=3) / 2

        output_file.create_dataset("Conv_ij", data=Conv_ij)
        output_file.create_dataset("TKE_conv", data=TKE_conv)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on momentum turbulent diffusion terms...")
        start_time = time.time()

        Momentum_turb_diffusion = np.zeros((*dRij_dxk.shape[:3], 3))

        Momentum_turb_diffusion[..., 0] = -np.sum(dRij_dxk[..., [0, 10, 14]], axis=3)
        Momentum_turb_diffusion[..., 1] = -np.sum(dRij_dxk[..., [9, 4, 17]], axis=3)
        Momentum_turb_diffusion[..., 2] = -np.sum(dRij_dxk[..., [12, 16, 8]], axis=3)

        output_file.create_dataset(
            "Momentum_turb_diffusion", data=Momentum_turb_diffusion
        )
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on dissipation terms...")
        start_time = time.time()
        Diss_ij = np.array(input_file["pseudoDiss_struct"][()])

        # Compute dissipation terms
        Diss_ij[..., 0] -= np.sum(dUidXj[..., 0:3] ** 2, axis=-1)
        Diss_ij[..., 1] -= np.sum(dUidXj[..., 3:6] ** 2, axis=-1)
        Diss_ij[..., 2] -= np.sum(dUidXj[..., 6:9] ** 2, axis=-1)
        Diss_ij[..., 3] -= np.sum(dUidXj[..., 0:3] * dUidXj[..., 3:6], axis=-1)
        Diss_ij[..., 4] -= np.sum(dUidXj[..., 0:3] * dUidXj[..., 6:9], axis=-1)
        Diss_ij[..., 5] -= np.sum(dUidXj[..., 3:6] * dUidXj[..., 6:9], axis=-1)

        Diss_ij = -2.0 / Rer_here * Diss_ij
        TKE_diss = np.sum(Diss_ij[..., 0:3], axis=-1) / 2

        # Save results to output HDF5 file
        output_file.create_dataset("Diss_ij", data=Diss_ij, compression=None)
        output_file.create_dataset("TKE_diss", data=TKE_diss, compression=None)
        del Diss_ij
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        # %%
        print("--------------working on turbulent transport...")
        start_time = time.time()

        # Define index mappings
        ind_dRjkdxk = np.array(
            [
                (np.array([1, 4, 5]) - 1) * 3 + np.array([1, 2, 3]) - 1,
                (np.array([4, 2, 6]) - 1) * 3 + np.array([1, 2, 3]) - 1,
                (np.array([5, 6, 3]) - 1) * 3 + np.array([1, 2, 3]) - 1,
            ]
        )

        ind_tripple = np.zeros((3, 3, 3), dtype=int)
        ind_tripple[:, 0, 0] = (np.array([1, 4, 5]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 0, 1] = (np.array([4, 6, 10]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 0, 2] = (np.array([5, 10, 8]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 1, 0] = (np.array([4, 6, 10]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 1, 1] = (np.array([6, 2, 7]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 1, 2] = (np.array([10, 7, 9]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 2, 0] = (np.array([5, 10, 8]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 2, 1] = (np.array([10, 7, 9]) - 1) * 3 + np.array([1, 2, 3]) - 1
        ind_tripple[:, 2, 2] = (np.array([8, 9, 3]) - 1) * 3 + np.array([1, 2, 3]) - 1

        # Allocate memory
        TurbTrans_ij = np.zeros((*Nxyz, 3, 3), dtype=np.float32)

        # Compute turbulent transport
        for i in range(3):
            for j in range(3):
                TurbTrans_ij[..., i, j] = (
                    -np.array(input_file["dUiUjUkdx_struct"][..., ind_tripple[0, i, j]])
                    - np.array(
                        input_file["dUiUjUkdx_struct"][..., ind_tripple[1, i, j]]
                    )
                    - np.array(
                        input_file["dUiUjUkdx_struct"][..., ind_tripple[2, i, j]]
                    )
                )

                TurbTrans_ij[..., i, j] += (
                    UVW[..., i]
                    * np.sum(UVW * dUidXj[..., (np.arange(3) + j * 3)], axis=-1)
                    + UVW[..., j]
                    * np.sum(UVW * dUidXj[..., (np.arange(3) + i * 3)], axis=-1)
                    + UVW[..., i] * np.sum(dRij_dxk[..., ind_dRjkdxk[j, :]], axis=-1)
                    + UVW[..., j] * np.sum(dRij_dxk[..., ind_dRjkdxk[i, :]], axis=-1)
                )

        # Reorder indices
        TurbTrans_ij = TurbTrans_ij.reshape(TurbTrans_ij.shape[:-2] + (-1,), order="F")
        TurbTrans_ij = TurbTrans_ij[..., [0, 4, 8, 1, 2, 5]]

        # Final computation
        TurbTrans_ij -= Conv_ij + Prod_ij
        TKE_turbTrans = np.sum(TurbTrans_ij[..., 0:3], axis=-1) / 2

        # Save results
        output_file.create_dataset("TurbTrans_ij", data=TurbTrans_ij, compression=None)
        output_file.create_dataset(
            "TKE_turbTrans", data=TKE_turbTrans, compression=None
        )

        # Cleanup
        del TurbTrans_ij, Conv_ij, Prod_ij, dRij_dxk

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %%
        print("--------------working on viscous diffusion...")
        start_time = time.time()

        # Read datasets
        d2Ui_dx2 = np.array(input_file["d2Uidx2_struct"])
        d2Rij_dx2 = np.array(input_file["d2UiUjdx2_struct"])

        # Compute d2Rij_dx2
        d2Rij_dx2 = (
            d2Rij_dx2
            - UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
            * d2Ui_dx2[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]]
            - UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]]
            * d2Ui_dx2[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5]]
            - 2
            * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5]]
            * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]]
        )

        # Compute viscous diffusion
        ViscDiff_ij = (1 / Rer_here) * np.sum(
            d2Rij_dx2.reshape((*Nxyz, 3, -1), order="F"), axis=3
        )

        # Compute TKE viscous diffusion
        TKE_ViscDiff = np.sum(ViscDiff_ij[..., 0:3], axis=-1) / 2

        # Save results
        output_file.create_dataset("ViscDiff_ij", data=ViscDiff_ij, compression=None)
        output_file.create_dataset("TKE_ViscDiff", data=TKE_ViscDiff, compression=None)

        # Cleanup
        del d2Rij_dx2, ViscDiff_ij

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Momentum viscous diffusion terms
        print("--------------working on momentum viscous diffusion terms...")
        start_time = time.time()

        Momentum_viscous_diffusion = (1 / Rer_here) * np.sum(
            d2Ui_dx2.reshape(*Nxyz, 3, 3, order="F"), axis=3
        )

        output_file.create_dataset(
            "Momentum_viscous_diffusion",
            data=Momentum_viscous_diffusion,
            compression=None,
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Momentum residual terms
        print("--------------working on momentum residual terms...")
        start_time = time.time()

        Momentum_residual = (
            Momentum_viscous_diffusion
            + Momentum_turb_diffusion
            + Momentum_pressure
            - Momentum_convection
        )

        output_file.create_dataset(
            "Momentum_residual", data=Momentum_residual, compression=None
        )

        # Cleanup
        del (
            Momentum_viscous_diffusion,
            Momentum_turb_diffusion,
            Momentum_pressure,
            Momentum_convection,
            Momentum_residual,
        )

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Pressure-rate-of-strain terms
        print("--------------working on pressure-rate-of-strain terms...")
        start_time = time.time()

        PRS_ij = np.array(input_file["PGij_struct"])
        PRS_ij = PRS_ij - P * dUidXj
        PRS_ij = PRS_ij[..., [0, 4, 8, 1, 2, 5]] + PRS_ij[..., [0, 4, 8, 3, 6, 7]]

        output_file.create_dataset("PRS_ij", data=PRS_ij, compression=None)

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %%
        print("--------------working on pressure transport terms...")
        start_time = time.time()

        dpdx = np.array(input_file["dPdx_struct"])
        dpudx = np.array(input_file["dPUidxj_struct"])

        dpudx = (
            dpudx
            - (P * dUidXj)
            - (
                UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]]
                * dpdx[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
            )
        )

        PressTrans_ij = -(
            dpudx[..., [0, 4, 8, 1, 2, 5]] + dpudx[..., [0, 4, 8, 3, 6, 7]]
        )

        output_file.create_dataset("dpdx", data=dpdx, compression=None)
        output_file.create_dataset(
            "PressTrans_ij", data=PressTrans_ij, compression=None
        )

        del dpudx
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Velocity-Pressure Gradient Terms
        print("--------------working on velocity-pressure-gradient terms...")
        start_time = time.time()

        VelPressGrad_ij = PressTrans_ij - PRS_ij
        TKE_VelPressGrad = np.sum(VelPressGrad_ij[..., 0:3], axis=-1) / 2

        output_file.create_dataset(
            "VelPressGrad_ij", data=VelPressGrad_ij, compression=None
        )
        output_file.create_dataset(
            "TKE_VelPressGrad", data=TKE_VelPressGrad, compression=None
        )

        del PRS_ij, PressTrans_ij, VelPressGrad_ij
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% TKE budget residual
        print("--------------working on TKE budgets...")
        start_time = time.time()

        TKE_residual = (
            TKE_prod
            + TKE_diss
            + TKE_turbTrans
            + TKE_ViscDiff
            + TKE_VelPressGrad
            + TKE_conv
        )

        output_file.create_dataset("TKE_residual", data=TKE_residual, compression=None)

        del (
            TKE_prod,
            TKE_diss,
            TKE_turbTrans,
            TKE_ViscDiff,
            TKE_VelPressGrad,
            TKE_conv,
            TKE_residual,
        )
        print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %%
    print("All computations completed and data saved successfully.")
