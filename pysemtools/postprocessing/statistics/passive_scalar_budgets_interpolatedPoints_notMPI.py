# %%
#####################################################################################
#####################################################################################
#####################################################################################
# function to read and store the interpolated fields as structured and averaged fields
#####################################################################################
import time


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
    dUjS2dxj_vec = np.column_stack([load_field(63)])
    dUiUjSdxj_vec = np.column_stack([load_field(i) for i in range(64, 67)])
    dPSdxj_vec = np.column_stack([load_field(i) for i in range(67, 70)])
    dUidSdxjdxj_vec = np.column_stack([load_field(i) for i in range(70, 73)])
    dSdUidxjdxj_vec = np.column_stack([load_field(i) for i in range(73, 76)])

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

        # %% Mean velocity gradients
        print("--------------loading mean velocity gradients...")
        start_time = time.time()

        dUidXj = np.array(input_fluid_file["dUidxj_struct"])

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Mean velocity second derivatives
        print("--------------loading mean velocity second derivatives...")
        start_time = time.time()

        d2UidXj2 = np.array(input_fluid_file["d2Uidx2_struct"])

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Pressure
        print("--------------loading pressure...")
        start_time = time.time()

        P = np.array(input_fluid_file["P_struct"])

        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Pressure gradients
        print("--------------loading pressure gradients...")
        start_time = time.time()

        dPdXj = np.array(input_fluid_file["dPdxj_struct"])

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

        S_convection = -S_convection  # Put on RHS

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

        S_residual = S_turb_diffusion + S_molecular_diffusion + S_convection

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

        S2_convection = -S2_convection  # Put on RHS

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

        dUjS2dXj = np.array(input_scalar_file["dUjS2dxj_struct"])

        dirac = [0, 4, 8]
        dUjS2dXj = dUjS2dXj - np.sum(
            Ui * dS2dXj
            + S2 * dUidXj[..., dirac]
            + 2 * UiS * dSdXj
            + 2 * S * dUiSdXj[..., dirac]
            + Ui * dS2dXj
            + S2 * dUidXj[..., dirac],
            axis=-1,
        )
        # (
        #     dUiS2dXj
        #     - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]]
        #     * dS2dXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        #     - S2 * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        #     - 2 * UiS * dSdXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        #     - 2 * S * dUiSdXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        #     - Ui[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]]
        #     * dS2dXj[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]]
        #     - S2 * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8]]
        # )

        S2_turb_diffusion = -dUjS2dXj

        output_file.create_dataset(
            "S2_turb_diffusion", data=S2_turb_diffusion, compression=None
        )
        del dUjS2dXj
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
            + S2_convection
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

        UiS_convection = -UiS_convection  # Put on RHS

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

        dUiUjSdXj = np.array(input_scalar_file["dUiUjSdxj_struct"])

        for i in range(3):
            dUiUjSdXj[..., i] = (
                dUiUjSdXj[..., i]
                - Ui[..., i] * np.sum(Ui * dSdXj, axis=-1)
                - S * np.sum(Ui * dUidXj[..., 3 * i : 3 * i + 3], axis=-1)
                - Ui[..., i] * S * np.sum(dUidXj[..., [0, 4, 8]], axis=-1)
                - np.sum(Ui * dUiSdXj[..., 3 * i : 3 * i + 3], axis=-1)
                - UiS[..., i] * np.sum(dUidXj[..., [0, 4, 8]], axis=-1)
                - Ui[..., i] * np.sum(dUiSdXj[..., [0, 4, 8]], axis=-1)
                - np.sum(UiS * dUidXj[..., 3 * i : 3 * i + 3], axis=-1)
            )

        dUiUjSdXj[..., 0] = (
            dUiUjSdXj[..., 0]
            - S * np.sum(dUiUjdXk[..., [0, 10, 14]], axis=-1)
            - np.sum(UiUj[..., [0, 3, 4]] * dSdXj, axis=-1)
        )
        dUiUjSdXj[..., 1] = (
            dUiUjSdXj[..., 1]
            - S * np.sum(dUiUjdXk[..., [9, 4, 17]], axis=-1)
            - np.sum(UiUj[..., [3, 1, 5]] * dSdXj, axis=-1)
        )
        dUiUjSdXj[..., 2] = (
            dUiUjSdXj[..., 2]
            - S * np.sum(dUiUjdXk[..., [12, 16, 8]], axis=-1)
            - np.sum(UiUj[..., [4, 5, 2]] * dSdXj, axis=-1)
        )

        UiS_turb_diffusion = -np.sum(dUiUjSdXj, axis=-1)

        # inds_i = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1])
        # inds_j = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2])
        # inds_k = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        # inds_didk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5])
        # inds_djdk = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8])
        # inds_ij = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5])
        # dUiUjSdXk = (
        #     dUiUjSdXk
        #     - Ui[..., inds_i] * Ui[..., inds_j] * dSdXj[..., inds_k]
        #     - Ui[..., inds_j] * S * dUidXj[..., inds_didk]
        #     - Ui[..., inds_i] * S * dUidXj[..., inds_djdk]
        #     - Ui[..., inds_j] * dUiSdXj[..., inds_didk]
        #     - UiS[..., inds_i] * dUidXj[..., inds_djdk]
        #     - Ui[..., inds_i] * dUiSdXj[..., inds_djdk]
        #     - UiS[..., inds_j] * dUidXj[..., inds_didk]
        #     - S * dUiUjdXk
        #     - UiUj[..., inds_ij] * dSdXj[..., inds_k]
        # )

        # UiS_turb_diffusion = np.zeros((*Nxyz, 3))
        # for i in range(3):
        #     inds = np.where((inds_i == i) & (inds_j == inds_k))[0]
        #     UiS_turb_diffusion[..., i] = -np.sum(dUiUjSdXk[..., inds], axis=-1)

        output_file.create_dataset(
            "UiS_turb_diffusion", data=UiS_turb_diffusion, compression=None
        )
        del dUiUjSdXk
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: molecular diffusion
        print(
            "--------------working on turbulent scalar flux eqn: molecular diffusion..."
        )
        start_time = time.time()

        dSdUidXjdXj = np.column_stack(
            [
                np.array(input_scalar_file["dSdUdxjdxj_struct"]),
                np.array(input_scalar_file["dSdVdxjdxj_struct"]),
                np.array(input_scalar_file["dSdWdxjdxj_struct"]),
            ]
        )

        for i in range(3):
            dSdUidXjdXj[..., i] = (
                dSdUidXjdXj[..., i]
                - np.sum(dSdXj * dUidXj[..., 3 * i : 3 * i + 3], axis=-1)
                - S * np.sum(d2UidXj2[..., 3 * i : 3 * i + 3], axis=-1)
            )

        dUidSdXjdXj = np.column_stack(
            [
                np.array(input_scalar_file["dUdxjSdxj_struct"]),
                np.array(input_scalar_file["dVdxjSdxj_struct"]),
                np.array(input_scalar_file["dWdxjSdxj_struct"]),
            ]
        )

        for i in range(3):
            dUidSdXjdXj[..., i] = (
                dUidSdXjdXj[..., i]
                - np.sum(dSdXj * dUidXj[..., 3 * i : 3 * i + 3], axis=-1)
                - Ui[..., i] * np.sum(d2SdXj2, axis=-1)
            )

        UiS_molecular_diffusion = (1 / (Rer_here * Prt_here)) * dUidSdXjdXj + (
            1 / Rer_here
        ) * dSdUidXjdXj

        output_file.create_dataset(
            "UiS_molecular_diffusion", data=UiS_molecular_diffusion, compression=None
        )
        del dSdUidXjdXj, dUidSdXjdXj
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: scalar-pressure correlation gradient
        print(
            "--------------working on turbulent scalar flux eqn: scalar-pressure correlation gradient..."
        )
        start_time = time.time()

        dPSdXj = np.array(input_scalar_file["dPSdxj_struct"])

        dPSdXj = dPSdXj - P * dSdXj - S * dPdXj

        UiS_scalar_pressure_correlation_gradient = -dPSdXj

        output_file.create_dataset(
            "UiS_scalar_pressure_correlation_gradient",
            data=UiS_scalar_pressure_correlation_gradient,
            compression=None,
        )
        del dPSdXj
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: pressure-scalar gradient correlation
        print(
            "--------------working on turbulent scalar flux eqn: pressure-scalar gradient correlation..."
        )
        start_time = time.time()

        PdSdXi = np.array(input_scalar_file["PdSdxi_struct"])

        PdSdXi = PdSdXi - P * dSdXj

        UiS_pressure_scalar_gradient_correlation = PdSdXi

        output_file.create_dataset(
            "UiS_pressure_scalar_gradient_correlation",
            data=UiS_pressure_scalar_gradient_correlation,
            compression=None,
        )
        del PdSdXi
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: dissipation
        print("--------------working on turbulent scalar flux eqn: dissipation...")
        start_time = time.time()

        UiSDiss = np.array(input_scalar_file["SDiss_struct"])[..., [1, 2, 3]]

        for i in range(3):
            UiSDiss[..., i] = UiSDiss[..., i] - np.sum(
                dSdXj * dUidXj[..., 3 * i : 3 * i + 3], axis=-1
            )

        UiS_dissipation = -((1 / Rer_here) + (1 / (Rer_here * Prt_here))) * UiSDiss
        output_file.create_dataset(
            "UiS_dissipation", data=UiS_dissipation, compression=None
        )
        del UiSDiss
        print(f"Done in {time.time() - start_time:.2f} seconds.")

        # %% Turbulent scalar flux eqn: residual
        print("--------------working on turbulent scalar flux eqn: residual...")
        start_time = time.time()

        UiS_residual = (
            UiS_velocity_gradient_production
            + UiS_scalar_gradient_production
            + UiS_turb_diffusion
            + UiS_molecular_diffusion
            + UiS_scalar_pressure_correlation_gradient
            + UiS_pressure_scalar_gradient_correlation
            + UiS_dissipation
            + UiS_convection
        )

        output_file.create_dataset("UiS_residual", data=UiS_residual, compression=None)
        del (
            UiS_velocity_gradient_production,
            UiS_scalar_gradient_production,
            UiS_turb_diffusion,
            UiS_molecular_diffusion,
            UiS_scalar_pressure_correlation_gradient,
            UiS_pressure_scalar_gradient_correlation,
            UiS_dissipation,
            UiS_convection,
            UiS_residual,
        )
        print(f"Done in {time.time() - start_time:.2f} seconds.")

    # %%
    print("All computations completed and data saved successfully.")
