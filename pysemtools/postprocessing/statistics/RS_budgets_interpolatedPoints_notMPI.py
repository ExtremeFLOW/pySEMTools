
#%%
#####################################################################################
#####################################################################################
#####################################################################################
# function to read and store the interpolated fields as structured and averaged fields
#####################################################################################
def read_interpolated_stat_hdf5_fields( 
        path_to_files , 
        Reynolds_number , 
        If_average , If_convert_to_single , 
        Nstruct , av_func , 
        output_fname='averaged_and_renamed_interpolated_fields.hdf5'
        ):
    
    #%%
    import h5py
    import numpy as np
    import time

    #%%
    print('reading xyz coordinates...')
    start_time = time.time()
    with h5py.File(path_to_files+'/'+'coordinates_interpolated_fields.hdf5', 'r') as f:
        XYZ_vec = np.array(f['/xyz']).T  # Transpose to match MATLAB's permute([2,1])
    XYZ_vec = XYZ_vec.T 

    Npts = XYZ_vec.shape[0]
    print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%%
    print('reading the 44 statistics fields...')
    start_time = time.time()
    UVW_vec = np.zeros((Npts, 3))
    for i in range(1, 4):
        filename = path_to_files+'/'+f'interpolated_fields0000{i}.hdf5'
        with h5py.File(filename, 'r') as f:
            UVW_vec[:, i-1] = np.array(f['/field_0']).flatten()

    with h5py.File(path_to_files+'/'+'interpolated_fields00004.hdf5', 'r') as f:
        P_vec = np.array(f['/field_0']).flatten()

    with h5py.File(path_to_files+'/'+'interpolated_fields00005.hdf5', 'r') as f:
        P2_vec = np.array(f['/field_0']).flatten()

    UiUj_vec = np.zeros((Npts, 6))
    for i in range(1, 7):
        tmp = str(5 + i)
        fnum = ('000000'[:5 - len(tmp)] + tmp)
        filename = path_to_files+'/'+f'interpolated_fields{fnum}.hdf5'
        with h5py.File(filename, 'r') as f:
            UiUj_vec[:, i-1] = np.array(f['/field_0']).flatten()

    UiUjUk_vec = np.zeros((Npts, 10))
    for i in range(1, 11):
        tmp = str(11 + i)
        fnum = ('000000'[:5 - len(tmp)] + tmp)
        filename = path_to_files+'/'+f'interpolated_fields{fnum}.hdf5'
        with h5py.File(filename, 'r') as f:
            UiUjUk_vec[:, i-1] = np.array(f['/field_0']).flatten()

    # Reordering columns based on MATLAB's conversion
    UiUjUk_vec = UiUjUk_vec[:, [0, 1, 2, 3, 4, 5, 7, 8, 9, 6]]

    Ui4_vec = np.zeros((Npts, 3))
    for i in range(1, 4):
        tmp = str(21 + i)
        fnum = ('000000'[:5 - len(tmp)] + tmp)
        filename = path_to_files+'/'+f'interpolated_fields{fnum}.hdf5'
        with h5py.File(filename, 'r') as f:
            Ui4_vec[:, i-1] = np.array(f['/field_0']).flatten()

    with h5py.File(path_to_files+'/'+'interpolated_fields00025.hdf5', 'r') as f:
        P3_vec = np.array(f['/field_0']).flatten()

    with h5py.File(path_to_files+'/'+'interpolated_fields00026.hdf5', 'r') as f:
        P4_vec = np.array(f['/field_0']).flatten()

    PUi_vec = np.zeros((Npts, 3))
    for i in range(1, 4):
        tmp = str(26 + i)
        fnum = ('000000'[:5 - len(tmp)] + tmp)
        filename = path_to_files+'/'+f'interpolated_fields{fnum}.hdf5'
        with h5py.File(filename, 'r') as f:
            PUi_vec[:, i-1] = np.array(f['/field_0']).flatten()

    PGij_vec = np.zeros((Npts, 9))
    for i in range(1, 10):
        tmp = str(29 + i)
        fnum = ('000000'[:5 - len(tmp)] + tmp)
        filename = path_to_files+'/'+f'interpolated_fields{fnum}.hdf5'
        with h5py.File(filename, 'r') as f:
            PGij_vec[:, i-1] = np.array(f['/field_0']).flatten()

    pseudoDiss_vec = np.zeros((Npts, 6))
    for i in range(1, 7):
        tmp = str(38 + i)
        fnum = ('000000'[:5 - len(tmp)] + tmp)
        filename = path_to_files+'/'+f'interpolated_fields{fnum}.hdf5'
        with h5py.File(filename, 'r') as f:
            pseudoDiss_vec[:, i-1] = np.array(f['/field_0']).flatten()
    
    print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%% make 1D arrays 2D
    P_vec  = P_vec[:, np.newaxis]
    P2_vec = P2_vec[:, np.newaxis]
    P3_vec = P3_vec[:, np.newaxis]
    P4_vec = P4_vec[:, np.newaxis]

    #%%
    print('reading the additional 99 fields...')
    start_time = time.time()

    def read_hdf5_field(filename):
        with h5py.File(filename, 'r') as f:
            return np.array(f['/field_0']).flatten()

    def generate_filenames(start, count):
        return [path_to_files+'/'+f'interpolated_fields{str(start + i).zfill(5)}.hdf5' for i in range(1, count + 1)]

    dUidxj_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(44, 9)])
    d2Uidx2_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(53, 9)])
    dPdx_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(62, 3)])
    d2Pdx2_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(65, 3)])
    dPUidxj_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(68, 9)])
    dUiUjdx_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(77, 3)] +
                                [read_hdf5_field(f) for f in generate_filenames(83, 3)] +
                                [read_hdf5_field(f) for f in generate_filenames(89, 3)] +
                                [read_hdf5_field(f) for f in generate_filenames(95, 3)] +
                                [read_hdf5_field(f) for f in generate_filenames(101, 3)] +
                                [read_hdf5_field(f) for f in generate_filenames(107, 3)])
    d2UiUjdx2_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(80, 3)] +
                                    [read_hdf5_field(f) for f in generate_filenames(86, 3)] +
                                    [read_hdf5_field(f) for f in generate_filenames(92, 3)] +
                                    [read_hdf5_field(f) for f in generate_filenames(98, 3)] +
                                    [read_hdf5_field(f) for f in generate_filenames(104, 3)] +
                                    [read_hdf5_field(f) for f in generate_filenames(110, 3)])
    dUiUjUkdx_vec = np.column_stack([read_hdf5_field(f) for f in generate_filenames(113, 30)])

    # Reordering columns based on MATLAB's conversion
    dUiUjUkdx_vec = dUiUjUkdx_vec[:, list(range(18)) + list(range(21, 30)) + list(range(18, 21))]

    print('Finished reading all fields.')
    print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%% Convert arrays to single precision if required
    if If_convert_to_single:
        print('Converting arrays into single precision...')
        start_time = time.time()
        for name in list(locals().keys()):
            if name.endswith('_vec') and isinstance(locals()[name], np.ndarray):
                # print(np.shape(locals()[name]))
                locals()[name] = locals()[name].astype(np.float32)
        print('Conversion complete.')
        print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%% Reshape arrays based on Nstruct
    print('Reshaping into arrays...')
    start_time = time.time()
    for name in list(locals().keys()):
        if name.endswith('_vec') and isinstance(locals()[name], np.ndarray):
            reshaped_name = name[:-4] + '_struct'
            locals()[reshaped_name] = locals()[name].reshape((Nstruct[1], Nstruct[0], Nstruct[2], locals()[name].shape[1]),order="F")
            del locals()[name]
    print('Reshaping complete.')
    print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%% Permute arrays to original shape
    print('Permuting arrays into the original shape...')
    start_time = time.time()
    for name in list(locals().keys()):
        if name.endswith('_struct') and isinstance(locals()[name], np.ndarray):
            locals()[name] = np.transpose(locals()[name], (1, 0, 2, 3))
    print('Permutation complete.')
    print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%% Apply user-specified averaging function if required
    if If_average:
        print('Taking the user-specified average using function av_func...')
        start_time = time.time()
        for name in list(locals().keys()):
            if name.endswith('_struct') and isinstance(locals()[name], np.ndarray):
                locals()[name] = av_func(locals()[name])
        print('Averaging complete.')
        print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%% Reynolds number needed to calculate viscous related terms later
    Rer_here    = Reynolds_number

    #%% Save the data in HDF5 format
    print('Saving the data in HDF5 format...')
    start_time = time.time()
    with h5py.File(path_to_files+'/'+output_fname, 'w') as hf:
        global_vars = dict(locals())  # Create a copy to avoid modification issues
        for name, data in global_vars.items():
            if (name.endswith('_struct') or name.endswith('_here')) and isinstance(data, (np.ndarray, int, float)):
                hf.create_dataset(name, data=data)
    print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%%
    print('Data saved successfully in HDF5 format.')



#%%
#####################################################################################
#####################################################################################
#####################################################################################
# function to read the raw but averaged fields and calcuate the budgets in Cartesian coordinates
#####################################################################################
def calculate_budgets_in_Cartesian(
        path_to_files = './' ,
        input_filename  = "averaged_and_renamed_interpolated_fields.hdf5",
        output_filename = "pstat3d_format.hdf5" ):
    
    #%%
    import h5py
    import numpy as np
    import time

    #%%
    with h5py.File(path_to_files+'/'+input_filename, "r") as input_file, \
         h5py.File(path_to_files+'/'+output_filename, "w") as output_file:
        
        #%%
        Rer_here = np.float64(input_file['Rer_here'])
        print('Reynolds number = ', Rer_here)

        output_file.create_dataset('Rer_here', data=Rer_here, compression=None)

        #%% XYZ coordinates
        print('--------------working on XYZ coordinates...')
        start_time = time.time()

        XYZ = np.array(input_file['XYZ_struct'])
        output_file.create_dataset('XYZ', data=XYZ, compression=None)
        Nxyz = XYZ.shape[:3]
        print('Number of points: ' , Nxyz)

        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%%
        # Mean Velocities
        print('--------------working on mean velocities...')
        start_time = time.time()

        UVW = np.array(input_file['UVW_struct'])
        output_file.create_dataset('UVW', data=UVW, compression=None)

        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%% Reynolds Stresses
        print('--------------working on Reynolds stresses...')
        start_time = time.time()

        Rij = np.array(input_file['UiUj_struct'])
        Rij = Rij - (UVW[..., [0, 1, 2, 0, 0, 1]] * UVW[..., [0, 1, 2, 1, 2, 2]])

        output_file.create_dataset('Rij', data=Rij, compression=None)

        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%% Pressure and Its Moments
        print('--------------working on pressure and its moments...')
        start_time = time.time()

        P  = np.array(input_file['P_struct'])
        P2 = np.array(input_file['P2_struct'])
        P3 = np.array(input_file['P3_struct'])
        P4 = np.array(input_file['P4_struct'])

        P2 = P2 - P**2
        P3 = P3 - P**3 - 3 * P * P2
        P4 = P4 - P**4 - 6 * P**2 * P2 - 4 * P * P3

        output_file.create_dataset('P', data=P, compression=None)
        output_file.create_dataset('P2', data=P2, compression=None)
        output_file.create_dataset('P3', data=P3, compression=None)
        output_file.create_dataset('P4', data=P4, compression=None)

        del P2, P3, P4
        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%%
        print('--------------working on tripple products...')
        start_time = time.time()
        
        # Load relevant datasets from the HDF5 file
        UiUjUk = np.array(input_file['UiUjUk_struct'][:])
        
        # Perform calculations similar to Matlab
        UiUjUk = UiUjUk - (
            UVW[:, :, :, [0, 1, 2, 0, 0, 0, 1, 0, 1, 0]] *
            UVW[:, :, :, [0, 1, 2, 0, 0, 1, 1, 2, 2, 1]] *
            UVW[:, :, :, [0, 1, 2, 1, 2, 1, 2, 2, 2, 2]]
        ) - (
            UVW[:, :, :, [0, 1, 2, 0, 0, 0, 1, 0, 1, 0]] *
            Rij[:, :, :, [0, 1, 2, 3, 4, 1, 5, 2, 2, 5]]
        ) - (
            UVW[:, :, :, [0, 1, 2, 0, 0, 1, 1, 2, 2, 1]] *
            Rij[:, :, :, [0, 1, 2, 3, 4, 3, 5, 4, 5, 4]]
        ) - (
            UVW[:, :, :, [0, 1, 2, 1, 2, 1, 2, 2, 2, 2]] *
            Rij[:, :, :, [0, 1, 2, 0, 0, 3, 1, 4, 5, 3]]
        )
        
        # Save result into HDF5 file
        output_file.create_dataset('UiUjUk', data=UiUjUk)
        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%%
        print('--------------working on velocity kurtosis...')
        start_time = time.time()

        # Load Ui4 from the input file
        Ui4 = np.array(input_file['Ui4_struct'][:])
        
        # Perform kurtosis calculation
        Ui4 = Ui4 - UVW**4 - 6 * UVW**2 * Rij[:, :, :, 0:3] - 4 * UVW * UiUjUk[:, :, :, 0:3]
        
        # Save result into HDF5 file
        output_file.create_dataset('Ui4', data=Ui4)
        del UiUjUk, Ui4
        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%%
        print("--------------working on velocity gradients...")
        start_time = time.time()
        
        # Read velocity gradient tensor from input file
        dUidXj = np.array(input_file["dUidxj_struct"][()])
        
        # Save to output file
        output_file.create_dataset("dUidXj", data=dUidXj)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        #%%
        print("--------------working on momentum convection terms...")
        start_time = time.time()

        
        # Reshape dUidXj
        dUidXj_reshaped = dUidXj.reshape(Nxyz[0], Nxyz[1], Nxyz[2], 3, 3, order="F")

        # Compute Momentum convection
        Momentum_convection = np.sum(UVW[..., np.newaxis] * dUidXj_reshaped, axis=3)

        # Save to output file
        output_file.create_dataset("Momentum_convection", data=Momentum_convection)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        #%%
        print("--------------working on the residual momentum convection terms...")
        start_time = time.time()

        # Compute Residual Momentum convection terms
        Momentum_convectionRes = np.zeros((*Nxyz, 3))  # Shape: (N1, N2, N3, 3)
        Momentum_convectionRes[..., 0] = np.sum(UVW[..., [1, 2]] * dUidXj[..., [1, 2]], axis=-1)
        Momentum_convectionRes[..., 1] = np.sum(UVW[..., [0, 2]] * dUidXj[..., [3, 5]], axis=-1)
        Momentum_convectionRes[..., 2] = np.sum(UVW[..., [0, 1]] * dUidXj[..., [6, 7]], axis=-1)

        # Save to output file
        output_file.create_dataset("Momentum_convectionRes", data=Momentum_convectionRes)
        del Momentum_convectionRes
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

         #%%
        print("--------------working on momentum pressure terms...")
        print("Warning: assuming rho=1 everywhere!")
        start_time = time.time()

        # Compute Momentum pressure
        Momentum_pressure = -np.array(input_file["dPdx_struct"][()])

        # Save to output file
        output_file.create_dataset("Momentum_pressure", data=Momentum_pressure)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        #%%
        print('--------------working on production terms...')
        start_time = time.time()

        Prod_ij = np.zeros((*dUidXj.shape[:3], 6))

        Prod_ij[..., 0] = -2 * np.sum(Rij[..., [0, 3, 4]] * dUidXj[..., :3], axis=3)
        Prod_ij[..., 1] = -2 * np.sum(Rij[..., [3, 1, 5]] * dUidXj[..., 3:6], axis=3)
        Prod_ij[..., 2] = -2 * np.sum(Rij[..., [4, 5, 2]] * dUidXj[..., 6:9], axis=3)
        Prod_ij[..., 3] = -np.sum(Rij[..., [0, 3, 4]] * dUidXj[..., 3:6] +
                                Rij[..., [3, 1, 5]] * dUidXj[..., :3], axis=3)
        Prod_ij[..., 4] = -np.sum(Rij[..., [0, 3, 4]] * dUidXj[..., 6:9] +
                                Rij[..., [4, 5, 2]] * dUidXj[..., :3], axis=3)
        Prod_ij[..., 5] = -np.sum(Rij[..., [3, 1, 5]] * dUidXj[..., 6:9] +
                                Rij[..., [4, 5, 2]] * dUidXj[..., 3:6], axis=3)

        TKE_prod = np.sum(Prod_ij[..., :3], axis=3) / 2

        output_file.create_dataset('Prod_ij', data=Prod_ij)
        output_file.create_dataset('TKE_prod', data=TKE_prod)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        #%%
        print('--------------working on convection terms...')
        start_time = time.time()

        dRij_dxk = np.array(input_file['dUiUjdx_struct'][:])

        dRij_dxk = dRij_dxk - \
                UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1]] * \
                dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]] - \
                UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]] * \
                dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5]]

        Conv_ij = np.zeros((*UVW.shape[:3], 6))

        Conv_ij[..., 0] = np.sum(UVW[..., :3] * dRij_dxk[..., :3], axis=3)
        Conv_ij[..., 1] = np.sum(UVW[..., :3] * dRij_dxk[..., 3:6], axis=3)
        Conv_ij[..., 2] = np.sum(UVW[..., :3] * dRij_dxk[..., 6:9], axis=3)
        Conv_ij[..., 3] = np.sum(UVW[..., :3] * dRij_dxk[..., 9:12], axis=3)
        Conv_ij[..., 4] = np.sum(UVW[..., :3] * dRij_dxk[..., 12:15], axis=3)
        Conv_ij[..., 5] = np.sum(UVW[..., :3] * dRij_dxk[..., 15:18], axis=3)

        Conv_ij = -Conv_ij
        TKE_conv = np.sum(Conv_ij[..., :3], axis=3) / 2

        output_file.create_dataset('Conv_ij', data=Conv_ij)
        output_file.create_dataset('TKE_conv', data=TKE_conv)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        #%%
        print('--------------working on momentum turbulent diffusion terms...')
        start_time = time.time()

        Momentum_turb_diffusion = np.zeros((*dRij_dxk.shape[:3], 3))

        Momentum_turb_diffusion[..., 0] = -np.sum(dRij_dxk[..., [0, 10, 14]], axis=3)
        Momentum_turb_diffusion[..., 1] = -np.sum(dRij_dxk[..., [9, 4, 17]], axis=3)
        Momentum_turb_diffusion[..., 2] = -np.sum(dRij_dxk[..., [12, 16, 8]], axis=3)

        output_file.create_dataset('Momentum_turb_diffusion', data=Momentum_turb_diffusion)
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        #%%
        print('--------------working on dissipation terms...')
        start_time = time.time()
        Diss_ij = np.array(input_file['pseudoDiss_struct'][()])

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
        output_file.create_dataset('Diss_ij', data=Diss_ij, compression=None)
        output_file.create_dataset('TKE_diss', data=TKE_diss, compression=None)
        del Diss_ij
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        #%%
        print('--------------working on turbulent transport...')
        start_time = time.time()

        # Define index mappings
        ind_dRjkdxk = np.array([
            (np.array([1, 4, 5]) - 1) * 3 + np.array([1, 2, 3])-1,
            (np.array([4, 2, 6]) - 1) * 3 + np.array([1, 2, 3])-1,
            (np.array([5, 6, 3]) - 1) * 3 + np.array([1, 2, 3])-1
        ])

        ind_tripple = np.zeros((3, 3, 3), dtype=int)
        ind_tripple[:, 0, 0] = (np.array([1, 4, 5]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 0, 1] = (np.array([4, 6, 10]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 0, 2] = (np.array([5, 10, 8]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 1, 0] = (np.array([4, 6, 10]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 1, 1] = (np.array([6, 2, 7]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 1, 2] = (np.array([10, 7, 9]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 2, 0] = (np.array([5, 10, 8]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 2, 1] = (np.array([10, 7, 9]) - 1) * 3 + np.array([1, 2, 3])-1
        ind_tripple[:, 2, 2] = (np.array([8, 9, 3]) - 1) * 3 + np.array([1, 2, 3])-1

        # Allocate memory
        TurbTrans_ij = np.zeros((*Nxyz, 3, 3), dtype=np.float32)

        # Compute turbulent transport
        for i in range(3):
            for j in range(3):
                TurbTrans_ij[..., i, j] = (
                    - np.array(input_file['dUiUjUkdx_struct'][..., ind_tripple[0, i, j]])
                    - np.array(input_file['dUiUjUkdx_struct'][..., ind_tripple[1, i, j]])
                    - np.array(input_file['dUiUjUkdx_struct'][..., ind_tripple[2, i, j]])
                )

                TurbTrans_ij[..., i, j] += (
                    UVW[..., i] * np.sum(UVW * dUidXj[..., (np.arange(3) + j * 3)], axis=-1)
                    + UVW[..., j] * np.sum(UVW * dUidXj[..., (np.arange(3) + i * 3)], axis=-1)
                    + UVW[..., i] * np.sum(dRij_dxk[..., ind_dRjkdxk[j, :]], axis=-1)
                    + UVW[..., j] * np.sum(dRij_dxk[..., ind_dRjkdxk[i, :]], axis=-1)
                )

        # Reorder indices
        TurbTrans_ij = TurbTrans_ij.reshape(TurbTrans_ij.shape[:-2] + (-1,) , order="F")
        TurbTrans_ij = TurbTrans_ij[..., [0, 4, 8, 1, 2, 5]]

        # Final computation
        TurbTrans_ij -= Conv_ij + Prod_ij
        TKE_turbTrans = np.sum(TurbTrans_ij[..., 0:3], axis=-1) / 2

        # Save results
        output_file.create_dataset('TurbTrans_ij', data=TurbTrans_ij, compression=None)
        output_file.create_dataset('TKE_turbTrans', data=TKE_turbTrans, compression=None)

        # Cleanup
        del TurbTrans_ij, Conv_ij, Prod_ij, dRij_dxk

        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%%
        print('--------------working on viscous diffusion...')
        start_time = time.time()

        # Read datasets
        d2Ui_dx2 = np.array(input_file['d2Uidx2_struct'])
        d2Rij_dx2 = np.array(input_file['d2UiUjdx2_struct'])

        # Compute d2Rij_dx2
        d2Rij_dx2 = (
            d2Rij_dx2
            - UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
            * d2Ui_dx2[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]]
            - UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2]]
            * d2Ui_dx2[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5]]
            - 2 * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 0, 1, 2, 3, 4, 5]]
                * dUidXj[..., [0, 1, 2, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 6, 7, 8]]
        )

        # Compute viscous diffusion
        ViscDiff_ij = (1 / Rer_here) * np.sum(d2Rij_dx2.reshape((*Nxyz, 3, -1), order="F"), axis=3)

        # Compute TKE viscous diffusion
        TKE_ViscDiff = np.sum(ViscDiff_ij[..., 0:3], axis=-1) / 2

        # Save results
        output_file.create_dataset('ViscDiff_ij', data=ViscDiff_ij, compression=None)
        output_file.create_dataset('TKE_ViscDiff', data=TKE_ViscDiff, compression=None)

        # Cleanup
        del d2Rij_dx2, ViscDiff_ij

        print(f'Done in {time.time() - start_time:.2f} seconds.')

         #%% Momentum viscous diffusion terms
        print('--------------working on momentum viscous diffusion terms...')
        start_time = time.time()

        Momentum_viscous_diffusion = (1 / Rer_here) * \
            np.sum(d2Ui_dx2.reshape(*Nxyz, 3, 3, order="F"), axis=3)

        output_file.create_dataset('Momentum_viscous_diffusion', data=Momentum_viscous_diffusion, compression=None)

        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%% Momentum residual terms
        print('--------------working on momentum residual terms...')
        start_time = time.time()

        Momentum_residual = (
              Momentum_viscous_diffusion
            + Momentum_turb_diffusion
            + Momentum_pressure
            - Momentum_convection
        )

        output_file.create_dataset('Momentum_residual', data=Momentum_residual, compression=None)

        # Cleanup
        del Momentum_viscous_diffusion, Momentum_turb_diffusion, \
            Momentum_pressure, Momentum_convection, Momentum_residual

        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%% Pressure-rate-of-strain terms
        print('--------------working on pressure-rate-of-strain terms...')
        start_time = time.time()

        PRS_ij = np.array(input_file['PGij_struct'])
        PRS_ij = PRS_ij - P * dUidXj
        PRS_ij = PRS_ij[..., [0, 4, 8, 1, 2, 5]] + PRS_ij[..., [0, 4, 8, 3, 6, 7]]

        output_file.create_dataset('PRS_ij', data=PRS_ij, compression=None)

        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%%
        print('--------------working on pressure transport terms...')
        start_time = time.time()

        dpdx = np.array(input_file['dPdx_struct'])
        dpudx = np.array(input_file['dPUidxj_struct'])

        dpudx = dpudx - (P * dUidXj) \
            - (UVW[..., [0, 0, 0, 1, 1, 1, 2, 2, 2]] * dpdx[..., [0, 1, 2, 0, 1, 2, 0, 1, 2]])

        PressTrans_ij = -(dpudx[..., [0, 4, 8, 1, 2, 5]] + dpudx[..., [0, 4, 8, 3, 6, 7]])

        output_file.create_dataset('dpdx', data=dpdx, compression=None)
        output_file.create_dataset('PressTrans_ij', data=PressTrans_ij, compression=None)

        del dpudx
        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%% Velocity-Pressure Gradient Terms
        print('--------------working on velocity-pressure-gradient terms...')
        start_time = time.time()

        VelPressGrad_ij = PressTrans_ij - PRS_ij
        TKE_VelPressGrad = np.sum(VelPressGrad_ij[..., 0:3], axis=-1) / 2

        output_file.create_dataset('VelPressGrad_ij', data=VelPressGrad_ij, compression=None)
        output_file.create_dataset('TKE_VelPressGrad', data=TKE_VelPressGrad, compression=None)

        del PRS_ij, PressTrans_ij, VelPressGrad_ij
        print(f'Done in {time.time() - start_time:.2f} seconds.')

        #%% TKE budget residual
        print('--------------working on TKE budgets...')
        start_time = time.time()

        TKE_residual = (
            TKE_prod + TKE_diss + TKE_turbTrans + TKE_ViscDiff + TKE_VelPressGrad + TKE_conv
        )

        output_file.create_dataset('TKE_residual', data=TKE_residual, compression=None)

        del TKE_prod, TKE_diss, TKE_turbTrans, TKE_ViscDiff, \
            TKE_VelPressGrad, TKE_conv, TKE_residual
        print(f'Done in {time.time() - start_time:.2f} seconds.')

    #%%
    print("All computations completed and data saved successfully.")
