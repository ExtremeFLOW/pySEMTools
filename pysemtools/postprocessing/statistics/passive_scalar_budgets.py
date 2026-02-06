###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
## Preliminary functions to make life easier.
###########################################################################################
###########################################################################################
#%% generic function to compute the gradient of a scalar field
def compute_scalar_first_derivative(comm, msh, coef, scalar, scalar_deriv):
    if msh.gdim == 3:
        scalar_deriv.c1 = coef.dudxyz(scalar, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv.c2 = coef.dudxyz(scalar, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv.c3 = coef.dudxyz(scalar, coef.drdz, coef.dsdz, coef.dtdz)
    elif msh.gdim == 2:
        scalar_deriv.c1 = coef.dudxyz(scalar, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv.c2 = coef.dudxyz(scalar, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv.c3 = 0.0 * scalar_deriv.c2
    else:
        import sys
        sys.exit("supports either 2D or 3D data")
###########################################################################################
###########################################################################################


#%% generic function to compute the diagonal second derivatives of a scalar field from its gradient
def compute_scalar_second_derivative(comm, msh, coef, scalar_deriv, scalar_deriv2):
    if msh.gdim == 3:
        scalar_deriv2.c1 = coef.dudxyz(scalar_deriv.c1, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv2.c2 = coef.dudxyz(scalar_deriv.c2, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv2.c3 = coef.dudxyz(scalar_deriv.c3, coef.drdz, coef.dsdz, coef.dtdz)
    elif msh.gdim == 2:
        scalar_deriv2.c1 = coef.dudxyz(scalar_deriv.c1, coef.drdx, coef.dsdx, coef.dtdx)
        scalar_deriv2.c2 = coef.dudxyz(scalar_deriv.c2, coef.drdy, coef.dsdy, coef.dtdy)
        scalar_deriv2.c3 = 0.0 * scalar_deriv2.c3
    else:
        import sys
        sys.exit("supports either 2D or 3D data")
###########################################################################################
###########################################################################################


#%% generic function to write a 9 component field with input as 3 vectors of 3 components each
def write_file_9c(comm, msh, dU_dxi, dV_dxi, dW_dxi, fname_gradU, if_write_mesh):
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.io.ppymech.neksuite import pynekwrite
    import numpy as np

    gradU = FieldRegistry(comm)

    gradU.add_field(comm, field_name="c1", field=dU_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c2", field=dU_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c3", field=dU_dxi.c3, dtype=np.single)
    gradU.add_field(comm, field_name="c4", field=dV_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c5", field=dV_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c6", field=dV_dxi.c3, dtype=np.single)
    gradU.add_field(comm, field_name="c7", field=dW_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c8", field=dW_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c9", field=dW_dxi.c3, dtype=np.single)

    pynekwrite(fname_gradU, comm, msh=msh, fld=gradU, wdsz=4, write_mesh=if_write_mesh)

    gradU.clear()
###########################################################################################
###########################################################################################


#%% generic function to write a 6-component field with input as 2 vectors of 3 components each
def write_file_6c(comm, msh, dU_dxi, dV_dxi, fname_gradU, if_write_mesh):
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.io.ppymech.neksuite import pynekwrite
    import numpy as np

    gradU = FieldRegistry(comm)

    gradU.add_field(comm, field_name="c1", field=dU_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c2", field=dU_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c3", field=dU_dxi.c3, dtype=np.single)
    gradU.add_field(comm, field_name="c4", field=dV_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c5", field=dV_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c6", field=dV_dxi.c3, dtype=np.single)

    pynekwrite(fname_gradU, comm, msh=msh, fld=gradU, wdsz=4, write_mesh=if_write_mesh)

    gradU.clear()
###########################################################################################
###########################################################################################


#%% generic function to write a 3-component (vector) field
def write_file_3c(comm, msh, dU_dxi, fname_gradU, if_write_mesh):
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.io.ppymech.neksuite import pynekwrite
    import numpy as np

    gradU = FieldRegistry(comm)

    gradU.add_field(comm, field_name="c1", field=dU_dxi.c1, dtype=np.single)
    gradU.add_field(comm, field_name="c2", field=dU_dxi.c2, dtype=np.single)
    gradU.add_field(comm, field_name="c3", field=dU_dxi.c3, dtype=np.single)

    pynekwrite(fname_gradU, comm, msh=msh, fld=gradU, wdsz=4, write_mesh=if_write_mesh)

    gradU.clear()
###########################################################################################
###########################################################################################



#%% function to generate the list of fields from the file header
def return_list_of_vars_from_filename(fname):
    from pymech.neksuite.field import read_header

    header = read_header(fname)
    vars_=  header.nb_vars

    vel_fields = vars_[1]
    pres_fields = vars_[2]
    temp_fields = vars_[3]
    scal_fields = vars_[4]
    if scal_fields>37:
        import warnings
        print("Number of scalar fields: "+(f'{(scal_fields):.0f}'))
        warnings.warn("The number of scalar fields above 39 is not supported. "+
                      "This was done to make converted 2D statistics files consistent! "+
                      "Limiting the number to 37...")
        scal_fields = 37
        

    field_names = []
    for i in range(vel_fields):
        tmp = [("vel_"+str(i))]
        field_names = field_names + tmp

    if pres_fields==1:
        field_names = field_names + ["pres"]

    if temp_fields==1:
        field_names = field_names + ["temp"]

    for i in range(scal_fields):
        tmp = [("scal_"+str(i))]
        field_names = field_names + tmp

    return field_names
###########################################################################################
###########################################################################################

#%% do dssum on a vector with components c1, c2, c3
def do_dssum_on_3comp_vector(dU_dxi, msh_conn, msh):
    msh_conn.dssum(field=dU_dxi.c1, msh=msh, average="multiplicity")
    msh_conn.dssum(field=dU_dxi.c2, msh=msh, average="multiplicity")
    msh_conn.dssum(field=dU_dxi.c3, msh=msh, average="multiplicity")
# def do_dssum_on_3comp_vector(dU_dxi, coef, msh):
#     coef.dssum(dU_dxi.c1, msh)
#     coef.dssum(dU_dxi.c2, msh)
#     coef.dssum(dU_dxi.c3, msh)
###########################################################################################
###########################################################################################



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
## generic function to compute the additional fields required for budget terms, etc.
###########################################################################################
###########################################################################################
def compute_and_write_additional_sstat_fields(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    if_write_mesh=False,
    which_code="NEKO",
    if_do_dssum_on_derivatives=False,
):

    ###########################################################################################
    # do some initial checks
    import sys

    # check if file names are the same
    if fname_mean != fname_stat:
        sys.exit(
            "fname_mean must be the same as fname_stat"
        )

    # Scalar statistics only implemented for Neko
    if which_code.casefold() != "NEKO":
        sys.exit("only NEKO is supported for passive scalar statistics")

    ###########################################################################################
    import warnings
    from mpi4py import MPI  # equivalent to the use of MPI_init() in C
    import numpy as np

    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.coef import Coef
    from pysemtools.io.ppymech.neksuite import pynekread

    if if_do_dssum_on_derivatives:
        from pysemtools.datatypes.msh_connectivity import MeshConnectivity

    ###########################################################################################
    # Get mpi info
    comm = MPI.COMM_WORLD

    ###########################################################################################
    # intialize the mesh and some fields
    msh = Mesh(comm, create_connectivity=False)
    # print('mesh dimension: ', msh.gdim )
    # else:
    #     msh = Mesh(comm, create_connectivity=False)

    stat_fields = FieldRegistry(comm)

    # generic scalar and vector derivatives
    dQ1_dxi  = FieldRegistry(comm)
    dQ2_dxi  = FieldRegistry(comm)
    dQ3_dxi  = FieldRegistry(comm)
    d2Q1_dxi2 = FieldRegistry(comm)


    ###########################################################################################
    # using the same .fXXXXXX extenstion as the mean fields
    this_ext = fname_mean[-8:]
    this_ext_check = fname_stat[-8:]

    # check the two files match. can be replaced by an error, but that seems too harsh and limiting
    if this_ext != this_ext_check:
        warnings.warn(
            "File index of fname_stat and fname_mean differ! Hope you know what you are doing!"
        )

    full_fname_mesh = which_dir + "/" + fname_mesh
    full_fname_stat = which_dir + "/" + fname_stat

    ###########################################################################################
    # read mesh and compute coefs
    pynekread(full_fname_mesh, comm, msh=msh, data_dtype=np.single)
    if if_do_dssum_on_derivatives:
        msh_conn = MeshConnectivity(comm, msh, rel_tol=1e-5)

    if msh.gdim < 3:
        sys.exit("only 3D data is supported at the moment! ")

    coef = Coef(msh, comm, get_area=False)

    ###########################################################################################
    # define file_keys of the fields based on the codes

    # Neko key names taken from: https://neko.cfd/docs/develop/df/d8f/statistics-guide.html
    file_keys_S     = ["pres"]
    #                   "US"      "VS"      "WS"
    file_keys_UiS   = ["vel_0", "vel_1", "vel_2"]
    file_keys_SS    = ["temp"]
    #                   "USS"      "VSS"     "WSS"
    file_keys_UjSS  = ["scal_2", "scal_3", "scal_4"]
    #                   "UUS"     "VVS"     "WWS"     "UVS"     "UWS"     "VWS"
    file_keys_UiUjS = ["scal_5", "scal_6", "scal_7", "scal_8", "scal_9", "scal_10"]
    file_keys_PS    = ["scal_11"]



    ###########################################################################################
    # S first and second derivatives
    ###########################################################################################
    
    stat_fields.add_field(
        comm,
        field_name="S",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_S,
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["S"], dQ1_dxi
    )
    compute_scalar_second_derivative(comm, msh, coef, dQ1_dxi, d2Q1_dxi2)

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(d2Q1_dxi2, msh_conn, msh)
    
    this_file_name = which_dir + "/dnSdxn" + this_ext
    write_file_6c(
        comm, msh, dQ1_dxi, d2Q1_dxi2, this_file_name, if_write_mesh=if_write_mesh
    )

    stat_fields.clear()

    ###########################################################################################
    # SS first and second derivatives
    ###########################################################################################
    
    stat_fields.add_field(
        comm,
        field_name="SS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SS,
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SS"], dQ1_dxi
    )
    compute_scalar_second_derivative(comm, msh, coef, dQ1_dxi, d2Q1_dxi2)

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(d2Q1_dxi2, msh_conn, msh)

    this_file_name = which_dir + "/dnSSdxn" + this_ext
    write_file_6c(
        comm, msh, dQ1_dxi, d2Q1_dxi2, this_file_name, if_write_mesh=if_write_mesh
    )

    stat_fields.clear()
    del d2Q1_dxi2

    ###########################################################################################
    # UiS first derivative
    ###########################################################################################

    stat_fields.add_field(
        comm,
        field_name="US",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiS[0],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="VS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiS[1],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="WS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiS[2],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["US"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VS"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["WS"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dUSdx" + this_ext
    write_file_9c(
        comm, msh, dQ1_dxi, dQ2_dxi, dQ3_dxi, 
        this_file_name, if_write_mesh=if_write_mesh
    )

    stat_fields.clear()
    
    ###########################################################################################
    # UiSS first derivative
    ###########################################################################################
    
    stat_fields.add_field(
        comm,
        field_name="USS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiSS[0],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="VSS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiS[1],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="WSS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiSS[2],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["USS"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VSS"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["WSS"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dUSSdx" + this_ext
    write_file_9c(
        comm, msh, dQ1_dxi, dQ2_dxi, dQ3_dxi, 
        this_file_name, if_write_mesh=if_write_mesh
    )

    stat_fields.clear()
    del dQ2_dxi, dQ3_dxi

    ###########################################################################################
    # UiUjS first derivative
    ###########################################################################################

    actual_field_names = ["UUS", "VVS", "WWS", "UVS", "UWS", "VWS"]

    for icomp in range(0, 6):
        if comm.Get_rank() == 0:
            print("working on: " + actual_field_names[icomp])

        stat_fields.add_field(
            comm,
            field_name=actual_field_names[icomp],
            file_type="fld",
            file_name=full_fname_stat,
            file_key=file_keys_UiUjS[icomp],
            dtype=np.single,
        )

        compute_scalar_first_derivative(
            comm, msh, coef, stat_fields.registry[actual_field_names[icomp]], dQ1_dxi
        )

        if if_do_dssum_on_derivatives:
            do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)

        this_file_name = (
            which_dir + "/d" + actual_field_names[icomp] + "dx" + this_ext
        )
        write_file_3c(
            comm, msh, dQ1_dxi, this_file_name, if_write_mesh=if_write_mesh
        )

        stat_fields.clear()
        
    ###########################################################################################
    # PS first derivative
    ###########################################################################################
    
    stat_fields.add_field(
        comm,
        field_name="PS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_PS,
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["PS"], dQ1_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
    this_file_name = which_dir + "/dPSdx" + this_ext
    write_file_3c(
        comm, msh, dQ1_dxi, this_file_name, if_write_mesh=if_write_mesh
    )

    stat_fields.clear()
    del dQ1_dxi

    if comm.Get_rank() == 0:
        print("-------As a great man once said: run successful: dying ...")


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# interpolate the 42+N fields onto the user specified set of points
###########################################################################################
###########################################################################################
def interpolate_all_stat_and_sstat_fields_onto_points(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    xyz,
    which_code="NEKO",
    if_do_dssum_before_interp=True,
    if_create_boundingBox_for_interp=False,
    if_pass_points_to_rank0_only=True,
    interpolation_output_fname="interpolated_scalar_fields.hdf5",
    find_points_tol=None
):

    from mpi4py import MPI  # equivalent to the use of MPI_init() in C
    import numpy as np
    from scipy.io import savemat

    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.coef import Coef
    from pysemtools.io.ppymech.neksuite import pynekread
    from pysemtools.interpolation.probes import Probes

    if if_do_dssum_before_interp:
        from pysemtools.datatypes.msh_connectivity import MeshConnectivity

    if if_create_boundingBox_for_interp:
        from pysemtools.datatypes.msh_partitioning import MeshPartitioner

    ###########################################################################################
    # do some initial checks
    import sys

    # check if file names are the same
    if fname_mean != fname_stat:
        sys.exit(
            "fname_mean must be the same as fname_stat"
        )

    # Scalar statistics only implemented for Neko
    if which_code.casefold() != "NEKO":
        sys.exit("only NEKO is supported for passive scalar statistics")

    ###########################################################################################
    # Get mpi info
    comm = MPI.COMM_WORLD

    ###########################################################################################
    # intialize the mesh and some fields
    msh = Mesh(comm, create_connectivity=False)
    mean_fields = FieldRegistry(comm)

    ###########################################################################################
    # using the same .fXXXXXX extenstion as the mean fields
    this_ext = fname_mean[-8:]
    this_ext_check = fname_stat[-8:]

    # check the two files match. can be replaced by an error, but that seems too harsh and limiting
    if this_ext != this_ext_check:
        warnings.warn(
            "File index of fname_stat and fname_mean differ! Hope you know what you are doing!"
        )

    full_fname_mesh = which_dir + "/" + fname_mesh

    # get the file name for the 42 fileds collected in run time
    these_names = which_dir + "/" + fname_stat

    # add the name of the additional fields
    these_names.extend([which_dir + "/dnSdxn" + this_ext,
                        which_dir + "/dnSSdxn" + this_ext,
                        which_dir + "/dUSdx" + this_ext,
                        which_dir + "/dUSSdx" + this_ext])

    actual_field_names = ["UUS", "VVS", "WWS", "UVS", "UWS", "VWS"]
    for icomp in range(0, 6):
        this_file_name = (
            which_dir + "/d" + actual_field_names[icomp] + "dx" + this_ext
        )
        these_names.append(this_file_name)

    thse_names.append(which_dir + "/dPSdx" + this_ext)

    # if comm.Get_rank() == 0:
    #     print(these_names)

    ###########################################################################################
    # read mesh and redefine it based on the boundaring box if said
    pynekread(full_fname_mesh, comm, msh=msh, data_dtype=np.single)

    if msh.gdim < 3:
        sys.exit("only 3D data is supported!" , 
                 "you can convert your data to 3D using 'convert_2Dstats_to_3D'!")

    if if_do_dssum_before_interp:
        msh_conn = MeshConnectivity(comm, msh, rel_tol=1e-5)

    if if_create_boundingBox_for_interp:
        xyz_max = np.max(xyz, axis=0)
        xyz_min = np.min(xyz, axis=0)

        if comm.Get_rank() == 0:
            print("xyz_min: ", xyz_min)
            print("xyz_max: ", xyz_max)

        cond = (
            (msh.x >= xyz_min[0])
            & (msh.x <= xyz_max[0])
            & (msh.y >= xyz_min[1])
            & (msh.y <= xyz_max[1])
            & (msh.z >= xyz_min[2])
            & (msh.z <= xyz_max[2])
        )

        mp = MeshPartitioner(comm, msh=msh, conditions=[cond])
        msh = mp.create_partitioned_mesh(
            msh, partitioning_algorithm="load_balanced_linear", create_conectivity=True
        )

    ###########################################################################################
    # compute coef, for interpolation
    coef = Coef(msh, comm, get_area=False)

    ###########################################################################################
    # initiate probes
    # probes = Probes(comm, probes=xyz, msh=msh, \
    #                 point_interpolator_type="multiple_point_legendre_numpy", \
    #                 global_tree_type="domain_binning" , \
    #                 max_pts = 256 )
    if not if_pass_points_to_rank0_only:
        probes = Probes(
            comm,
            probes=xyz,
            msh=msh,
            point_interpolator_type="multiple_point_legendre_numpy",
            max_pts=128,
            output_fname = interpolation_output_fname,
            find_points_tol=find_points_tol
        )
    else:
        if comm.Get_rank() == 0:
            probes = Probes(
                comm,
                probes=xyz,
                msh=msh,
                point_interpolator_type="multiple_point_legendre_numpy",
                max_pts=128,
                output_fname = interpolation_output_fname,
                find_points_tol=find_points_tol
            )
        else:
            probes = Probes(
                comm,
                probes=None,
                msh=msh,
                point_interpolator_type="multiple_point_legendre_numpy",
                max_pts=128,
                output_fname = interpolation_output_fname,
                find_points_tol=find_points_tol
            )

    ###########################################################################################
    for fname in these_names:

        if comm.Get_rank() == 0:
            print("----------- working on file: ", fname)

        #########################
        field_names = return_list_of_vars_from_filename(fname)
        # if comm.Get_rank() == 0:
        #     print(field_names)
        #########################

        for icomp in range(0, len(field_names)):
            if comm.Get_rank() == 0:
                print(
                    "---working on field ", icomp, "from a total of ", len(field_names)
                )

            # load the field
            mean_fields.add_field(
                comm,
                field_name="tmpF",
                file_type="fld",
                file_name=fname,
                file_key=field_names[icomp],
                dtype=np.single,
            )

            if if_create_boundingBox_for_interp:
                mean_fields = mp.create_partitioned_field(
                    mean_fields, partitioning_algorithm="load_balanced_linear"
                )

            # do dssum to make it continuous
            if if_do_dssum_before_interp:
                msh_conn.dssum(field=mean_fields.registry["tmpF"], msh=msh, average="multiplicity")
                # coef.dssum(mean_fields.registry["tmpF"], msh)

            # interpolate the fields
            probes.interpolate_from_field_list(
                0, [mean_fields.registry["tmpF"]], comm, write_data=True
            )

            mean_fields.clear()
###########################################################################################
###########################################################################################
