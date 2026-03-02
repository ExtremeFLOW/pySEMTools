###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
## Preliminary functions to make life easier.
###########################################################################################
###########################################################################################
# %% generic function to compute the gradient of a scalar field
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


# %% generic function to compute the diagonal second derivatives of a scalar field from its gradient
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


# %% generic function to write a 9 component field with input as 3 vectors of 3 components each
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


# %% generic function to write a 6-component field with input as 2 vectors of 3 components each
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


# %% generic function to write a 3-component (vector) field
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


# %% generic function to write the trace of three 3-component (vector) fields (as a scalar)
def write_file_trace3(comm, msh, dU_dxi, dV_dxi, dW_dxi, fname_gradU, if_write_mesh):
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.io.ppymech.neksuite import pynekwrite
    import numpy as np

    gradU = FieldRegistry(comm)

    gradU.add_field(comm, field_name="sum", field=dU_dxi.c1 + dV_dxi.c2 + dW_dxi.c3, dtype=np.single)

    pynekwrite(fname_gradU, comm, msh=msh, fld=gradU, wdsz=4, write_mesh=if_write_mesh)

    gradU.clear()


###########################################################################################
###########################################################################################


# %% function to generate the list of fields from the file header
def return_list_of_vars_from_filename(fname):
    from pymech.neksuite.field import read_header

    header = read_header(fname)
    vars_ = header.nb_vars

    vel_fields = vars_[1]
    pres_fields = vars_[2]
    temp_fields = vars_[3]
    scal_fields = vars_[4]
    if scal_fields > 37:
        import warnings

        print("Number of scalar fields: " + (f"{(scal_fields):.0f}"))
        warnings.warn(
            "The number of scalar fields above 39 is not supported. "
            + "This was done to make converted 2D statistics files consistent! "
            + "Limiting the number to 37..."
        )
        scal_fields = 37

    field_names = []

    for i in range(vel_fields):
        tmp = ["vel_" + str(i)]
        field_names = field_names + tmp
    
    if pres_fields == 1:
        field_names = field_names + ["pres"]

    if temp_fields == 1:
        field_names = field_names + ["temp"]

    for i in range(scal_fields):
        tmp = ["scal_" + str(i)]
        field_names = field_names + tmp

    return field_names


###########################################################################################
###########################################################################################


# %% do dssum on a vector with components c1, c2, c3
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

#%% convert 2D statistics to 3D
def convert_scalar_2Dstats_to_3D(stats2D_filename,stats3D_filename,datatype='single'):
    from mpi4py import MPI 
    import numpy as np
    import warnings
    from pysemtools.io.ppymech.neksuite import pynekread,pynekwrite
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.utils import extrude_2d_sem_mesh
    from pysemtools.interpolation.interpolator import get_bbox_from_coordinates
    from pysemtools.interpolation.point_interpolator.single_point_helper_functions import GLL_pwts

    # Get mpi info
    comm = MPI.COMM_WORLD

    # initialize fields
    msh = Mesh(comm, create_connectivity=False)
    fld = FieldRegistry(comm)

    if datatype=='single':
        data_type = np.single
        wdsz = 4
    else:
        data_type = np.double
        wdsz = 8

    # read mesh and fields
    pynekread(stats2D_filename, comm, data_dtype=data_type, msh=msh, fld=fld)

    # get the 
    field_names = return_list_of_vars_from_filename(stats2D_filename)
    print('fields in the given file: ', field_names )

    # extruding mesh and fields
    ## Find how much to extrude - Extrude the size of the smallest element
    bbox = get_bbox_from_coordinates(msh.x, msh.y, msh.z)
    bbox_dist = np.zeros((bbox.shape[0], 3))
    bbox_dist[:, 0] = bbox[:, 1] - bbox[:, 0]
    bbox_dist[:, 1] = bbox[:, 3] - bbox[:, 2]
    bbox_dist[:, 2] = bbox[:, 5] - bbox[:, 4]
    local_bbox_min_dist = np.min(
        np.sqrt(bbox_dist[:, 0] ** 2 + bbox_dist[:, 1] ** 2 + bbox_dist[:, 2] ** 2) / 2
    )
    bbox_min_dist = comm.allreduce(local_bbox_min_dist, op=MPI.MIN)
    ## Generate the point distribution
    x_, _ = GLL_pwts(msh.lx)
    extrusion_size = bbox_min_dist
    point_dist = np.flip(x_ * extrusion_size) 
    msh3d, fld3d = extrude_2d_sem_mesh(comm, lz = msh.lx, msh = msh, fld = fld, point_dist=point_dist)

    # filling in the missing z velocity
    z_vel = fld3d.registry['s37']
    fld3d.add_field(comm, field_name="w", field=z_vel, dtype=np.single)
    warnings.warn('The s37 field (<ws>) was not removed from the file. '+ 
                  'Be careful with potential inconsistencies!')
    
    # writing the extruded stats file
    pynekwrite(stats3D_filename, comm, msh=msh3d, fld=fld3d, write_mesh=True, wdsz=wdsz)



###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
## generic function to compute the additional fields required for budget terms, etc.
## only implemented for Neko
###########################################################################################
###########################################################################################
def compute_and_write_additional_sstat_fields(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    if_write_mesh=False,
    if_do_dssum_on_derivatives=False,
):

    ###########################################################################################
    # do some initial checks
    import sys

    # check if file names are the same
    if fname_mean != fname_stat:
        sys.exit("fname_mean must be the same as fname_stat")


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
    dQ1_dxi = FieldRegistry(comm)
    dQ2_dxi = FieldRegistry(comm)
    dQ3_dxi = FieldRegistry(comm)
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
    file_keys_S = "pres"
    #                   "US"      "VS"      "WS"
    file_keys_UiS = ["vel_0", "vel_1", "vel_2"]
    file_keys_SS = "temp"
    #                   "USS"      "VSS"     "WSS"
    file_keys_UiSS = ["scal_2", "scal_3", "scal_4"]
    #                   "UUS"     "VVS"     "WWS"     "UVS"     "UWS"     "VWS"
    file_keys_UiUjS = ["scal_5", "scal_6", "scal_7", "scal_8", "scal_9", "scal_10"]
    file_keys_PS = "scal_11"

    file_keys_UidSdxj = [
        "scal_15",
        "scal_16",
        "scal_17",
        "scal_18",
        "scal_19",
        "scal_20",
        "scal_21",
        "scal_22",
        "scal_23",
    ]
    file_keys_SdUidxj = [
        "scal_24",
        "scal_25",
        "scal_26",
        "scal_27",
        "scal_28",
        "scal_29",
        "scal_30",
        "scal_31",
        "scal_32",
    ]

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

    compute_scalar_first_derivative(comm, msh, coef, stat_fields.registry["S"], dQ1_dxi)
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

    this_file_name = which_dir + "/dUiSdxj" + this_ext
    write_file_9c(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    ###########################################################################################
    # UiSS divergence
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
        file_key=file_keys_UiSS[1],
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

    this_file_name = which_dir + "/dUjSSdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    ###########################################################################################
    # UiUjS gradients: dUiUjS/dxj
    ###########################################################################################

    # dUUjS/dxj

    stat_fields.add_field(
        comm,
        field_name="UUS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[0],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="UVS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[3],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="UWS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[4],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UUS"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UVS"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UWS"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dUUjSdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    # dVUjS/dxj

    stat_fields.add_field(
        comm,
        field_name="UVS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[3],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="VVS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[1],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="VWS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[5],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UVS"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VVS"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VWS"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dVUjSdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    # dWUjS/dxj

    stat_fields.add_field(
        comm,
        field_name="UWS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[4],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="VWS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[5],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="WWS",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UiUjS[2],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UWS"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VWS"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["WWS"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dWUjSdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
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
    this_file_name = which_dir + "/dPSdxj" + this_ext
    write_file_3c(comm, msh, dQ1_dxi, this_file_name, if_write_mesh=if_write_mesh)

    stat_fields.clear()

    ###########################################################################################
    # UidSdxj gradients: d(UidS/dxj)/dxj
    ###########################################################################################

    # d(UdS/dxj)/dxj

    stat_fields.add_field(
        comm,
        field_name="UdSdx",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[0],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="UdSdy",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[1],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="UdSdz",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[2],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UdSdx"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UdSdy"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["UdSdz"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dUdSdxjdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    # d(VdS/dxj)/dxj

    stat_fields.add_field(
        comm,
        field_name="VdSdx",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[3],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="VdSdy",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[4],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="VdSdz",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[5],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VdSdx"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VdSdy"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["VdSdz"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dVdSdxjdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    # d(WdS/dxj)/dxj

    stat_fields.add_field(
        comm,
        field_name="WdSdx",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[6],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="WdSdy",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[7],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="WdSdz",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_UidSdxj[8],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["WdSdx"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["WdSdy"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["WdSdz"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dWdSdxjdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    ###########################################################################################
    # SdUidxj gradients: d(SdUi/dxj)/dxj
    ###########################################################################################

    # d(SdU/dxj)/dxj

    stat_fields.add_field(
        comm,
        field_name="SdUdx",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[0],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="SdUdy",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[1],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="SdUdz",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[2],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdUdx"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdUdy"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdUdz"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dSdUdxjdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    # d(SdV/dxj)/dxj

    stat_fields.add_field(
        comm,
        field_name="SdVdx",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[3],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="SdVdy",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[4],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="SdVdz",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[5],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdVdx"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdVdy"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdVdz"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dSdVdxjdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    # d(SdW/dxj)/dxj

    stat_fields.add_field(
        comm,
        field_name="SdWdx",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[6],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="SdWdy",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[7],
        dtype=np.single,
    )
    stat_fields.add_field(
        comm,
        field_name="SdWdz",
        file_type="fld",
        file_name=full_fname_stat,
        file_key=file_keys_SdUidxj[8],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdWdx"], dQ1_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdWdy"], dQ2_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["SdWdz"], dQ3_dxi
    )

    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dQ1_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ2_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dQ3_dxi, msh_conn, msh)

    this_file_name = which_dir + "/dSdWdxjdxj" + this_ext
    write_file_trace3(
        comm,
        msh,
        dQ1_dxi,
        dQ2_dxi,
        dQ3_dxi,
        this_file_name,
        if_write_mesh=if_write_mesh,
    )

    stat_fields.clear()

    del dQ1_dxi, dQ2_dxi, dQ3_dxi
    if comm.Get_rank() == 0:
        print("-------As a great man once said: run successful: dying ...")


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# interpolate the 42+N fields onto the user specified set of points
# only implemented for Neko
###########################################################################################
###########################################################################################
def interpolate_all_stat_and_sstat_fields_onto_points(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    xyz,
    if_do_dssum_before_interp=True,
    if_create_boundingBox_for_interp=False,
    if_pass_points_to_rank0_only=True,
    interpolation_output_fname="interpolated_scalar_fields.hdf5",
    find_points_tol=None,
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
        sys.exit("fname_mean must be the same as fname_stat")


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
    these_names = [ which_dir + "/" + fname_stat ]

    # add the name of the additional fields
    these_names.extend(
        [
            which_dir + "/dnSdxn" + this_ext,
            which_dir + "/dnSSdxn" + this_ext,
            which_dir + "/dUiSdxj" + this_ext,
            which_dir + "/dUjSSdxj" + this_ext,
            which_dir + "/dUUjSdxj" + this_ext,
            which_dir + "/dVUjSdxj" + this_ext,
            which_dir + "/dWUjSdxj" + this_ext,
            which_dir + "/dPSdxj" + this_ext,
            which_dir + "/dUdSdxjdxj" + this_ext,
            which_dir + "/dVdSdxjdxj" + this_ext,
            which_dir + "/dWdSdxjdxj" + this_ext,
            which_dir + "/dSdUdxjdxj" + this_ext,
            which_dir + "/dSdVdxjdxj" + this_ext,
            which_dir + "/dSdWdxjdxj" + this_ext,
        ]
    )

    # if comm.Get_rank() == 0:
    #     print(these_names)

    ###########################################################################################
    # read mesh and redefine it based on the boundaring box if said
    pynekread(full_fname_mesh, comm, msh=msh, data_dtype=np.single)

    if msh.gdim < 3:
        sys.exit(
            "only 3D data is supported!",
            "you can convert your data to 3D using 'convert_2Dstats_to_3D'!",
        )

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

    probe_kwargs = {
        "comm": comm,
        "msh": msh,
        "point_interpolator_type": "multiple_point_legendre_numpy",
        "max_pts": 128,
        "output_fname": interpolation_output_fname,
    } 
    if find_points_tol is not None:
        probe_kwargs["find_points_tol"] = find_points_tol
  
    if not if_pass_points_to_rank0_only:
        probes = Probes(probes=xyz, **probe_kwargs)
    else:
        if comm.Get_rank() == 0:
            probes = Probes(probes=xyz, **probe_kwargs)
        else:
            probes = Probes(probes=None, **probe_kwargs)

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
                msh_conn.dssum(
                    field=mean_fields.registry["tmpF"], msh=msh, average="multiplicity"
                )
                # coef.dssum(mean_fields.registry["tmpF"], msh)

            # interpolate the fields
            probes.interpolate_from_field_list(
                0, [mean_fields.registry["tmpF"]], comm, write_data=True
            )

            mean_fields.clear()


###########################################################################################
###########################################################################################
