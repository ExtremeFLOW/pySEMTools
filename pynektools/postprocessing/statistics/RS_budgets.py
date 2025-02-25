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


#%% generic function to give the file name depending on the code used
def give_me_the_stat_file_name(
    which_dir, fname_base, stat_file_number, which_code="NEKO", nek5000_stat_type="s"
):
    if which_code.casefold() == "neko":
        output_file_name = which_dir + "/" + fname_base
    elif which_code.casefold() == "nek5000":
        output_file_name = (
            which_dir + "/" + nek5000_stat_type + stat_file_number + fname_base
        )
    # print('filename here was: ', output_file_name )
    return output_file_name
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
    if scal_fields>39:
        import warnings
        print("Number of scalar fields: "+(f'{(scal_fields):.0f}'))
        warnings.warn("The number of scalar fields above 39 is not supported. "+
                      "This was done to make converted 2D statistics files consistent! "+
                      "Limiting the number to 39...")
        scal_fields = 39
        

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

#%% convert 2D statistics to 3D
def convert_2Dstats_to_3D(stats2D_filename,stats3D_filename,datatype='single'):
    from mpi4py import MPI 
    import numpy as np
    import warnings
    from pysemtools.io.ppymech.neksuite import pynekread,pynekwrite
    from pysemtools.datatypes.msh import Mesh
    from pysemtools.datatypes.field import FieldRegistry
    from pysemtools.datatypes.utils import extrude_2d_sem_mesh

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
    print('fields in the give file: ', field_names )

    # extruding mesh and fields
    msh3d, fld3d = extrude_2d_sem_mesh(comm, lz = msh.lx, msh = msh, fld = fld)

    # filling in the missing z velocity
    z_vel = fld3d.registry['s39']
    fld3d.add_field(comm, field_name="w", field=z_vel, dtype=np.single)
    warnings.warn('The s39 field (z-velocity) was not removed from the file. '+ 
                  'Be careful with potential inconsistencies!')
    
    # writing the extruded stats file
    pynekwrite(stats3D_filename, comm, msh=msh3d, fld=fld3d, write_mesh=True, wdsz=wdsz)




###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
## genertic function to compute the additional fields required for budget terms, etc.
###########################################################################################
###########################################################################################
def compute_and_write_additional_pstat_fields(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    if_write_mesh=False,
    which_code="NEKO",
    nek5000_stat_type="s",
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

    # see if nek5000 file names, etc. are correct
    if which_code.casefold() == "nek5000":
        if nek5000_stat_type != "s" and nek5000_stat_type != "t":
            sys.exit(
                'for NEK5000 statistics nek5000_stat_type can be either "s" or "t"'
            )

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

    mean_fields = FieldRegistry(comm)
    stat_fields = FieldRegistry(comm)

    dU_dxi = FieldRegistry(comm)
    dV_dxi = FieldRegistry(comm)
    dW_dxi = FieldRegistry(comm)

    # pressure gradient and scond derivative
    dP_dxi   = FieldRegistry(comm)
    d2P_dxi2 = FieldRegistry(comm)

    # pressure tranposrt
    dPU_dxi = FieldRegistry(comm)
    dPV_dxi = FieldRegistry(comm)
    dPW_dxi = FieldRegistry(comm)

    # generic quantity
    dQ_dxi   = FieldRegistry(comm)
    d2Q_dxi2 = FieldRegistry(comm)

    ###########################################################################################
    # using the same .fXXXXXX extenstion as the mean fields
    this_ext = fname_mean[-8:]
    this_ext_check = fname_stat[-8:]

    # check the two files match. can be replaced by an error, but that seems too harsh and limiting
    if this_ext != this_ext_check:
        warnings.warn(
            "File index of fname_stat and fname_mean differ! Hope you know what you are doing!"
        )

    fname_gradU = which_dir + "/dUdx" + this_ext
    fname_hessU = which_dir + "/d2Udx2" + this_ext
    fname_derivP = which_dir + "/dnPdxn" + this_ext
    fname_gradPU = which_dir + "/dPUdx" + this_ext
    full_fname_mesh = which_dir + "/" + fname_mesh

    ###########################################################################################
    # read mesh and compute coefs
    pynekread(full_fname_mesh, comm, msh=msh, data_dtype=np.single)
    if if_do_dssum_on_derivatives:
        msh_conn = MeshConnectivity(comm, msh, rel_tol=1e-5)

    if msh.gdim < 3:
        sys.exit("only 3D data is supported at the moment! "+
                 "You can, however, use 'convert_2Dstats_to_3D' to extrude the file to 3D!")

    coef = Coef(msh, comm, get_area=False)

    ###########################################################################################
    # define file_keys of the fields based on the codes

    # direty fix: not needed for neko. but can't leave it empty or undefined
    stat_file_number_PU = ["03", "04", "04"]
    stat_file_number_UiUj = ["02", "02", "02", "03", "03", "03"]
    stat_file_number_UiUjUk = [
        "06",
        "07",
        "07",
        "07",
        "07",
        "08",
        "09",
        "08",
        "08",
        "08",
    ]

    if which_code.casefold() == "neko":
        file_keys_mean_fields = ["vel_0", "vel_1", "vel_2", "pres"]

        # key names taken from: https://neko.cfd/docs/develop/df/d8f/statistics-guide.html
        #                            "PU"       "PV"      "PW"
        file_keys_PU = ["scal_21", "scal_22", "scal_23"]

        #                           "UU"  , "VV"  , "WW"  , "UV" ,  "UW"  ,  "VW"  ]
        file_keys_UiUj = ["scal_0", "scal_1", "scal_2", "scal_3", "scal_4", "scal_5"]

        #                           "UUU"  , "VVV"  , "WWW"  , "UUV"  , "UUW"  , "UVV"  , "UVW"  , "VVW"  ,  "UWW"  ,  "VWW"
        file_keys_UiUjUk = [
            "scal_6",
            "scal_7",
            "scal_8",
            "scal_9",
            "scal_10",
            "scal_11",
            "scal_12",
            "scal_13",
            "scal_14",
            "scal_15",
        ]

    elif which_code.casefold() == "nek5000":
        file_keys_mean_fields = ["vel_0", "vel_1", "vel_2", "temp"]

        # key names taken from https://github.com/KTH-Nek5000/KTH_Toolbox/blob/devel/tools/stat/stat_IO.f
        #                       "PU"    "PV"    "PW"
        file_keys_PU = ["temp", "vel_0", "vel_1"]

        #                           "UU"  , "VV"  , "WW"  , "UV"  , "UW"  , "VW"
        file_keys_UiUj = [
            "vel_0",
            "vel_1",
            "vel_2",
            "vel_0",
            "vel_2",
            "vel_1",
        ]  # not a mistake! nek5000's order is a bit strange

        #                          "UUU" , "VVV" , "WWW" , "UUV" , "UUW", "UVV" , "UVW" , "VVW" , "UWW" ,"VWW"
        file_keys_UiUjUk = [
            "temp",
            "vel_0",
            "vel_1",
            "vel_2",
            "temp",
            "vel_0",
            "vel_2",
            "vel_1",
            "vel_2",
            "temp",
        ]

    else:
        sys.exit("which_code can be either NEKO or NEK5000")

    ###########################################################################################
    # Read velocity and pressure
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_mean, "01", which_code, nek5000_stat_type
    )
    mean_fields.add_field(
        comm,
        field_name="U",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[0],
        dtype=np.single,
    )
    mean_fields.add_field(
        comm,
        field_name="V",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[1],
        dtype=np.single,
    )
    mean_fields.add_field(
        comm,
        field_name="W",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[2],
        dtype=np.single,
    )
    mean_fields.add_field(
        comm,
        field_name="P",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_mean_fields[3],
        dtype=np.single,
    )

    ###########################################################################################
    # velocity first and second derivatives
    ###########################################################################################
    #
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["U"], dU_dxi)
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["V"], dV_dxi)
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["W"], dW_dxi)
    if (
        if_do_dssum_on_derivatives
    ):  # this could be a terrible idea to do dssum on a vector that is then differentiated again
        if comm.Get_rank() == 0:
            warnings.warn(
                "you are doing dssum on a derivative that you are differentiating again. maybe change this."
            )
        do_dssum_on_3comp_vector(dU_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dV_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dW_dxi, msh_conn, msh)

    write_file_9c(
        comm, msh, dU_dxi, dV_dxi, dW_dxi, fname_gradU, if_write_mesh=if_write_mesh
    )

    compute_scalar_second_derivative(comm, msh, coef, dU_dxi, dU_dxi)
    compute_scalar_second_derivative(comm, msh, coef, dV_dxi, dV_dxi)
    compute_scalar_second_derivative(comm, msh, coef, dW_dxi, dW_dxi)
    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dU_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dV_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dW_dxi, msh_conn, msh)
    write_file_9c(
        comm, msh, dU_dxi, dV_dxi, dW_dxi, fname_hessU, if_write_mesh=if_write_mesh
    )

    ###############################
    # free up some memory
    ###############################
    del dU_dxi
    del dV_dxi
    del dW_dxi
    ################################

    ###########################################################################################
    # pressure first and second derivatives
    ###########################################################################################
    compute_scalar_first_derivative(comm, msh, coef, mean_fields.registry["P"], dP_dxi)
    compute_scalar_second_derivative(comm, msh, coef, dP_dxi, d2P_dxi2)
    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dP_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(d2P_dxi2, msh_conn, msh)
    write_file_6c(
        comm, msh, dP_dxi, d2P_dxi2, fname_derivP, if_write_mesh=if_write_mesh
    )

    ###############################
    # free up some memory
    ###############################
    del dP_dxi
    del d2P_dxi2
    mean_fields.clear()  # no longer needed
    ###############################

    ###########################################################################################
    # pressure velocity product
    ###########################################################################################
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_stat, stat_file_number_PU[0], which_code, nek5000_stat_type
    )
    stat_fields.add_field(
        comm,
        field_name="PU",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_PU[0],
        dtype=np.single,
    )
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_stat, stat_file_number_PU[1], which_code, nek5000_stat_type
    )
    stat_fields.add_field(
        comm,
        field_name="PV",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_PU[1],
        dtype=np.single,
    )
    this_file_name = give_me_the_stat_file_name(
        which_dir, fname_stat, stat_file_number_PU[2], which_code, nek5000_stat_type
    )
    stat_fields.add_field(
        comm,
        field_name="PW",
        file_type="fld",
        file_name=this_file_name,
        file_key=file_keys_PU[2],
        dtype=np.single,
    )

    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["PU"], dPU_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["PV"], dPV_dxi
    )
    compute_scalar_first_derivative(
        comm, msh, coef, stat_fields.registry["PW"], dPW_dxi
    )
    if if_do_dssum_on_derivatives:
        do_dssum_on_3comp_vector(dPU_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dPV_dxi, msh_conn, msh)
        do_dssum_on_3comp_vector(dPW_dxi, msh_conn, msh)
    write_file_9c(
        comm, msh, dPU_dxi, dPV_dxi, dPW_dxi, fname_gradPU, if_write_mesh=if_write_mesh
    )

    stat_fields.clear()

    ###########################################################################################
    # velocity product terms
    ###########################################################################################
    # these need both first and second derivatives
    # we will be more agreesive from here on to save memory
    actual_field_names = ["UU", "VV", "WW", "UV", "UW", "VW"]

    for icomp in range(0, 6):
        if comm.Get_rank() == 0:
            print("working on: " + actual_field_names[icomp])

        this_file_name = give_me_the_stat_file_name(
            which_dir,
            fname_stat,
            stat_file_number_UiUj[icomp],
            which_code,
            nek5000_stat_type,
        )
        stat_fields.add_field(
            comm,
            field_name=actual_field_names[icomp],
            file_type="fld",
            file_name=this_file_name,
            file_key=file_keys_UiUj[icomp],
            dtype=np.single,
        )

        compute_scalar_first_derivative(
            comm, msh, coef, stat_fields.registry[actual_field_names[icomp]], dQ_dxi
        )

        ###############################
        stat_fields.clear()  # no longer needed
        ###############################

        compute_scalar_second_derivative(comm, msh, coef, dQ_dxi, d2Q_dxi2)

        if if_do_dssum_on_derivatives:
            do_dssum_on_3comp_vector(dQ_dxi, msh_conn, msh)
            do_dssum_on_3comp_vector(d2Q_dxi2, msh_conn, msh)

        this_file_name = (
            which_dir + "/dn" + actual_field_names[icomp] + "dxn" + this_ext
        )

        write_file_6c(
            comm, msh, dQ_dxi, d2Q_dxi2, this_file_name, if_write_mesh=if_write_mesh
        )

        ###############################
        # del d2Q_dxi2
        # d2Q_dxi2.clear()    # to not exceed a 6-component array worth of memory in this loop
        # in addition d2Q_dxi2 is no longer needed after this loop
        ###############################

    del d2Q_dxi2

    ###########################################################################################
    # tripple product terms
    ###########################################################################################
    actual_field_names = [
        "UUU",
        "VVV",
        "WWW",
        "UUV",
        "UUW",
        "UVV",
        "UVW",
        "VVW",
        "UWW",
        "VWW",
    ]

    for icomp in range(0, 10):
        if comm.Get_rank() == 0:
            print("working on: " + actual_field_names[icomp])

        this_file_name = give_me_the_stat_file_name(
            which_dir,
            fname_stat,
            stat_file_number_UiUjUk[icomp],
            which_code,
            nek5000_stat_type,
        )

        stat_fields.add_field(
            comm,
            field_name=actual_field_names[icomp],
            file_type="fld",
            file_name=this_file_name,
            file_key=file_keys_UiUjUk[icomp],
            dtype=np.single,
        )

        compute_scalar_first_derivative(
            comm, msh, coef, stat_fields.registry[actual_field_names[icomp]], dQ_dxi
        )

        if if_do_dssum_on_derivatives:
            do_dssum_on_3comp_vector(dQ_dxi, msh_conn, msh)

        ###############################
        stat_fields.clear()  # no longer needed
        ###############################

        this_file_name = which_dir + "/d" + actual_field_names[icomp] + "dx" + this_ext

        write_file_3c(comm, msh, dQ_dxi, this_file_name, if_write_mesh=if_write_mesh)

    ###########################################################################################
    # finishing up
    ###########################################################################################
    del dQ_dxi
    # dQ_dxi.clear()
    stat_fields.clear()

    if comm.Get_rank() == 0:
        print("-------As a great man once said: run successful: dying ...")
###########################################################################################
###########################################################################################


###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
# interpolate the 44+N fields onto the user specified set of points
###########################################################################################
###########################################################################################
def interpolate_all_stat_and_pstat_fields_onto_points(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    xyz,
    which_code="NEKO",
    nek5000_stat_type="s",
    if_do_dssum_before_interp=True,
    if_create_boundingBox_for_interp=False,
    if_pass_points_to_rank0_only=True,
    interpolation_output_fname="interpolated_fields.hdf5"
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

    # see if nek5000 file names, etc. are correct
    if which_code.casefold() == "nek5000":
        if nek5000_stat_type != "s" and nek5000_stat_type != "t":
            sys.exit(
                'for NEK5000 statistics nek5000_stat_type can be either "s" or "t"'
            )

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

    fname_gradU  = which_dir + "/dUdx" + this_ext
    fname_hessU  = which_dir + "/d2Udx2" + this_ext
    fname_derivP = which_dir + "/dnPdxn" + this_ext
    fname_gradPU = which_dir + "/dPUdx" + this_ext
    # full_fname_mean = which_dir+"/"+fname_mean
    # full_fname_stat = which_dir+"/"+fname_stat
    full_fname_mesh = which_dir + "/" + fname_mesh

    ###########################################################################################
    # get the file name for the 44 fileds collected in run time
    if which_code.casefold() == "neko":
        these_names = [
            give_me_the_stat_file_name(
                which_dir, fname_stat, "00", which_code, nek5000_stat_type
            )
        ]
        # these_field_names = [  ]
    elif which_code.casefold() == "nek5000":
        these_names = [
            give_me_the_stat_file_name(
                which_dir, fname_mean, "01", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "02", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "03", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "04", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "05", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "06", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "07", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "08", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "09", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "10", which_code, nek5000_stat_type
            ),
            give_me_the_stat_file_name(
                which_dir, fname_mean, "11", which_code, nek5000_stat_type
            )
        ]

    # add the name of the additional fields, these are common between neko and nek5000
    these_names.extend([fname_gradU, fname_hessU, fname_derivP, fname_gradPU])

    actual_field_names = ["UU", "VV", "WW", "UV", "UW", "VW"]
    for icomp in range(0, 6):
        this_file_name = (
            which_dir + "/dn" + actual_field_names[icomp] + "dxn" + this_ext
        )
        these_names.append(this_file_name)

    actual_field_names = [
        "UUU",
        "VVV",
        "WWW",
        "UUV",
        "UUW",
        "UVV",
        "UVW",
        "VVW",
        "UWW",
        "VWW",
    ]
    for icomp in range(0, 10):
        this_file_name = which_dir + "/d" + actual_field_names[icomp] + "dx" + this_ext
        these_names.append(this_file_name)

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
            output_fname = interpolation_output_fname
        )
    else:
        if comm.Get_rank() == 0:
            probes = Probes(
                comm,
                probes=xyz,
                msh=msh,
                point_interpolator_type="multiple_point_legendre_numpy",
                max_pts=128,
                output_fname = interpolation_output_fname
            )
        else:
            probes = Probes(
                comm,
                probes=None,
                msh=msh,
                point_interpolator_type="multiple_point_legendre_numpy",
                max_pts=128,
                output_fname = interpolation_output_fname
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
