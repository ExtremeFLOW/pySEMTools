#%% geometric stretching between 0 and xmax with a maximum dx
def geometric_stretching(dx0,stretching,dxmax,xmax):
    import numpy as np

    dx_base     = dx0 * (stretching**np.linspace(0,1000,1001))
    x_base      = np.cumsum( dx_base )
    N_inrange   = np.sum( x_base<=xmax )
    x_base      = x_base[0:N_inrange]
    x_base      = x_base/np.max(x_base)

    x                = np.zeros(N_inrange+1)
    x[0]             = 0.0
    x[1:N_inrange+1] = x_base

    return x

#%%
def shift_points_to_cosine_pipe(R,theta,z , retau,r_crit,kplus,Lz,nmodesZ ):
    import numpy as np

    pipe_R  = 1.0
    lamZ    = Lz / nmodesZ
    lamTH   = lamZ

    surf_y = kplus/retau * np.cos(2.0*np.pi*z/lamZ) \
                         * np.cos(2.0*np.pi*theta*pipe_R/lamTH)
    rr = r_crit + (pipe_R-r_crit) * (R-r_crit)/(pipe_R-r_crit) * (1-surf_y)

    R  = (R>r_crit)*rr + (R<=r_crit)*R

    return R,theta,z

#%%
def polar_to_cartesian(R,T):
    import numpy as np

    X = R * np.cos(T)
    Y = R * np.sin(T)

    return X,Y

#%% user specified function to define the interpolation points
def user_defined_interpolating_points():
    import numpy as np
    import pysemtools.interpolation.pointclouds as pcs
    import pysemtools.interpolation.utils as interp_utils

    print("generate interpolation points")

    # Create the coordinates of the plane you want
    theta_bbox  = [0.0, 2.*np.pi]
    z_bbox      = [0, 4.*np.pi]  # See how here I am just setting this to be one value

    ntheta  = 168*7
    nz      = 1

    # add one point to exclude since periodic
    theta = np.linspace( theta_bbox[0] , theta_bbox[1] , ntheta+1 )
    # z     = np.linspace( z_bbox[0] , z_bbox[1] , nz+1 )
    
    # exclude last points since periodic
    theta = .5 * ( theta[0:ntheta]+theta[1:] )
    z     = .5
    # z     = .5 * ( z[0:nz]+z[1:])

    re_tau      = 1000.
    dr0         = .3/re_tau
    stretching  = 1.05
    drmax       = 15./re_tau
    rmax        = 1.0

    r_base      = geometric_stretching(dr0,stretching,drmax,rmax)
    r_base[0]   = 0.1/re_tau    # to avoid potential issues when maping to the exact walls
    r_base      = 1-r_base
    nr          = r_base.size
    print('nr=', nr)

    R,T,Z = np.meshgrid(r_base, theta, z, indexing="ij")

    r_crit  = 0.8
    kplus   = 40.0
    Lz      = 4.0 * np.pi
    nmodesZ = 1.0 * 42
##    R,T,Z   = shift_points_to_cosine_pipe(R,T,Z , re_tau,r_crit,kplus,Lz,nmodesZ )
    X,Y     = polar_to_cartesian(R,T)

    xyz = interp_utils.transform_from_array_to_list(nr, ntheta, nz, [X, Y, Z])

    return xyz
###########################################################################################
###########################################################################################





###########################################################################################
###########################################################################################
###########################################################################################
###########################################################################################
#%% the presumed complete workflow
###########################################################################################
###########################################################################################
from pysemtools.postprocessing.statistics.RS_budgets import compute_and_write_additional_pstat_fields
from pysemtools.postprocessing.statistics.RS_budgets import interpolate_all_stat_and_pstat_fields_onto_points
from pysemtools.postprocessing.statistics.RS_budgets import convert_2Dstats_to_3D

from mpi4py import MPI  # equivalent to the use of MPI_init() in C
comm = MPI.COMM_WORLD

# filenames
# fname_mesh = "s01FD_3d0.f00001"
# fname_mean = "FD_3d0.f00001"
# fname_stat = "FD_3d0.f00001"
stats2D_filename = "batch_fluid_stats0.f00000"
stats3D_filename = "batch_fluid_stats_3D0.f00000"
fname_mesh = stats3D_filename
fname_mean = stats3D_filename
fname_stat = stats3D_filename
which_dir = "./"
which_code = "neko"
nek5000_stat_type = ""

if_do_dssum_on_derivatives = False          # whether to do dssum to computed derivatives... not recommended
if_do_dssum_before_interp = False            # whether to do dssum before interpolation
if_create_boundingBox_for_interp = False    # whether to create a bounding box around the point cloud
                                            # useful for cases where there is a cluster of points somewhere
if_pass_points_to_rank0_only = True         # pass all points to rank0 only. this is to avoid passing duplicate points
                                            # alternative is for each rank to generate its own points only, but that is too complex

# step 1:
# the script to average the stat files in TIME goes here

# step 1.5:
# can take the average in space here: currently only if the output is still 3D
# the filenames might need to be changed after this call
# some_function_to_average in space

# step <2:
convert_2Dstats_to_3D(stats2D_filename,stats3D_filename,datatype='single')

# step 2:
# compute the additional fields based on the 44 stat fields.
# do not do this for basic statistics
compute_and_write_additional_pstat_fields(which_dir,fname_mesh,fname_mean,fname_stat,\
                                          if_write_mesh=True,which_code=which_code,nek5000_stat_type=nek5000_stat_type, \
                                          if_do_dssum_on_derivatives=if_do_dssum_on_derivatives)

# step <3:
# define interpolation points
# WARNING HERE
# this should either be called from rank 0 only and then propoerly distributed into mpi ranks
# or each rank should generate its own points
if comm.Get_rank() == 0:
    xyz = user_defined_interpolating_points()
else:
    xyz = 0

# step 3:
# interpolate the 44+N fields onto the interpolation points
# this function is not working at the moment
# For starters, we want this function to write the interpolated fields in MATLAB format so we can debug
interpolate_all_stat_and_pstat_fields_onto_points(
    which_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    xyz,
    which_code=which_code,
    nek5000_stat_type=nek5000_stat_type,
    if_do_dssum_before_interp=if_do_dssum_before_interp,
    if_create_boundingBox_for_interp=if_create_boundingBox_for_interp,
    if_pass_points_to_rank0_only=if_pass_points_to_rank0_only
)

# step 3.5:
# potential script to perform averaging on the interpolated fields is called here
# this could include rotation, for example in the pipe

# step 4:
# function to compute budgets is called  here

# step 4.5:
# scrtipt to plot, etc. the budget terms is called here

# what is missing??

###########################################################################################
###########################################################################################
