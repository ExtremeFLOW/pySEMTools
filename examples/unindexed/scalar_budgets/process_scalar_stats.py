# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import os

# Get mpi info
comm = MPI.COMM_WORLD

sem_data_path = "../../data/sem_data/"
if not os.path.exists(sem_data_path):
    print("Sem data not found, cloning repository...")
    os.system(f"git clone https://github.com/adperezm/sem_data.git {sem_data_path}")
else:
    print("Sem data found.")

stats_dir = "./"

## % Convert 2D stats to 3D stats

fluid_stats2D_fname = os.path.join(sem_data_path,"statistics/channel_nelv_300_scalars/", "merged_fluid_stats2D0.f00000")
scalar_stats2D_fname = os.path.join(sem_data_path,"statistics/channel_nelv_300_scalars/", "merged_scalar_stats2D0.f00000")
fluid_stats3D_fname = "merged_fluid_stats3D.f00000"
scalar_stats3D_fname = "merged_scalar_stats3D.f00000"

from pysemtools.postprocessing.statistics.RS_budgets import convert_2Dstats_to_3D
from pysemtools.postprocessing.statistics.passive_scalar_budgets import convert_scalar_2Dstats_to_3D

convert_2Dstats_to_3D(fluid_stats2D_fname, fluid_stats3D_fname, datatype='single')
convert_scalar_2Dstats_to_3D(scalar_stats2D_fname, scalar_stats3D_fname, datatype='single')


## % Process the statistics: additional fields

from pysemtools.postprocessing.statistics.RS_budgets import compute_and_write_additional_pstat_fields
from pysemtools.postprocessing.statistics.passive_scalar_budgets import compute_and_write_additional_sstat_fields


compute_and_write_additional_pstat_fields(
    stats_dir,
    fluid_stats3D_fname,
    fluid_stats3D_fname,
    fluid_stats3D_fname,
    if_write_mesh=True,
    which_code='neko',
    nek5000_stat_type='',
    if_do_dssum_on_derivatives=False)


compute_and_write_additional_sstat_fields(
    stats_dir,
    scalar_stats3D_fname,
    scalar_stats3D_fname,
    scalar_stats3D_fname,
    if_write_mesh=True,
    if_do_dssum_on_derivatives=False)


## % Interpolate

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

def user_defined_interpolating_points():
    import numpy as np
    import pysemtools.interpolation.pointclouds as pcs
    import pysemtools.interpolation.utils as interp_utils

    print("generate interpolation points")

    # Create the coordinates of the plane you want
    x_bbox  = [0.0, 2.*np.pi]

    nz  = 1
    nx  = 201

    # add one point to exclude since periodic
    x     = np.linspace( x_bbox[0] , x_bbox[1] , nx+1 )

    # exclude last points since periodic
    z     = .001
    x     = .5 * ( x[0:nz]+x[1:])

    re_tau      = 180.
    dy0         = .3/re_tau
    stretching  = 1.05
    dymax       = 15./re_tau
    ymax        = 1.0

    y_base      = geometric_stretching(dy0,stretching,dymax,ymax)
    y_base[0]   = 0.1/re_tau    # to avoid potential issues when maping to the exact walls
    y_base      = y_base - 1
    ny          = y_base.size
    Nstruct     = [nx, ny, nz]
    print('Nstruct=', Nstruct)

    X,Y,Z = np.meshgrid(x, y_base, z, indexing="ij")

    xyz = interp_utils.transform_from_array_to_list(nx, ny, nz, [X, Y, Z])

    return xyz, Nstruct

# Call the functions

# Call the functions
# NOTE: this should either be called from rank 0 only and then propoerly distributed into mpi ranks
#       or each rank should generate its own points
if comm.Get_rank() == 0:
    xyz , Nstruct = user_defined_interpolating_points()
else:
    xyz = 0


from pysemtools.postprocessing.statistics.RS_budgets import interpolate_all_stat_and_pstat_fields_onto_points
from pysemtools.postprocessing.statistics.passive_scalar_budgets import interpolate_all_stat_and_sstat_fields_onto_points

interpolate_all_stat_and_pstat_fields_onto_points(
    stats_dir,
    fluid_stats3D_fname,
    fluid_stats3D_fname,
    fluid_stats3D_fname,
    xyz,
    which_code='neko',
    nek5000_stat_type='',
    if_do_dssum_before_interp=False,
    if_create_boundingBox_for_interp=False,
    if_pass_points_to_rank0_only=True
)

interpolate_all_stat_and_sstat_fields_onto_points(
    stats_dir,
    scalar_stats3D_fname,
    scalar_stats3D_fname,
    scalar_stats3D_fname,
    xyz,
    if_do_dssum_before_interp=False,
    if_create_boundingBox_for_interp=False,
    if_pass_points_to_rank0_only=True
)


## % Finish processing profiles and average

def av_func(x):
    import numpy as np
    return np.mean(x, axis=0, keepdims=True)


from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import read_interpolated_stat_hdf5_fields
from pysemtools.postprocessing.statistics.passive_scalar_budgets_interpolatedPoints_notMPI import read_interpolated_scalar_stat_hdf5_fields

from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import calculate_budgets_in_Cartesian
from pysemtools.postprocessing.statistics.passive_scalar_budgets_interpolatedPoints_notMPI import calculate_scalar_budgets_in_Cartesian

Reynolds_number = 180
Prandtl_number = 0.71
If_average = True
If_convert_to_single = True
fname_averaged_fluid  = 'averaged_and_renamed_interpolated_fluid_fields.hdf5'
fname_averaged_scalar = 'averaged_and_renamed_interpolated_scalar_fields.hdf5'
fname_budget_fluid = 'fluid_budgets.hdf5'
fname_budget_scalar = 'scalar_budgets.hdf5'


if comm.Get_rank() == 0:
    read_interpolated_stat_hdf5_fields(
            stats_dir ,
            Reynolds_number ,
            If_average ,
            If_convert_to_single ,
            Nstruct ,
            av_func ,
            output_fname = fname_averaged_fluid
            )

    read_interpolated_scalar_stat_hdf5_fields(
            stats_dir ,
            Reynolds_number ,
            Prandtl_number ,
            If_average ,
            If_convert_to_single ,
            Nstruct ,
            av_func ,
            output_fname = fname_averaged_scalar
            )

    calculate_budgets_in_Cartesian(
            path_to_files   = stats_dir ,
            input_filename  = fname_averaged_fluid ,
            output_filename = fname_budget_fluid )

    calculate_scalar_budgets_in_Cartesian(
            path_to_files   = stats_dir ,
            input_scalar_filename  = fname_averaged_scalar ,
            input_fluid_filename  = fname_averaged_fluid ,
            output_filename = fname_budget_scalar )
