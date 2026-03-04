##########################################################################################
## Some case specific functions are defined here
##########################################################################################
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

#%% convert polar to Cartesian coordinates
def polar_to_cartesian(R,T):
    import numpy as np

    X = R * np.cos(T)
    Y = R * np.sin(T)

    return X,Y

#%% define interpolation points
def user_defined_interpolating_points():
    import numpy as np
    import pysemtools.interpolation.pointclouds as pcs
    import pysemtools.interpolation.utils as interp_utils

    print("generate interpolation points")

    # Create the coordinates of the plane you want
    theta_bbox  = [0.0, 2.*np.pi]

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
    Nstruct     = [nr, ntheta, nz]
    print('Nstruct=', Nstruct)

    R,T,Z = np.meshgrid(r_base, theta, z, indexing="ij")

    # r_crit  = 0.8
    # kplus   = 40.0
    # Lz      = 4.0 * np.pi
    # nmodesZ = 1.0 * 42
##    R,T,Z   = shift_points_to_cosine_pipe(R,T,Z , re_tau,r_crit,kplus,Lz,nmodesZ )
    X,Y     = polar_to_cartesian(R,T)

    xyz = interp_utils.transform_from_array_to_list(nr, ntheta, nz, [X, Y, Z])

    return xyz, Nstruct

#%% define point-wise unit vectors for a cylindrical coordinate system to rotate tensors into
def define_cylindrical_coord_system_vectors(XYZ):
    import numpy as np

    # unit vector in direction 1 is along z
    v1 = np.zeros_like(XYZ) 
    v1[...,2] = 1.0

    # unit vector in direction 2 is along the wall normal
    # this is based on dimension one of XYZ corresponding to r
    v2 = XYZ[-1,...]-XYZ[0,...]
    v2 = v2/np.sqrt(np.sum(v2**2,axis=-1,keepdims=True))
    v2 = v2 * np.ones_like(v1)

    # the last vector in a right-handed orthogonal coordinate system is the cross product of the first two
    v3 = np.cross(v1,v2,axis=3)

    return v1,v2,v3
###########################################################################################
###########################################################################################


###############################################################################################
# NOTE: this is the MPI part
###############################################################################################
from pysemtools.postprocessing.statistics.RS_budgets import compute_and_write_additional_pstat_fields
from pysemtools.postprocessing.statistics.RS_budgets import interpolate_all_stat_and_pstat_fields_onto_points
from pysemtools.postprocessing.statistics.RS_budgets import convert_2Dstats_to_3D

from mpi4py import MPI  # equivalent to the use of MPI_init() in C
comm = MPI.COMM_WORLD

#%% step 0: calculate the 44 STATS fields and store them in the same format as Neko statistics files
# this step can be replaced by calcuating the 44 fields from snaphsots

#%% step 1 (optional): the script to average the stat fields in TIME goes here
# # first Index the files
# stats_dir = "/lustre/iwst/iwst088h/ROUGH/PIPE/Re1000/tests/4pi_smooth/struct/STATS_r1to3"
# stat_start_time = 0.3000001526876E+03
# from pysemtools.postprocessing.file_indexing import index_files_from_folder
# index_files_from_folder(comm, 
#                         folder_path=stats_dir, 
#                         run_start_time=0, 
#                         stat_start_time = stat_start_time 
#                         )

# # then calculate statistics
# from pysemtools.postprocessing.statistics.time_averaging import average_field_files
# average_field_files(comm, 
#                     field_index_name = stats_dir + "/" + "fluid_stats_index.json", 
#                     output_folder = stats_dir, 
#                     output_batch_t_len=1000)

#%% step 1.5 (optional): calculate averages in space of the 44 STATS fields here


#%% step <2 (optional): if collected statistics are on a 2D mesh, they must first be extruded to 3D
stats_dir        = "./" 
stats2D_filename = "batch_fluid_stats0.f00000"
stats3D_filename = "batch_fluid_stats_3D0.f00000"
convert_2Dstats_to_3D(
    stats_dir+"/"+stats2D_filename,
    stats_dir+"/"+stats3D_filename,
    datatype='single')

#%% step 2: compute the 99 additional fields based on the 44 stat fields.
# NOTE: does not work with basic (i.e., without budgets) statistics
# stats_dir        = "/lustre/iwst/iwst088h/ROUGH/PIPE/Re1000/tests/4pi_smooth/struct/STATS_r1to3"
# stats3D_filename = "batch_fluid_stats_3D0.f00000"
fname_mesh       = stats3D_filename
fname_mean       = stats3D_filename
fname_stat       = fname_mean
which_code       = "neko"
nek5000_stat_type = ""
if_do_dssum_on_derivatives = False          # whether to do dssum to computed derivatives... strongly NOT recommended

compute_and_write_additional_pstat_fields(
    stats_dir,
    fname_mesh,
    fname_mean,
    fname_stat,
    if_write_mesh=True,
    which_code=which_code,
    nek5000_stat_type=nek5000_stat_type,
    if_do_dssum_on_derivatives=if_do_dssum_on_derivatives)

#%% step <3: define the interpolation points
# NOTE: this should either be called from rank 0 only and then propoerly distributed into mpi ranks
#       or each rank should generate its own points
# NOTE: currently 'Nstruct_from_function' is read from here meaning you MUST have all the points on rank 0
if comm.Get_rank() == 0:
    xyz , Nstruct_from_function = user_defined_interpolating_points()
else:
    xyz = 0

#%% step 3: interpolate the 143 fields from SEM mesh onto the interpolation points
# NOTE: does not work with basic (i.e., without budgets) statistics 
if_do_dssum_before_interp        = False    # whether to do dssum before interpolation
if_create_boundingBox_for_interp = False    # whether to create a bounding box around the point cloud
                                            # useful for cases where there is a cluster of points somewhere
if_pass_points_to_rank0_only     = True     # pass all points to rank0 only. this is to avoid passing duplicate points
                                            # alternative is for each rank to generate its own points only, but that is too complex

interpolate_all_stat_and_pstat_fields_onto_points(
    stats_dir,
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


###############################################################################################
###############################################################################################
###############################################################################################
# NOTE: this part of post processing is NOT in MPI and should NOT be run in MPI
###############################################################################################
###############################################################################################
###############################################################################################
#%% load the required functions
path_to_files   = "./" 

# pipe flow at Re_tau=1000
Reynolds_number     = 1 / 0.0000530504 

If_convert_to_single = True     # converts the statiscs/budegts data into single precision
If_average           = True     # calculate average in space

if comm.Get_rank() == 0:
    Nstruct     = Nstruct_from_function # structured formation of the interpolation mesh

fname_averaged  = 'averaged_and_renamed_interpolated_fields.hdf5'   # name of the hdf5 written after rehsaping into structured formation and averaging in space
fname_budget    = "pstat3d_format.hdf5"     # name of the hdf5 file containing calculated budgets

#%% step 3.5: perform averaging on the interpolated fields
# NOTE: the use of Reynolds number assumes all fields are in non-dimensional form
#       this should be fixed later
from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import read_interpolated_stat_hdf5_fields

def av_func(x):
    import numpy as np
    return np.mean(x, axis=2, keepdims=True) * np.ones((*x.shape[:2], 6,x.shape[3]))

if comm.Get_rank() == 0:
    read_interpolated_stat_hdf5_fields( 
            path_to_files , 
            Reynolds_number , 
            If_average , 
            If_convert_to_single , 
            Nstruct , 
            av_func , 
            output_fname = fname_averaged
            )

#%% step 4: calciate budgets from interpolated and averged fields
# NOTE: budgets are done in Cartesian cooridnates
# NOTE: this part uses Re number, meaning it assumes fields are in non-dimensional form
from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import calculate_budgets_in_Cartesian
if comm.Get_rank() == 0:
    calculate_budgets_in_Cartesian(
            path_to_files   = path_to_files , 
            input_filename  = fname_averaged ,
            output_filename = fname_budget )

#%% step 5: rotate fields and perform additional averaging 
# NOTE: not properly scripted yet, for demonstraiton purposes only
import h5py
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from pysemtools.postprocessing.statistics.tensor_manipulations import rotate_6c2D_tensor
from pysemtools.postprocessing.statistics.tensor_manipulations import rotate_vector
from pysemtools.postprocessing.statistics.tensor_manipulations import define_rotation_tensor_from_vectors
from pysemtools.postprocessing.statistics.tensor_manipulations import rotate_2D_tensor

path_to_refdata     = '/lustre/iwst/iwst088h/ROUGH/PIPE/Re1000/tests/4pi_smooth/'
fnames_refdata_vel  = 'PIPE_Re1K_MEAN.dat'
fnames_refdata_RS   = 'PIPE_Re1K_RMS.dat'
fnames_refdata_buds = ['PIPE_Re1K_RSTE_uzuz.dat','PIPE_Re1K_RSTE_urur.dat',
                    'PIPE_Re1K_RSTE_utut.dat','PIPE_Re1K_RSTE_uruz.dat']

input_filename = path_to_files+'/'+fname_budget

if comm.Get_rank() == 0:
    
    with h5py.File(input_filename, "r") as input_file:
        XYZ         = np.array(input_file['XYZ'])
        v1,v2,v3    = define_cylindrical_coord_system_vectors(XYZ)
        Q           = define_rotation_tensor_from_vectors(v1,v2,v3)
        Qtrans      = np.transpose(Q,axes=(0,1,2 , 4,3 ))
        Nxyz        = XYZ.shape

        UVW     = np.array(input_file['UVW'])
        Rij     = np.array(input_file['Rij'])  
        dUidXj  = np.array(input_file['dUidXj'])

        Prod_ij         = np.array(input_file['Prod_ij']) 
        Diss_ij         = np.array(input_file['Diss_ij'])
        PRS_ij          = np.array(input_file['PRS_ij']) 
        Conv_ij         = np.array(input_file['Conv_ij']) 
        PressTrans_ij   = np.array(input_file['PressTrans_ij']) 
        TurbTrans_ij    = np.array(input_file['TurbTrans_ij']) 
        VelPressGrad_ij = np.array(input_file['VelPressGrad_ij']) 
        ViscDiff_ij     = np.array(input_file['ViscDiff_ij'])

        UVW     = rotate_vector(UVW,Qtrans,Q)
        UVW     = np.mean( np.mean( UVW ,axis=1 ),axis=1)

        Rij             = rotate_6c2D_tensor(Rij,Qtrans,Q)
        Prod_ij         = rotate_6c2D_tensor(Prod_ij,Qtrans,Q)
        Diss_ij         = rotate_6c2D_tensor(Diss_ij,Qtrans,Q)
        PRS_ij          = rotate_6c2D_tensor(PRS_ij,Qtrans,Q)
        Conv_ij         = rotate_6c2D_tensor(Conv_ij,Qtrans,Q)
        PressTrans_ij   = rotate_6c2D_tensor(PressTrans_ij,Qtrans,Q)
        TurbTrans_ij    = rotate_6c2D_tensor(TurbTrans_ij,Qtrans,Q)
        VelPressGrad_ij = rotate_6c2D_tensor(VelPressGrad_ij,Qtrans,Q)
        ViscDiff_ij     = rotate_6c2D_tensor(ViscDiff_ij,Qtrans,Q)
        

        Rij             = np.mean( np.mean( Rij ,axis=1 ),axis=1)
        Prod_ij         = np.mean( np.mean( Prod_ij ,axis=1 ),axis=1)
        Diss_ij         = np.mean( np.mean( Diss_ij ,axis=1 ),axis=1)
        PRS_ij          = np.mean( np.mean( PRS_ij ,axis=1 ),axis=1)
        Conv_ij         = np.mean( np.mean( Conv_ij ,axis=1 ),axis=1)
        PressTrans_ij   = np.mean( np.mean( PressTrans_ij ,axis=1 ),axis=1)
        TurbTrans_ij    = np.mean( np.mean( TurbTrans_ij ,axis=1 ),axis=1)
        VelPressGrad_ij = np.mean( np.mean( VelPressGrad_ij ,axis=1 ),axis=1)
        ViscDiff_ij     = np.mean( np.mean( ViscDiff_ij ,axis=1 ),axis=1)

        Res_ij = Prod_ij + Diss_ij + PRS_ij + Conv_ij + PressTrans_ij + TurbTrans_ij + ViscDiff_ij 
        
        dUidXj  = np.reshape( dUidXj , (Nxyz[0],Nxyz[1],Nxyz[2],3,3) , order="F" )
        dUidXj  = rotate_2D_tensor(dUidXj , Qtrans,Q)
        dUidXj  = np.mean( np.mean( dUidXj ,axis=1 ),axis=1)


        # Rer     = 1 / 0.0000530504 # 
        Rer = np.float64(input_file['Rer_here'])

        fluid_nu        = 1/Rer 
        # print(dUidXj[0,...])
        tauW_over_rho   = fluid_nu * dUidXj[0,1,0]
        u_tau           = np.sqrt(tauW_over_rho)
        delta_nu        = fluid_nu / u_tau
        Re_tau          = 1/delta_nu
        print('Re_tau = ', Re_tau )

        dr = XYZ[1:,...]-XYZ[:-1,...]
        dr = np.sqrt( np.sum(dr**2 , axis=-1) )
        r = np.cumsum(dr,axis=0)
        print( np.shape(r))
        r = r[:,0,0]
        print( np.shape(r))
        rtmp = r
        r    = np.zeros(Nxyz[0])
        r[1:] = rtmp

        yplus = r/delta_nu

        plt.figure(1)
        plt.semilogx(yplus,UVW[:,0] /u_tau   ,'k')
        df = pd.read_csv(path_to_refdata+'/'+fnames_refdata_vel, delim_whitespace=True, header=None) 
        plt.semilogx(df.iloc[:,2],df.iloc[:,5]  ,':b')
        plt.xlim((.5,Re_tau))
        plt.savefig(path_to_files+'/'+'Uplus')

        plt.figure(2)
        plt.semilogx(yplus,Rij[:,:4] /u_tau**2  ,'k')
        df = pd.read_csv(path_to_refdata+'/'+fnames_refdata_RS, delim_whitespace=True, header=None) 
        plt.semilogx(df.iloc[:,2],df.iloc[:,3:]  ,':b')
        plt.xlim((.5,Re_tau))
        plt.savefig(path_to_files+'/'+'RSplus')
        
        for k in range(4):
            plt.figure(11+k)
            plt.semilogx(yplus,Prod_ij[:,k] /(u_tau**3/delta_nu)       ,'k')
            plt.semilogx(yplus,Diss_ij[:,k] /(u_tau**3/delta_nu)       ,'b')
            plt.semilogx(yplus,PRS_ij[:,k] /(u_tau**3/delta_nu)        ,'r')
            plt.semilogx(yplus,PressTrans_ij[:,k] /(u_tau**3/delta_nu) ,'g')
            plt.semilogx(yplus,TurbTrans_ij[:,k] /(u_tau**3/delta_nu)  ,'c')
            plt.semilogx(yplus,ViscDiff_ij[:,k] /(u_tau**3/delta_nu)   ,'grey')
            plt.semilogx(yplus,Res_ij[:,k] /(u_tau**3/delta_nu)        ,':k')
            if k<4:
                df = pd.read_csv(path_to_refdata+'/'+fnames_refdata_buds[k], delim_whitespace=True, header=None) 
                if not k==3:
                    plt.semilogx(df.iloc[:,2],df.iloc[:,3:]  ,':b')
                else:
                    plt.semilogx(df.iloc[:,2],-df.iloc[:,3:]  ,':b')
            plt.xlim((.5,Re_tau))
            plt.legend(('Production','Dissipation','PRS','Pres. Trans.','Turb. Tran.','Visc. Diff','Residual'))
            plt.savefig(path_to_files+'/'+'budget')

    plt.show()