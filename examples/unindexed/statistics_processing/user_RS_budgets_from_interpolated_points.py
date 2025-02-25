###############################################################################################
# example file for how to do channel statistics from 3D collected stats (or 2D converted to 3D)
# NOTE: this part of post processing is NOT in MPI and should NOT be run in MPI
###############################################################################################

#%% load the required functions
from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import read_interpolated_stat_hdf5_fields
from pysemtools.postprocessing.statistics.RS_budgets_interpolatedPoints_notMPI import calculate_budgets_in_Cartesian

#%% 
path_to_files = './'
Reynolds_number = 10000
If_average  = True
If_convert_to_single = True
fname_averaged = 'averaged_and_renamed_interpolated_fields.hdf5'
fname_budget = "pstat3d_format.hdf5" 

# need to know the structured formation! but is know from the previous ones
nx = 41*7 
ny = 35*7
nz = 50*7
Nstruct=(nx,ny,nz)

# averaging function: here it averages along x and z
# to keep things 3D and consistent we do not reduce the averaged dimensions
def av_func(x):
    import numpy as np
    return np.mean(x, axis=(0,2), keepdims=True) * np.ones((6,x.shape[1],6,x.shape[3]))

#%%
# this scripts reads the interpolated fields, reshape thme into structured formation, and computes the spartial averages
# NOTE: input MUST be in vector formation
read_interpolated_stat_hdf5_fields(
        path_to_files ,
        Reynolds_number ,
        If_average , If_convert_to_single ,
        Nstruct , av_func ,
        output_fname = fname_averaged
        )

# this script reads the file written by 'read_interpolated_stat_hdf5_fields' and calculates the budgets
calculate_budgets_in_Cartesian(
        path_to_files = path_to_files , 
        input_filename  = fname_averaged ,
        output_filename = fname_budget )

#%% read the files written by 'calculate_budgets_in_Cartesian' and plot them for channel flow' 
import h5py
import numpy as np
from matplotlib import pyplot as plt

input_filename = path_to_files+'/'+fname_budget

# Open the input HDF5 file
k=0
with h5py.File(input_filename, "r") as input_file:
    XYZ     = np.array(input_file['XYZ'])
    dUidXj  = np.mean( np.mean(np.array(input_file['dUidXj']) ,axis=0 ),axis=1)
    Rer     = 10000.0 # np.float64(input_file['Rer_here'])

    UVW     = np.mean( np.mean(np.array(input_file['UVW']) ,axis=0 ),axis=1)
    Rij     = np.mean( np.mean(np.array(input_file['Rij']) ,axis=0 ),axis=1)
    UiUjUk  = np.mean( np.mean(np.array(input_file['UiUjUk']) ,axis=0 ),axis=1)
    Ui4     = np.mean( np.mean(np.array(input_file['Ui4']) ,axis=0 ),axis=1)
    P       = np.mean( np.mean(np.array(input_file['P']) ,axis=0 ),axis=1)

    Prod_ij         = np.mean( np.mean(np.array(input_file['Prod_ij']) ,axis=0 ),axis=1)
    Diss_ij         = np.mean( np.mean(np.array(input_file['Diss_ij']) ,axis=0 ),axis=1)
    PRS_ij          = np.mean( np.mean(np.array(input_file['PRS_ij']) ,axis=0 ),axis=1)
    Conv_ij         = np.mean( np.mean(np.array(input_file['Conv_ij']) ,axis=0 ),axis=1)
    PressTrans_ij   = np.mean( np.mean(np.array(input_file['PressTrans_ij']) ,axis=0 ),axis=1)
    TurbTrans_ij    = np.mean( np.mean(np.array(input_file['TurbTrans_ij']) ,axis=0 ),axis=1)
    VelPressGrad_ij = np.mean( np.mean(np.array(input_file['VelPressGrad_ij']) ,axis=0 ),axis=1)
    ViscDiff_ij     = np.mean( np.mean(np.array(input_file['ViscDiff_ij']) ,axis=0 ),axis=1)

    Res_ij = Prod_ij + Diss_ij + PRS_ij + Conv_ij + PressTrans_ij + TurbTrans_ij + ViscDiff_ij 

    fluid_nu = 1/Rer 
    tauW_over_rho = fluid_nu * dUidXj[0,1]
    u_tau = np.sqrt(tauW_over_rho)
    delta_nu = fluid_nu / u_tau
    Re_tau = 1/delta_nu
    print('Re_tau = ', Re_tau )

    # fluid_nu = 1./10000.
    # delta_nu = 1./550.
    # u_tau = fluid_nu / delta_nu
    y   = XYZ[0,:,0,1]+1
    yplus = y /delta_nu

    for k in range(6):
        plt.figure(11+k)
        plt.semilogx(yplus,Prod_ij[:,k] /(u_tau**3/delta_nu)       ,'k')
        plt.semilogx(yplus,Diss_ij[:,k] /(u_tau**3/delta_nu)       ,'b')
        plt.semilogx(yplus,PRS_ij[:,k] /(u_tau**3/delta_nu)        ,'r')
        plt.semilogx(yplus,PressTrans_ij[:,k] /(u_tau**3/delta_nu) ,'g')
        plt.semilogx(yplus,TurbTrans_ij[:,k] /(u_tau**3/delta_nu)  ,'c')
        plt.semilogx(yplus,ViscDiff_ij[:,k] /(u_tau**3/delta_nu)   ,'grey')
        plt.semilogx(yplus,Res_ij[:,k] /(u_tau**3/delta_nu)        ,':k')

        plt.xlim((.5,Re_tau))
        plt.legend(('Production','Dissipation','PRS','Pres. Trans.','Turb. Tran.','Visc. Diff','Residual'))

    plt.figure(1)
    plt.semilogx(yplus,UVW[:,0] /u_tau   ,'k')
    plt.xlim((.5,Re_tau))

    plt.figure(2)
    plt.semilogx(yplus,Rij[:,:4] /u_tau**2  ,'k')
    plt.xlim((.5,Re_tau))


plt.show()
