# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import cProfile

import os
os.environ["PYSEMTOOLS_DEBUG"] = 'true'

# Get mpi info
comm = MPI.COMM_WORLD

# Data types
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field, FieldRegistry

# Readers
from pysemtools.io.ppymech.neksuite import preadnek, pynekread

# Writers
from pysemtools.io.ppymech.neksuite import pwritenek, pynekwrite

# Sampler
from pysemtools.compression.gpc_direct_sampling import DirectSampler

def main():

    # Read the data
    msh = Mesh(comm, create_connectivity=False)    
    pynekread(filename=file_sequence[0], comm = comm, data_dtype=dtype, msh = msh)

    # Initialize coef 
    coef = Coef(msh=msh, comm=comm)

    if ifcompress:
        # Compress
        if profile:
            prof = cProfile.Profile()
            prof.enable()

        for i, fname in enumerate(file_sequence):

            print("=============================================")
            fld = FieldRegistry(comm)
            pynekread(fname, comm, data_dtype=dtype, fld = fld)
            
            # Initialize the sampler
            ds = DirectSampler(comm=comm, msh=msh, bckend=bckend, max_elements_to_process=max_elements_to_process, dtype=dtype)
            
            # Calculate options    
            bitrate = n_samples/(msh.lx*msh.ly*msh.lz)

            # Sample all the fields in the registry
            for key in fld.registry.keys():
                # Compress
                ds.sample_field(field=fld.registry[key], field_name=key, covariance_method="dlt", compression_method="fixed_bitrate", bitrate = bitrate, covariance_keep_modes=n_modes_to_keep, max_samples_per_it= max_samples_per_it)

            if ifwrite_uq:
                uq_fld = FieldRegistry(comm)
                uq_fld_dx = FieldRegistry(comm)
                uq_fld_dy = FieldRegistry(comm)
                uq_fld_dz = FieldRegistry(comm)
                for key in fld.registry.keys():  
                    # Get estimated of uncertainty for the velocity and its derivatives
                    _, u_std_d = ds.reconstruct_field(field_name = key, get_std = True, unsampled_field_available=True)
                    
                    # Add directly yo the field registry
                    uq_fld.add_field(comm, field_name=f"{key}", field=u_std_d.cpu().numpy(), dtype=dtype)
                    
                    # Get the UQ of the derivatives
                    _, dudr_std_d = ds.reconstruct_field(field_name = key, get_std = True, unsampled_field_available=True, std_op=ds.dr_d)
                    _, duds_std_d = ds.reconstruct_field(field_name = key, get_std = True, unsampled_field_available=True, std_op=ds.ds_d)
                    if msh.gdim > 2:
                        _, dudt_std_d = ds.reconstruct_field(field_name = "u", get_std = True, unsampled_field_available=True, std_op=ds.dt_d)

                    if msh.gdim == 2:

                        dudx_std = dudr_std_d.cpu().numpy()*coef.drdx + duds_std_d.cpu().numpy()*coef.dsdx
                        dudy_std = dudr_std_d.cpu().numpy()*coef.drdy + duds_std_d.cpu().numpy()*coef.dsdy

                    elif msh.gdim == 3:
                        dudx_std = dudr_std_d.cpu().numpy()*coef.drdx + duds_std_d.cpu().numpy()*coef.dsdx + dudt_std_d.cpu().numpy()*coef.dtdx
                        dudy_std = dudr_std_d.cpu().numpy()*coef.drdy + duds_std_d.cpu().numpy()*coef.dsdy + dudt_std_d.cpu().numpy()*coef.dtdy
                        dudz_std = dudr_std_d.cpu().numpy()*coef.drdz + duds_std_d.cpu().numpy()*coef.dsdz + dudt_std_d.cpu().numpy()*coef.dtdz

                    # Put the in the files accordinglt
                    uq_fld_dx.add_field(comm, field_name=f"{key}", field=dudx_std, dtype=dtype)
                    uq_fld_dy.add_field(comm, field_name=f"{key}", field=dudy_std, dtype=dtype)
                    if msh.gdim == 3:
                        uq_fld_dz.add_field(comm, field_name=f"{key}", field=dudz_std, dtype=dtype)

                # Write the UQ fields
                pynekwrite(f"{output_folder}uq_{decompression_output}0.f{str(i).zfill(5)}", comm=comm, msh=msh, fld=uq_fld, wdsz=4)
                pynekwrite(f"{output_folder}uq_dfdx_{decompression_output}0.f{str(i).zfill(5)}", comm=comm, msh=msh, fld=uq_fld_dx, wdsz=4)
                pynekwrite(f"{output_folder}uq_dfdy_{decompression_output}0.f{str(i).zfill(5)}", comm=comm, msh=msh, fld=uq_fld_dy, wdsz=4)
                if msh.gdim == 3:
                    pynekwrite(f"{output_folder}uq_dfdz_{decompression_output}0.f{str(i).zfill(5)}", comm=comm, msh=msh, fld=uq_fld_dz, wdsz=4)

            ## Compress
            ds.compress_samples(lossless_compressor="bzip2")

            ## Write
            ds.write_compressed_samples(comm=comm, filename=f"{output_folder}{compression_output}_{i}")
            
        if profile:
            prof.disable()
            prof.dump_stats('./compression_cpu_%d.prof' %comm.Get_rank())

    if ifdecompress:
         # Decompress
        if profile:
            prof = cProfile.Profile()
            prof.enable()
        
        for i in range(len(file_sequence)):
            
            ## Read
            ds_read = DirectSampler(comm=comm, filename=f"{output_folder}{compression_output}_{i}", bckend=bckend, max_elements_to_process=max_elements_to_process, dtype=dtype)
            
            fld_ = FieldRegistry(comm)
            
            for key in ds_read.uncompressed_data.keys():
                
                rct, _ = ds_read.reconstruct_field(field_name=key, get_mean=True, get_std=False)
                
                fld_.add_field(comm, field_name = key, field = rct.cpu().numpy(), dtype = dtype)

            ## Write
            pynekwrite(f"{output_folder}{decompression_output}0.f{str(i).zfill(5)}", comm=comm, msh=msh, fld=fld_, wdsz=4) 
        
        if profile:
            prof.disable()
            prof.dump_stats('./decompression_cpu_%d.prof' %comm.Get_rank())



fname = '../data/mixlay0.f00001'
input_foler = '../data/'
file_sequence = [fname]
output_folder = '../data/'
compression_output = "compressed_field"
decompression_output = "decompressed_field"

bckend = "torch"
max_elements_to_process = 1000
n_modes_to_keep = 8
n_samples = 2
max_samples_per_it = 1
profile = False
ifcompress = True
ifwrite_uq = True
ifdecompress = True
dtype = np.double
main()