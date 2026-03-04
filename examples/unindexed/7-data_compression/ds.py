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
    fld = FieldRegistry(comm)
    pynekread(fname, comm, data_dtype=np.double, msh = msh, fld = fld)

    # Initialize coef 
    coef = Coef(msh=msh, comm=comm)

    # Initialize the sampler
    ds = DirectSampler(comm=comm, msh=msh, bckend="torch", max_elements_to_process=1000)
    
    # Calculate options    
    bitrate = n_samples/(msh.lx*msh.ly*msh.lz)

    # Compress

    if profile:
        prof = cProfile.Profile()
        prof.enable()

    ## Sample here
    ds.sample_field(field=fld.registry["u"], field_name="u", covariance_method="dlt", compression_method="fixed_bitrate", bitrate = bitrate, covariance_keep_modes=64)
    #ds.sample_field(field=fld.registry["u"], field_name="u", covariance_method="average", covariance_elements_to_average=1, compression_method="fixed_bitrate", bitrate = bitrate)

    ## Compress
    ds.compress_samples(lossless_compressor="bzip2")

    ## Write
    ds.write_compressed_samples(comm=comm, filename="test")
    
    if profile:
        prof.disable()
        prof.dump_stats('./compression_cpu_%d.prof' %comm.Get_rank())

    # Decompress
    if profile:
        prof = cProfile.Profile()
        prof.enable()
    
    ## Read
    ds_read = DirectSampler(comm=comm, filename="test", bckend="torch", max_elements_to_process=256)
    print(ds_read.uncompressed_data.keys())
    #predict
    rct, rct_std = ds_read.reconstruct_field(field_name="u", get_mean=True, get_std=True)
    
    if profile:
        prof.disable()
        prof.dump_stats('./decompression_cpu_%d.prof' %comm.Get_rank())

    # Get the error
    error = np.linalg.norm(rct - fld.registry["u"])/np.linalg.norm(fld.registry["u"])
    print("Error: ", error)

fname = '../data/mixlay0.f00001'
#fname = '../data/tc_channel0.f00001'
#fname = '/home/adperez/Documents/gaussian_process/Gaussian Process_0823/data/turbPipe/turbPipe0.f00001'
n_samples = 2
profile = True
main()