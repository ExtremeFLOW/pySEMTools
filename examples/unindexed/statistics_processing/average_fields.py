from mpi4py import MPI
from pysemtools.postprocessing.statistics.time_averaging import average_field_files

# Get communicator
comm = MPI.COMM_WORLD

# Average the field files in the index
dpath = "../data/sem_data/statistics/cylinder_rbc_nelv_600/"
average_field_files(comm, field_index_name = dpath + "mean_field_index.json", output_folder = dpath, output_batch_t_len=50)
