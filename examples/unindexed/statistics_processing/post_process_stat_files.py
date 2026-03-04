from mpi4py import MPI
from pysemtools.postprocessing.statistics.time_averaging import average_field_files
from pysemtools.postprocessing.file_indexing import index_files_from_folder

# Get communicator
comm = MPI.COMM_WORLD

# Average the field files in the index
dpath = "../data/sem_data/statistics/cylinder_rbc_nelv_600/"
average_field_files(comm, field_index_name = dpath + "mean_field_index.json", output_folder = dpath, output_batch_t_len=50)

index_files_from_folder(comm, folder_path="../data/sem_data/statistics/cylinder_rbc_nelv_600/", run_start_time=0, stat_start_time = 50, include_time_interval=True, include_file_contents=True)
