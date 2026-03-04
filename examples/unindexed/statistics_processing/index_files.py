from mpi4py import MPI
import os
from pysemtools.postprocessing.file_indexing import index_files_from_folder

# Get communicator
comm = MPI.COMM_WORLD


# Clone the repository with the data
os.system("git clone https://github.com/adperezm/sem_data.git ../data/sem_data")

# Index the filestatistics_start_times
index_files_from_folder(comm, folder_path="../data/sem_data/statistics/cylinder_rbc_nelv_600/", run_start_time=0, stat_start_time = 50, include_time_interval=True, include_file_contents=True)
