from mpi4py import MPI
import os
from pysemtools.postprocessing.file_indexing import merge_index_files

# Get communicator
comm = MPI.COMM_WORLD


# Create a list with all the index files to be merged
indices = ["./sem_data/statistics/mean_field_index.json"]
merge_index_files(comm, index_list=indices, output_fname="case_files_index.json", sort_by_time = False)
