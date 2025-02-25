# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import json
# Get mpi info
comm = MPI.COMM_WORLD

from pysemtools.interpolation.wrappers import interpolate_fields_from_disk

def main():

    # Read the inputs file
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)

    query_points_fname = inputs['query_points_fname']
    sem_mesh_fname = inputs['spectral_element_mesh_fname']
    interpolated_fields_output_fname = inputs['output_sequence_fname']
    sem_dtype_str = inputs.get('spectral_element_mesh_type_in_memory', 'single')

    if sem_dtype_str == 'single':
        sem_dtype = np.single
    elif sem_dtype_str == 'double':
        sem_dtype = np.double
    else:
        raise ValueError(f"Invalid spectral element mesh data type: {sem_dtype_str}")

    field_interpolation_dictionary = {}
    field_interpolation_dictionary['input_type'] = "file_index"
    field_interpolation_dictionary['file_index'] = inputs["file_index_to_interpolate"]
    field_interpolation_dictionary['fields_to_interpolate'] = inputs["fields_to_interpolate"]

    # Interpolation settings that must have the same format as probe
    interpolation_settings = inputs.get('interpolation_settings', {})

    interpolate_fields_from_disk(comm, query_points_fname, [sem_mesh_fname, sem_dtype], field_interpolation_dictionary, interpolated_fields_output_fname=interpolated_fields_output_fname, **interpolation_settings)

if __name__ == "__main__":
    main()