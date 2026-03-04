""" Methods to facilitate UQ analysis. """


from typing import Union
import h5py
import numpy as np
from ...io.wrappers import read_data
import os


def NOBM(comm, field_sequence: Union[list[str], list[dict[str, np.ndarray]]], field_names: list["str"], output_file: str = None, output_field_names: str = None):
    """ 
    Use the Non overlapping batch mean method to estimate the mean and varince of the field sequence. 

    For this to work, we intrinsically assume that the fields in the sequence have the same time interval.

    Parameters
    ----------
    comm : MPI.COMM_WORLD
        The MPI communicator.
    field_sequence : Union[list[str], list[dict[str, np.ndarray]]]
        A list of the file names containing the fields or a list of dictionaries containing the fields.
    field_names : list[str]
        A list of the names of the fields. Particularly important if field_sequence is a string, since the 
        field name will be used as a key to access the data in the file.
    output_file : str, optional
        The file to save the mean and variance fields, by default None
    output_field_names : str, optional
        The name that the fields should have when saved to, for example, and hdf5 file. By default None.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary containing the mean and variance fields.
    """

    n = len(field_sequence)
    for i, field_id in enumerate(field_sequence):

        # Check whch type of file it is
        if isinstance(field_id, str):

            data = read_data(comm, field_id, field_names, dtype = np.single)

        elif isinstance(field_id, dict):

            data = {}
            for key in field_id:
                if key in field_names:
                    data[key] = field_id[key]
        
        else:
            raise ValueError("The field sequence should be a list of strings or a list of dictionaries")
        
        if i == 0:

            mean = {}
            var = {}
            for key in data:
                mean[key] = np.zeros_like(data[key])
                var[key] = np.zeros_like(data[key])

        for key in data:
            mean[key] += data[key]
            var[key] += data[key]**2

    for key in mean:
        mean[key] /= n
        var[key] = (var[key]/n - mean[key]**2)/n

    if not isinstance(output_file, type(None)):
        path = os.path.dirname(output_file)
        if path == '': path = '.'
        prefix = os.path.basename(output_file).split('.')[0]
        extension = os.path.basename(output_file).split('.')[1]
        if extension == 'hdf5':
            with h5py.File(output_file, 'w') as f:
                for i, field_name in enumerate(field_names):
                    if isinstance(output_field_names, type(None)):
                        f.create_dataset(f"mean_{field_name}", data=mean[field_name])
                        f.create_dataset(f"var_{field_name}", data=var[field_name])
                    else:
                        f.create_dataset(f"mean_{output_field_names[i]}", data=mean[field_name])
                        f.create_dataset(f"var_{output_field_names[i]}", data=var[field_name])
        else:
            raise ValueError('The output file must be in HDF5 format')
    else:

        if not isinstance(output_field_names, type(None)):
            for i, field_name in enumerate(field_names):
                new_key = output_field_names[i]
                mean[f"{new_key}"] = mean.pop(field_name)
                var[f"{new_key}"] = var.pop(field_name)

        return mean, var