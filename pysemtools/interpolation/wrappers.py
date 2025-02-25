"""Contains functions that wrap the interpolation types to easily post process data"""

from typing import List, Tuple, Union
from mpi4py import MPI
import numpy as np
from ..io.ppymech.neksuite import pynekread
from ..datatypes.field import FieldRegistry
from .probes import Probes
import json

def interpolate_fields_from_disk(comm: MPI.Comm,
                                query_points_fname: str,
                                sem_mesh_fname: Union[list, str],
                                field_interpolation_settings: dict,
                                interpolated_fields_output_fname: str = "interpolated_fields.hdf5",
                                dtype: np.dtype = np.single,
                                write_coords: bool = True,
                                point_interpolator_type: str = 'multiple_point_legendre_numpy',
                                max_pts: int = 256,
                                find_points_iterative: list = [False, 5000],
                                find_points_comm_pattern: str = 'point_to_point',
                                elem_percent_expansion: float = 0.01,
                                global_tree_type: str = "rank_bbox",
                                global_tree_nbins: int = 1024,
                                use_autograd: bool = False,
                                find_points_tol: float = np.finfo(np.double).eps * 10,
                                find_points_max_iter: int = 50 
                                ) -> None:
    """
    Interpolates fields from disk using the probes object.

    This function wraps the probes object to make everything easier to use. 
    It reads the query points and the SEM mesh from disk, and writes the interpolated fields to disk. 
 
    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator
    query_points_fname : str
        The filename of the query points
    sem_mesh_fname : Union[str, list]
        The filename of the SEM mesh
        if the input is a list, the first element must be the filename and the second element must be the dtype of the SEM mesh in memory
    field_interpolation_settings : dict
        The settings for the field interpolation
        The dictionary must have the following keys:
        "input_type" : str. The type should be either file_sequence or file_index
        "file_sequence" : List[str]. The sequence of files to interpolate if input_type is file_sequence
        "file_index" : str. the name of the index file to use to create the file sequence if input_type is file_index 
        "field_names" : List[str]. The names of the fields to interpolate
    interpolated_fields_output_fname : str, optional
        The filename of the interpolated fields, by default "interpolated_fields.hdf5"
    dtype : np.dtype, optional
        The dtype of the interpolated fields, by default np.single
    write_coords : bool, optional
        Whether to write the coordinates of the query points, by default True
    point_interpolator_type : str, optional
        The type of point interpolator to use, by default 'multiple_point_legendre_numpy'
    max_pts : int, optional
        The maximum number of points to interpolate at once, by default 256
    find_points_iterative : list, optional
        Whether to use iterative point finding, by default [False, 5000]
    find_points_comm_pattern : str, optional
        The communication pattern to use for finding points, by default 'point_to_point'
    elem_percent_expansion : float, optional
        The percent expansion of the elements, by default 0.01
    global_tree_type : str, optional
        The type of global tree to use, by default "rank_bbox"
    global_tree_nbins : int, optional
        The number of bins in the global tree, by default 1024
    use_autograd : bool, optional
        Whether to use autograd, by default False
    find_points_tol : float, optional
        The tolerance for finding points, by default np.finfo(np.double).eps * 10
    find_points_max_iter : int, optional
        The maximum number of iterations for finding points, by default 50 
    """

    # Initialize the probes object given the inputs
    probes = Probes(comm,
        output_fname = interpolated_fields_output_fname,
        probes = query_points_fname,
        msh = sem_mesh_fname,
        write_coords  = write_coords,
        point_interpolator_type = point_interpolator_type,
        max_pts = max_pts,
        find_points_iterative = find_points_iterative,
        find_points_comm_pattern = find_points_comm_pattern ,
        elem_percent_expansion = elem_percent_expansion,
        global_tree_type = global_tree_type,
        global_tree_nbins = global_tree_nbins,
        use_autograd = use_autograd,
        find_points_tol = find_points_tol,
        find_points_max_iter = find_points_max_iter,
    )

    # Prepare a list with the inputs
    typ = field_interpolation_settings.get('input_type', None)
    if typ == "file_sequence":
        pass
    elif typ == "file_index":
        
        with open(field_interpolation_settings["file_index"], "r") as infile:
            index_file = json.load(infile)

        field_interpolation_settings['file_sequence'] = []
        for key in index_file.keys():
            try:
                int_key = int(key)
            except ValueError:
                continue
            field_interpolation_settings['file_sequence'].append(index_file[key]['path'])     
    else:
        raise ValueError("The input type must be either file_sequence or file_index")

    # Initialize a field to be used to read
    fld = FieldRegistry(comm)
    
    # Loop throught the files in the sequence
    for fname in field_interpolation_settings['file_sequence']:


        # Read the field in the sequence    
        pynekread(fname, comm, data_dtype=dtype, fld=fld)

        # Prepare a list with the inputs
        if field_interpolation_settings['fields_to_interpolate'] == ["all"]:
            field_list = [fld.registry[key] for key in fld.registry.keys()] 
            field_names = [key for key in fld.registry.keys()]
        else:
            field_list = [fld.registry[key] for key in field_interpolation_settings['fields_to_interpolate']]
            field_names = field_interpolation_settings['fields_to_interpolate']
            
        # Interpolate the fields
        probes.interpolate_from_field_list(fld.t, field_list, comm, write_data=True, field_names=field_names)

        # Clear the fields
        fld.clear()