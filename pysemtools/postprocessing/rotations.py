"""Module that contains methods to perform rotations on 3D data"""

from typing import Union
import h5py
import os
import numpy as np
from ..io.wrappers import read_data, write_data


def cartesian_to_cylindrical_rotation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """
    Obtain the rotation matrix for a given coordinate

    This function will accept one coordinate and return the rotation matrix at that coordinate
    """

    rotation_matrix = np.zeros((3, 3))

    # Check if the radius is zero
    if np.sqrt(x**2 + y**2) <= 1e-7:
        return np.eye(3)

    # Define the rotation matrix
    angle = np.arctan2(y, x)
    if angle < 0:
        angle = angle + 2*np.pi

    rotation_matrix[0, 0] = np.cos(angle)
    rotation_matrix[0, 1] = np.sin(angle)
    rotation_matrix[0, 2] = 0
    rotation_matrix[1, 0] = -np.sin(angle)
    rotation_matrix[1, 1] = np.cos(angle)
    rotation_matrix[1, 2] = 0
    rotation_matrix[2, 0] = 0
    rotation_matrix[2, 1] = 0
    rotation_matrix[2, 2] = 1

    return rotation_matrix

def rotate_tensor(comm, msh: Union[str, list[np.ndarray]], fld: Union[str, list[np.ndarray]], tensor_field_names: list, rotation_matrix: callable, output_file: Union[str] = None, output_field_names: list = None, parallel_io: bool = False, distributed_axis: int = 0, dtype = np.single, write_mesh: bool = False) -> np.ndarray:
    """
    Rotate tensor from a file

    This function will rotate a tensor from a file using the rotation matrix

    Parameters
    ----------
    comm : MPI.COMM_WORLD
        The MPI communicator.
    msh : Union[str, list[np.ndarray]]
        If it is a string, it is the mesh file name.
        If it is a list of np.ndarrays, it is the mesh data. Should be a list with [x,y,z]
    fld : Union[str, list[np.ndarray]]
        If it is a string, it is the field file name.
        If it is a list of np.ndarrays, it is the field data. Should be the same size a tensor field names
    tensor_field_names : list
        The names of the tensor fields to be rotated in order
    rotation_matrix : callable
        The rotation matrix generator to be used
    output_file : Union[str], optional
        The output file name, by default None
    output_field_names : list, optional
        The names of the output fields, by default None

    Returns
    -------
    np.ndarray
        The rotated tensor
    """

    #Read the mesh data
    if isinstance(msh, str):
        
        data = read_data(comm, msh, ["x", "y", "z"], dtype = dtype, parallel_io = parallel_io, distributed_axis = distributed_axis)
        x = np.copy(data["x"])
        y = np.copy(data["y"])
        z = np.copy(data["z"])

    elif isinstance(msh, list):
        x = msh[0]
        y = msh[1]
        z = msh[2]
    else:
        raise ValueError('The mesh file must be in HDF5 format')
    
    # Read the tensor data
    if isinstance(fld, str):

        data = read_data(comm, fld, tensor_field_names, dtype = dtype, parallel_io = parallel_io, distributed_axis = distributed_axis)
        tensor = []
        for field_name in tensor_field_names:
            tensor.append(data[field_name])    

    elif isinstance(fld, list):
        tensor = fld
    else:
        raise ValueError('The field file must be in HDF5 format')
    
    # Rotate the tensor
    if len(tensor) == 3:
        rotated_tensor = rotate_rank1_tensor(x, y, z, tensor, rotation_matrix)
    elif len(tensor) == 9:
        rotated_tensor = rotate_rank2_tensor(x, y, z, tensor, rotation_matrix)
    else:
        raise ValueError('Rotation in this type of tensor is not implemented')
    
    if not isinstance(output_file, type(None)):
        # Construct the output data dictionary
        data = {}
        for i, field_name in enumerate(tensor_field_names):
            if isinstance(output_field_names, type(None)):
                data[f"rot_{field_name}"] = rotated_tensor[i]
            else:
                data[f"{output_field_names[i]}"] = rotated_tensor[i]
        # Write the data to the file
        write_data(comm, output_file, data, parallel_io = parallel_io, distributed_axis=distributed_axis, dtype = dtype, msh = [x, y, z], write_mesh=write_mesh)
    else:
        return rotated_tensor 

def rotate_rank1_tensor(x: np.ndarray, y: np.ndarray, z: np.ndarray, tensor: list[np.ndarray], rotation_matrix: callable) -> np.ndarray:
    """
    Rotate a rank 1 tensor

    This function will rotate a rank 1 tensor using the rotation matrix

    Parameters
    ----------
    x : np.ndarray
        The x coordinate of the tensor
    y : np.ndarray
        The y coordinate of the tensor
    z : np.ndarray
        The z coordinate of the tensor
    tensor : list[np.ndarray, np.ndarray, np.ndarray]
        The tensor to be rotated. A rank 1 tensor is a list of 3 np.ndarrays.
        each of them corresponding to the x, y, and z components of the column vector that is the tensor
    rotation_matrix : callable
        The rotation matrix generator to be used

    Returns
    -------
    list[np.ndarray, np.ndarray, np.ndarray]
        The rotated tensor
    """

    field_shape = x.shape
    field_flattened_shape = x.size

    # Flatten the fields to make them easier to treat
    x.shape = (field_flattened_shape,)
    y.shape = (field_flattened_shape,)
    z.shape = (field_flattened_shape,)
    
    # Flatten and allocate new data
    rotated_tensor = []
    for i, _ in enumerate(tensor):
        tensor[i].shape = (field_flattened_shape,)
        rotated_tensor.append(np.zeros_like(tensor[i]))


    point_tensor = np.zeros((3, 1))
    for i in range(0, x.size):

        # Get the rotation matrix
        rotation_matrix_ = rotation_matrix(x[i], y[i], z[i])

        # assig the tensor
        point_tensor[0, 0] = tensor[0][i]
        point_tensor[1, 0] = tensor[1][i]
        point_tensor[2, 0] = tensor[2][i]

        # Rotate the tensor
        rotated_point_tensor = rotation_matrix_@point_tensor

        # Assign the rotated tensor
        rotated_tensor[0][i] = rotated_point_tensor[0, 0]
        rotated_tensor[1][i] = rotated_point_tensor[1, 0]
        rotated_tensor[2][i] = rotated_point_tensor[2, 0]

    # Reshape the fields
    x.shape = field_shape
    y.shape = field_shape
    z.shape = field_shape
    for i, _ in enumerate(rotated_tensor):
        tensor[i].shape = field_shape
        rotated_tensor[i].shape = field_shape

    return rotated_tensor

def rotate_rank2_tensor(x: np.ndarray, y: np.ndarray, z: np.ndarray, tensor: list[np.ndarray], rotation_matrix: callable) -> np.ndarray:
    """
    Rotate a rank 2 tensor

    This function will rotate a rank 2 tensor using the rotation matrix

    Parameters
    ----------
    x : np.ndarray
        The x coordinate of the tensor
    y : np.ndarray
        The y coordinate of the tensor
    z : np.ndarray
        The z coordinate of the tensor
    tensor : list[np.ndarray, np.ndarray, np.ndarray]
        The tensor to be rotated. A rank 2 tensor is a list of 3x3 np.ndarrays.
        each 3 entries of the list will fill a row of the tensor
    rotation_matrix : callable
        The rotation matrix generator to be used

    Returns
    -------
    list[np.ndarray, np.ndarray, np.ndarray]
        The rotated tensor
    """

    field_shape = x.shape
    field_flattened_shape = x.size

    # Flatten the fields to make them easier to treat
    x.shape = (field_flattened_shape,)
    y.shape = (field_flattened_shape,)
    z.shape = (field_flattened_shape,)
    
    # Flatten and allocate new data
    rotated_tensor = []
    for i, _ in enumerate(tensor):
        tensor[i].shape = (field_flattened_shape,)
        rotated_tensor.append(np.zeros_like(tensor[i]))


    point_tensor = np.zeros((3, 3))
    for i in range(0, x.size):

        # Get the rotation matrix
        rotation_matrix_ = rotation_matrix(x[i], y[i], z[i])

        # assig the tensor
        point_tensor[0, 0] = tensor[0][i]
        point_tensor[0, 1] = tensor[1][i]
        point_tensor[0, 2] = tensor[2][i]
        point_tensor[1, 0] = tensor[3][i]
        point_tensor[1, 1] = tensor[4][i]
        point_tensor[1, 2] = tensor[5][i]
        point_tensor[2, 0] = tensor[6][i]
        point_tensor[2, 1] = tensor[7][i]
        point_tensor[2, 2] = tensor[8][i]  

        # Rotate the tensor
        rotated_point_tensor = rotation_matrix_@point_tensor@rotation_matrix_.T

        # Assign the rotated tensor
        rotated_tensor[0][i] = rotated_point_tensor[0, 0]
        rotated_tensor[1][i] = rotated_point_tensor[0, 1]
        rotated_tensor[2][i] = rotated_point_tensor[0, 2]
        rotated_tensor[3][i] = rotated_point_tensor[1, 0]
        rotated_tensor[4][i] = rotated_point_tensor[1, 1]
        rotated_tensor[5][i] = rotated_point_tensor[1, 2]
        rotated_tensor[6][i] = rotated_point_tensor[2, 0]
        rotated_tensor[7][i] = rotated_point_tensor[2, 1]
        rotated_tensor[8][i] = rotated_point_tensor[2, 2]

    # Reshape the fields
    x.shape = field_shape
    y.shape = field_shape
    z.shape = field_shape
    for i, _ in enumerate(rotated_tensor):
        tensor[i].shape = field_shape
        rotated_tensor[i].shape = field_shape

    return rotated_tensor