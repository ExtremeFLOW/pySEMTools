"""Module for utility functions for the datatypes module"""

import sys
import numpy as np
from pymech.core import HexaData
from pymech.neksuite.field import Header
from .msh import Mesh
from .field import Field, FieldRegistry
from ..io.ppymech.neksuite import pwritenek
from ..interpolation.interpolator import (
    get_bbox_from_coordinates,
    get_bbox_centroids_and_max_dist,
)
from ..interpolation.mesh_to_mesh import PRefiner
from typing import Union
from ..interpolation.point_interpolator.single_point_helper_functions import GLL_pwts


NoneType = type(None)


def create_hexadata_from_msh_fld(
    msh=None, fld=None, wdsz=4, istep=0, time=0, data_dtype=np.double, write_mesh=True
):
    """
    Create a HexaData object from a msh and fld object. Used to write fld files.

    This function is used as a preprocessing step before writing fld files.

    Parameters
    ----------
    msh : Mesh
        A mesh object.
         (Default value = None).
    fld : Field
        A field object.
         (Default value = None).
    wdsz : int
        Word size of data in number of bytes. 4 for single precision, 8 for double precision.
        This is relevant when writing a file to disk, not to data processing in memory.
         (Default value = 4).
    istep : int
        Used for writing multistep files. Not really used in practice.
         (Default value = 0).
    time : float
        Time to use in the header of the file and HexaData object.
         (Default value = 0).
    data_dtype : str
        Data type to be used in the hexadata type. "np.single" or "np.double".
        (Default value = "np.double")
    write_mesh : bool
        If true, the mesh will be written to the HexaData object and posterior fld files.
         (Default value = True)

    Returns
    -------
    HexaData
        A HexaData object with the mesh and fields from the msh and fld objects.

    Examples
    --------
    From a Mesh and Field object.

    >>> data = create_hexadata_from_msh_fld(msh=msh, fld=fld)

    :meta private:
    """

    msh_fields = msh.gdim
    vel_fields = fld.vel_fields
    pres_fields = fld.pres_fields
    temp_fields = fld.temp_fields
    scal_fields = fld.scal_fields
    lx = msh.lx
    ly = msh.ly
    lz = msh.lz
    nelv = msh.nelv

    header = Header(
        wdsz,
        (lx, ly, lz),
        nelv,
        nelv,
        time,
        istep,
        fid=0,
        nb_files=1,
        nb_vars=(msh_fields, vel_fields, pres_fields, temp_fields, scal_fields),
    )

    # Create the pymech hexadata object
    data = HexaData(
        header.nb_dims,
        header.nb_elems,
        header.orders,
        header.nb_vars,
        0,
        dtype=data_dtype,
    )
    data.time = header.time
    data.istep = header.istep
    data.wdsz = header.wdsz
    data.endian = sys.byteorder

    # Include the mesh
    put_coordinates_in_hexadata_from_msh(data, msh)

    # Include the fields
    for qoi in range(0, vel_fields):
        prefix = "vel"
        put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    for qoi in range(0, pres_fields):
        prefix = "pres"
        put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    for qoi in range(0, temp_fields):
        prefix = "temp"
        put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    for qoi in range(0, scal_fields):
        prefix = "scal"
        put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    return data


def put_coordinates_in_hexadata_from_msh(data, msh):
    """
    Populate a hexadata object with coordinates from a msh object.

    This function is used as a preprocessing step before writing fld files.

    Parameters
    ----------
    data : HexaData
        A HexaData object.
    msh : Mesh
        A mesh object.

    Returns
    -------
    HexaData
        A HexaData object with the coordinates from the msh object.

    :meta private:
    """
    nelv = data.nel
    # lx = data.lr1[0]
    # ly = data.lr1[1]
    # lz = data.lr1[2]

    for e in range(0, nelv):
        data.elem[e].pos[0, :, :, :] = msh.x[e, :, :, :]
        data.elem[e].pos[1, :, :, :] = msh.y[e, :, :, :]
        data.elem[e].pos[2, :, :, :] = msh.z[e, :, :, :]

    return


def put_field_in_hexadata(data, field, prefix, qoi):
    """
    Populate a hexadata object with a field from a field object.

    Used as preprocessing step before writing fld files.

    Parameters
    ----------
    data : HexaData
        A HexaData object.
    field : ndarray
        A field to be written to the HexaData object.
    prefix : str
        Prefix of the field. vel, pres, temp, scal.
    qoi :
        Quantity of interest. Index of the field.

    Returns
    -------
    HexaData
        A HexaData object with the field populated.

    :meta private:
    """
    nelv = data.nel
    # lx = data.lr1[0]
    # ly = data.lr1[1]
    # lz = data.lr1[2]

    if prefix == "vel":
        for e in range(0, nelv):
            data.elem[e].vel[qoi, :, :, :] = field[e, :, :, :]

    if prefix == "pres":
        for e in range(0, nelv):
            data.elem[e].pres[0, :, :, :] = field[e, :, :, :]

    if prefix == "temp":
        for e in range(0, nelv):
            data.elem[e].temp[0, :, :, :] = field[e, :, :, :]

    if prefix == "scal":
        for e in range(0, nelv):
            data.elem[e].scal[qoi, :, :, :] = field[e, :, :, :]

    return


def get_gradient(msh, coef, field_list=None):
    """
    Get gradient from a 3D field vector field wrt x, y, z directions.

    The gradient of a vector field is a tensor. The gradient of a scalar field is a vector field.

    For each point in a vector field :math:`u_i = [u_1, u_2, u_3]` in space with coordinates
    :math:`x_j = [x, y, z]`, the gradient is a 3x3 tensor :math:`G(u)_{i,j}` with:

    .. math::

        G(u)_{i,j} = \partial{u_i}/\partial{x_j}.

    whith :math:`i` and :math:`j` being the rows and columns of the matrix, respectively.

    Parameters
    ----------
    msh : Mesh
        A mesh object.
    coef : Coef
        A coef object.
    field_list : list
        A list of the vector/scalar fields to calcule the 3D gradient to.
        Each entry in the list should be a 4D ndarray with the shape (nelv, lz, ly, lx).
        (Default value = None).

    Returns
    -------
    ndarray
        A 6D ndarray with the shape (nelv, lz, ly, lx, number_of_fields, 3).

    Notes
    -----

    Each point in a vector field u(x,y,z), v(x,y,z), w(x,y,z) has a gradient (jacobian) that is a 3x3 tensor.

    Each point in a scalar field s(x,y,z) has a gradient (jacobian) that is a 1x3 tensor.

    The resulting gradient is a 6D tensor with the shape (nelv, lz, ly, lx, number_of_fields, 3).

    Examples
    --------
    For fields grad_u

    >>> grad = get_gradient(msh, coef, [u, v, w])
    """
    number_of_fields = len(field_list)

    if msh.lz == 1:

        sys.exit("Gradient calculation is not implemented for 2D meshes")

    else:

        grad = np.zeros((msh.nelv, msh.lz, msh.ly, msh.lx, number_of_fields, 3))

        for field in range(0, number_of_fields):

            grad[:, :, :, :, field, 0] = coef.dudxyz(
                field_list[field], coef.drdx, coef.dsdx, coef.dtdx
            )  # dfdx
            grad[:, :, :, :, field, 1] = coef.dudxyz(
                field_list[field], coef.drdy, coef.dsdy, coef.dtdy
            )  # dfdy
            grad[:, :, :, :, field, 2] = coef.dudxyz(
                field_list[field], coef.drdz, coef.dsdz, coef.dtdz
            )  # dfdz

    return grad


def get_strain_tensor(grad_u, msh):
    """
    Calculate the symetric part of the gradient.

    Calculate the symetric part of the gradient tensor, defined as:

    .. math::

        1/2 (G(u) + G(u)^T).

    Typically known in fluids as the strain tensor.

    Another notation is: :math:`1/2 (\partial{u_i}/\partial{x}_j + \partial{u}_j/\partial{x}_i)`.

    Parameters
    ----------
    grad_u : ndarray
        Gradient of a field. A 6D ndarray with the shape (nelv, lz, ly, lx, 3, 3) for 3d data.
    msh : Mesh
        A mesh object.

    Returns
    -------
    ndarray
        A 6D ndarrays with the shape (nelv, lz, ly, lx, 3, 3).

    Examples
    --------
    For grad

    >>> sij = get_strain_tensor(grad, msh)

    Useful for postprocessing
    """

    sij = np.zeros((msh.nelv, msh.lz, msh.ly, msh.lx, 3, 3))

    for e in range(0, msh.nelv):
        for k in range(0, msh.lz):
            for j in range(0, msh.ly):
                for i in range(0, msh.lx):
                    sij[e, k, j, i, :, :] = (1 / 2) * (
                        grad_u[e, k, j, i, :, :] + grad_u[e, k, j, i, :, :].T
                    )

    return sij


def get_angular_rotation_tensor(grad_u, msh):
    """
    Calculate the antisymetric part of the gradient.

    Calculate the antisymetric part of the gradient tensor, defined as:

    .. math::

        1/2 (G(u) - G(u)^T).

    Typically known in fluids as the angular rotation tensor.

    Another notation is: :math:`1/2 (\partial{u_i}/\partial{x}_j - \partial{u}_j/\partial{x}_i)`.

    Parameters
    ----------
    grad_u : ndarray
        Gradient of a field. A 6D ndarray with the shape (nelv, lz, ly, lx, 3, 3) for 3d data.

    msh : Mesh
        A mesh object.

    Returns
    -------
    ndarray
        A 6D ndarrays with the shape (nelv, lz, ly, lx, 3, 3).

    Examples
    --------
    For grad

    >>> aij = get_angular_rotation_tensor(grad, msh)

    Useful for postprocessing
    """
    aij = np.zeros((msh.nelv, msh.lz, msh.ly, msh.lx, 3, 3))

    for e in range(0, msh.nelv):
        for k in range(0, msh.lz):
            for j in range(0, msh.ly):
                for i in range(0, msh.lx):
                    aij[e, k, j, i, :, :] = (1 / 2) * (
                        grad_u[e, k, j, i, :, :] - grad_u[e, k, j, i, :, :].T
                    )

    return aij


def write_fld_file_from_list(fname, comm, msh, field_list=None):
    """
    Write fld file from a field list, each field written as a scalar.

    This function writes a fld from a field list in the scalar positions.

    Parameters
    ----------
    fname : str
        Name of the file to be written.
    comm : Comm
        A communicator object.
    msh : Mesh
        A mesh object.
    field_list : list
        A list of the fields to be written to the file.
         (Default value = None).

    Examples
    --------
    Having defined ndarrays u, v, w of shape (nelv, lz, ly, lx). To write them as scalars in the file:

    >>> write_fld_file_from_list("field0.f00001", comm, msh, [u, v, w])

    To write them in positions 0, 1, 2 of the vel keyword in the file, you should not use this function, and instead
    create a empty field object and update its metadata with the correct positions.

    >>> out_fld = Field(comm)
    >>> out_fld.fields["vel"].append(u)
    >>> out_fld.fields["vel"].append(v)
    >>> out_fld.fields["vel"].append(w)
    >>> out_fld.update_vars()
    >>> out_data = create_hexadata_from_msh_fld(msh=msh, fld=out_fld)
    >>> pwritenek("field0.f00001", out_data, comm)

    This function just wraps these commands and assumes they will be in the scalar keyword.
    """
    number_of_fields = len(field_list)

    ## Create an empty field and update its metadata
    out_fld = Field(comm)
    for field in range(0, number_of_fields):
        out_fld.fields["scal"].append(field_list[field])
    out_fld.update_vars()

    ## Create the hexadata to write out
    out_data = create_hexadata_from_msh_fld(msh=msh, fld=out_fld)

    ## Write out a file
    if comm.Get_rank() == 0:
        print("Writing file: " + fname)
    pwritenek(fname, out_data, comm)


def write_fld_subdomain_from_list(
    fname, comm, msh, field_list=None, subdomain=[], p=None
):
    """
    Write a subdomain and p-refine of the sem mesh into an fld file from a field list.

    This function writes a fld from a field list in the scalar positions.
    In a subdomain specified by the list of list subdomain.

    If p is not None, the mesh is refined/coarsened.

    Parameters
    ----------
    fname : str
        Name of the file to be written.
    comm : Comm
        A communicator object.
    msh : Mesh
        A mesh object.
    field_list : list
        A list of the fields to be written to the file.
        (Default value = None).
    subdomain : list
        A list of lists with the subdomain to be written.
        the format is: subdomain = [[x_min, x_max],[y_min, y_max],[z_min, z_max]].
        (Default value = []).
    p : int, optional
        Polynomial degree of the new mesh. If None, the mesh is not refined/coarsened.
        (Default value = None).
    """
    number_of_fields = len(field_list)
    # Decide if my rank should write data
    my_rank_writes = 1
    # write_these_e = np.where(msh.x == msh.x)[0]

    if subdomain is not []:

        # Find the which elements are contained in the subdomain
        bbox = get_bbox_from_coordinates(msh.x, msh.y, msh.z)
        bbox_centroids, _ = get_bbox_centroids_and_max_dist(bbox)

        condition_one = bbox_centroids[:, 0] > subdomain[0][0]
        condition_two = bbox_centroids[:, 1] > subdomain[1][0]
        condition_tree = bbox_centroids[:, 2] > subdomain[2][0]
        condition_four = bbox_centroids[:, 0] < subdomain[0][1]
        condition_five = bbox_centroids[:, 1] < subdomain[1][1]
        condition_six = bbox_centroids[:, 2] < subdomain[2][1]

        write_these_e = np.where(
            np.all(
                [
                    condition_one,
                    condition_two,
                    condition_tree,
                    condition_four,
                    condition_five,
                    condition_six,
                ],
                axis=0,
            )
        )[0]

        # Check if my rank should write data
        if write_these_e.size == 0:
            my_rank_writes = 0

    # Create communicator for writing
    write_comm = comm.Split(color=my_rank_writes, key=comm.Get_rank())

    if my_rank_writes == 1:

        x_sub = msh.x[write_these_e, :, :, :]
        y_sub = msh.y[write_these_e, :, :, :]
        z_sub = msh.z[write_these_e, :, :, :]
        msh_sub = Mesh(write_comm, x=x_sub, y=y_sub, z=z_sub)

        field_list_sub = []
        for field in range(0, number_of_fields):
            field_list_sub.append(field_list[field][write_these_e, :, :, :])

        # Refine the order of the mesh if needed:
        if not isinstance(p, NoneType):
            pref = PRefiner(n_old=msh_sub.lx, n_new=p)

            # Get the new mesh
            msh_sub = pref.create_refined_mesh(write_comm, msh=msh_sub)

            # Get the new fields
            for field in range(0, number_of_fields):
                field_list_sub[field] = pref.interpolate_from_field_list(
                    write_comm, [field_list_sub[field]]
                )[0]

        ## Create an empty field and update its metadata
        out_fld = Field(write_comm)
        for field in range(0, number_of_fields):
            out_fld.fields["scal"].append(field_list_sub[field])
        out_fld.update_vars()

        ## Create the hexadata to write out
        out_data = create_hexadata_from_msh_fld(msh=msh_sub, fld=out_fld)

        ## Write out a file
        if write_comm.Get_rank() == 0:
            print("Writing file: " + fname)
        pwritenek(fname, out_data, write_comm)

    comm.Barrier()
    write_comm.Free()

    return

def extrude_2d_sem_mesh(comm, lz : int = 1, msh : Mesh = None, fld: Union[Field, FieldRegistry] =None, point_dist : np.ndarray = None):
    """
    Extrude a 2D SEM mesh to 3D.

    Extrude a 2D SEM mesh to 3D by replicating the mesh and fields in the z direction.

    Parameters
    ----------
    comm : Comm
        A communicator object.
    lz : int
        Number of layers in the z direction.
    msh : Mesh
        A mesh object.
    
    fld : Field or FieldRegistry
        A field object.

    point_dist : ndarray
        Array with the z coordinates of the new mesh.
        Defaults to GLL points from -1 to 1.
    
    Returns
    -------
    Mesh
        A 3D mesh object.
    FieldRegistry
        A 3D field object.
    
    """

    if isinstance(msh, Mesh):
            
        if isinstance(point_dist, type(None)):
        
            x_, _ = GLL_pwts(lz)
            point_dist = np.flip(x_)

        x_ext = np.tile(msh.x, (1, lz, 1, 1))
        y_ext = np.tile(msh.y, (1, lz, 1, 1))
        z_ext = np.tile(msh.z, (1, lz, 1, 1))
        z_ext[:, :, :, :] = point_dist.reshape((1, lz, 1, 1))

        msh_ext = Mesh(
            comm, create_connectivity=msh.create_connectivity_bool, x=x_ext, y=y_ext, z=z_ext
        )
    
    if isinstance(fld, FieldRegistry):
        
        fld_ext = FieldRegistry(comm)

        for key in fld.registry.keys():
            field_ = np.tile(fld.registry[key], (1, lz, 1, 1))
            fld_ext.add_field(comm, field_name=key, field=field_.copy(), dtype=field_.dtype)

    elif isinstance(fld, Field):

        fld_ext = FieldRegistry(comm)

        for key in fld.fields.keys():
            for i in range(len(fld.fields[key])):
                field_ = np.tile(fld.fields[key][i], (1, lz, 1, 1))
                fld_ext.fields[key].append(field_.copy())

        fld_ext.t = fld.t
        fld_ext.update_vars()
    
    

    if not isinstance(msh, type(None)) and not isinstance(fld, type(None)):
        return msh_ext, fld_ext
    elif not isinstance(msh, type(None)):
        return msh_ext
    elif not isinstance(fld, type(None)):
        return fld_ext