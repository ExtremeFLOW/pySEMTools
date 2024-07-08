import numpy as np
from pymech.core import HexaData
from pymech.neksuite.field import Header
from pymech.core import HexaData
from .msh import msh_c
from .field import field_c
from ..ppymech.neksuite import pwritenek
from ..interpolation.interpolator import (
    get_bbox_from_coordinates,
    get_bbox_centroids_and_max_dist,
)
from ..interpolation.mesh_to_mesh import p_refiner_c
import sys

NoneType = type(None)


def create_hexadata_from_msh_fld(
    msh=None, fld=None, wdsz=4, istep=0, time=0, data_dtype="float64", write_mesh=True
):

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
    data = put_coordinates_in_hexadata_from_msh(data, msh)

    # Include the fields
    for qoi in range(0, vel_fields):
        prefix = "vel"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    for qoi in range(0, pres_fields):
        prefix = "pres"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    for qoi in range(0, temp_fields):
        prefix = "temp"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    for qoi in range(0, scal_fields):
        prefix = "scal"
        data = put_field_in_hexadata(data, fld.fields[prefix][qoi], prefix, qoi)

    return data


def put_coordinates_in_hexadata_from_msh(data, msh):

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    for e in range(0, nelv):
        data.elem[e].pos[0, :, :, :] = msh.x[e, :, :, :]
        data.elem[e].pos[1, :, :, :] = msh.y[e, :, :, :]
        data.elem[e].pos[2, :, :, :] = msh.z[e, :, :, :]

    return data


def put_field_in_hexadata(data, field, prefix, qoi):

    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

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

    return data


def get_gradient(msh, coef, field_list=[]):

    number_of_fields = len(field_list)

    if msh.lz == 1:

        exit("Gradient calculation is not implemented for 2D meshes")

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


def get_strain_tensor(grad_U, msh):
    # Calculate the symetric part of the gradient
    # This calculates 1/2(grad_U + grad_U^T), which is the stress tensor
    # Another form of this is simply 1/2(du_i/dx_j + du_j/dx_i)

    sij = np.zeros((msh.nelv, msh.lz, msh.ly, msh.lx, 3, 3))

    for e in range(0, msh.nelv):
        for k in range(0, msh.lz):
            for j in range(0, msh.ly):
                for i in range(0, msh.lx):
                    sij[e, k, j, i, :, :] = (1 / 2) * (
                        grad_U[e, k, j, i, :, :] + grad_U[e, k, j, i, :, :].T
                    )

    return sij


def get_angular_rotation_tensor(grad_U, msh):
    # Calculate the asymetric part of the gradient
    # This calculates 1/2(grad_U - grad_U^T), which is the stress tensor
    # Another form of this is simply 1/2(du_i/dx_j + du_j/dx_i)

    aij = np.zeros((msh.nelv, msh.lz, msh.ly, msh.lx, 3, 3))

    for e in range(0, msh.nelv):
        for k in range(0, msh.lz):
            for j in range(0, msh.ly):
                for i in range(0, msh.lx):
                    aij[e, k, j, i, :, :] = (1 / 2) * (
                        grad_U[e, k, j, i, :, :] - grad_U[e, k, j, i, :, :].T
                    )

    return aij


def write_fld_file_from_list(fname, comm, msh, field_list=[]):

    number_of_fields = len(field_list)

    ## Create an empty field and update its metadata
    out_fld = field_c(comm)
    for field in range(0, number_of_fields):
        out_fld.fields["scal"].append(field_list[field])
    out_fld.update_vars()

    ## Create the hexadata to write out
    out_data = create_hexadata_from_msh_fld(msh=msh, fld=out_fld)

    ## Write out a file
    if comm.Get_rank() == 0:
        print("Writing file: " + fname)
    pwritenek(fname, out_data, comm)

    return


def write_fld_subdomain_from_list(
    fname, comm, msh, field_list=[], subdomain=[], p=None
):

    number_of_fields = len(field_list)
    # Decide if my rank should write data
    my_rank_writes = 1
    # write_these_e = np.where(msh.x == msh.x)[0]

    if subdomain is not []:

        # Find the which elements are contained in the subdomain
        bbox = get_bbox_from_coordinates(msh.x, msh.y, msh.z)
        bbox_centroids, bbox_max_dist = get_bbox_centroids_and_max_dist(bbox)

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
        msh_sub = msh_c(write_comm, x=x_sub, y=y_sub, z=z_sub)

        field_list_sub = []
        for field in range(0, number_of_fields):
            field_list_sub.append(field_list[field][write_these_e, :, :, :])

        # Refine the order of the mesh if needed:
        if type(p) != NoneType:
            pref = p_refiner_c(n_old=msh_sub.lx, n_new=p)

            # Get the new mesh
            msh_sub = pref.get_new_mesh(write_comm, msh=msh_sub)

            # Get the new fields
            for field in range(0, number_of_fields):
                field_list_sub[field] = pref.interpolate_from_field_list(
                    write_comm, [field_list_sub[field]]
                )[0]

        ## Create an empty field and update its metadata
        out_fld = field_c(write_comm)
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

    return
