import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Import required modules
from mpi4py import MPI #equivalent to the use of MPI_init() in C
import matplotlib.pyplot as plt
import numpy as np
import json
# Get mpi info
comm = MPI.COMM_WORLD

from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite

def msh_spacing():

    # Read the inputs file
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)

    sem_mesh_fname = inputs['spectral_element_mesh_fname']
    sem_mesh_fname = "../../examples/data/sem_data/instantaneous/cylinder_rbc_nelv_600/field0.f00801"
    sem_dtype_str = inputs.get('spectral_element_mesh_type_in_memory', 'single')
    out_fname = inputs.get('msh_check_output_fname', 'msh_check0.f00001')

    if sem_dtype_str == 'single':
        sem_dtype = np.single
    elif sem_dtype_str == 'double':
        sem_dtype = np.double
    else:
        raise ValueError(f"Invalid spectral element mesh data type: {sem_dtype_str}")

    # Read the mesh
    msh = Mesh(comm)
    pynekread(sem_mesh_fname, comm, data_dtype=sem_dtype, msh=msh)
    coef = Coef(msh, comm)

    # Get the physical space change wrt reference coordinates.
    dxdr = coef.dudrst(msh.x, direction="r")
    dxds = coef.dudrst(msh.x, direction="s")
    dxdt = coef.dudrst(msh.x, direction="t")

    dydr = coef.dudrst(msh.y, direction="r")
    dyds = coef.dudrst(msh.y, direction="s")
    dydt = coef.dudrst(msh.y, direction="t")

    dzdr = coef.dudrst(msh.z, direction="r")
    dzds = coef.dudrst(msh.z, direction="s")
    dzdt = coef.dudrst(msh.z, direction="t")

    # Compute the difference between reference coordinates
    x_diff = np.zeros_like(coef.x) 
    x_diff[0] = (coef.x[1] - coef.x[0]) * 0.5 
    for i in range(1, msh.lx - 1):
        x_diff[i] = (coef.x[i + 1] - coef.x[i - 1])*0.5
    x_diff[-1] = (coef.x[-1] - coef.x[-2]) * 0.5
    r_diff = np.zeros((1, msh.lz, msh.ly, msh.lx), dtype=sem_dtype)
    s_diff = np.zeros((1, msh.lz, msh.ly, msh.lx), dtype=sem_dtype)
    t_diff = np.zeros((1, msh.lz, msh.ly, msh.lx), dtype=sem_dtype)
    for k in range(msh.lz):
        for j in range(msh.ly):
            for i in range(msh.lx):
                r_diff[0, k, j, i] = x_diff[i]
                s_diff[0, k, j, i] = x_diff[j]
                t_diff[0, k, j, i] = x_diff[k]

    # Then get the actual spacing of the mesh
    ## For dx, we get the magnitude of changes in the r, s, t directions
    dx = np.sqrt((dxdr * r_diff) ** 2 + (dxds * s_diff) ** 2 + (dxdt * t_diff) ** 2 )
    dy = np.sqrt((dydr * r_diff) ** 2 + (dyds * s_diff) ** 2 + (dydt * t_diff) ** 2 )
    dz = np.sqrt((dzdr * r_diff) ** 2 + (dzds * s_diff) ** 2 + (dzdt * t_diff) ** 2 )

    fld = FieldRegistry(comm)
    fld.add_field(comm, field_name="dx", field=dx, dtype=sem_dtype)
    fld.add_field(comm, field_name="dy", field=dy, dtype=sem_dtype)
    fld.add_field(comm, field_name="dz", field=dz, dtype=sem_dtype)
    
    pynekwrite(out_fname, comm, msh = msh, fld=fld)

    return dx, dy, dz

def calculate_mean_dissipations():
    
    # Read the inputs file
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)

    sem_mesh_fname = inputs['spectral_element_mesh_fname']
    sem_mesh_fname = "../../examples/data/sem_data/instantaneous/cylinder_rbc_nelv_600/field0.f00801"
    sem_dtype_str = inputs.get('spectral_element_mesh_type_in_memory', 'single')
    out_fname = inputs.get('msh_check_output_fname', 'msh_check0.f00001')
    out_fname = f"dissipation_{out_fname}"
    file_index_fname = inputs.get("file_index", None)
    calculate_dissipation = inputs.get("calculate_dissipation", True)
    dissipation_keys = inputs.get("dissipation_keys", [])
    pr = inputs.get("pr")
    ra = inputs.get("ra")

    if file_index_fname is None:
        raise ValueError("The file index must be provided in the inputs.json file")

    if calculate_dissipation is False and dissipation_keys == []:
        raise ValueError("Either calculate_dissipation must be True or dissipation_keys must be provided")

    if sem_dtype_str == 'single':
        sem_dtype = np.single
    elif sem_dtype_str == 'double':
        sem_dtype = np.double
    else:
        raise ValueError(f"Invalid spectral element mesh data type: {sem_dtype_str}")

    with open(file_index_fname, 'r') as f:
        file_index = json.load(f)
    file_sequence = []
    for key in file_index.keys():
        try:
            int_key = int(key)
        except ValueError:
            continue
        file_sequence.append(file_index[key]['path'])  

    # Read the mesh
    msh = Mesh(comm)
    pynekread(sem_mesh_fname, comm, data_dtype=sem_dtype, msh=msh)
    coef = Coef(msh, comm)

    eps_t = np.zeros_like(msh.x, dtype=sem_dtype)
    eps_k = np.zeros_like(msh.x, dtype=sem_dtype)

    fld = FieldRegistry(comm)

    for fidx, fname in enumerate(file_sequence):

        if calculate_dissipation:
            # Read the field
            pynekread(fname, comm, data_dtype=sem_dtype, fld=fld)

            u = fld.registry['u']
            v = fld.registry['v']
            w = fld.registry['w']
            t = fld.registry['t']

            mu = np.sqrt(pr/ra)
            lamb = 1 / np.sqrt(ra*pr)

            # Dissipation of kinetic energy
            #epsv=0.5*(du_i/dx_j+du_j/dx_i)**2*sqrt(Pr/Ra)
            #epsv = Sij * Sij * 2 * mu
            dudx = coef.dudxyz(u, coef.drdx, coef.dsdx, coef.dtdx)
            dudy = coef.dudxyz(u, coef.drdy, coef.dsdy, coef.dtdy)
            dudz = coef.dudxyz(u, coef.drdz, coef.dsdz, coef.dtdz)
            dvdx = coef.dudxyz(v, coef.drdx, coef.dsdx, coef.dtdx)
            dvdy = coef.dudxyz(v, coef.drdy, coef.dsdy, coef.dtdy)
            dvdz = coef.dudxyz(v, coef.drdz, coef.dsdz, coef.dtdz)
            dwdx = coef.dudxyz(w, coef.drdx, coef.dsdx, coef.dtdx)
            dwdy = coef.dudxyz(w, coef.drdy, coef.dsdy, coef.dtdy)
            dwdz = coef.dudxyz(w, coef.drdz, coef.dsdz, coef.dtdz)
            sij_sq = np.zeros_like(msh.x, dtype=sem_dtype)
            sij_sq += (dudx + dudx)**2 + (dudy + dvdx)**2 + (dudz + dwdx)**2
            sij_sq += (dvdx + dudy)**2 + (dvdy + dvdy)**2 + (dvdz + dwdy)**2
            sij_sq += (dwdx + dudz)**2 + (dwdy + dvdz)**2 + (dwdz + dwdz)**2
            sij_sq *= (0.5)**2
            eps_k_ = sij_sq * 2 * mu

            # Dissipation of thermal energy
            dtdx = coef.dudxyz(t, coef.drdx, coef.dsdx, coef.dtdx)
            dtdy = coef.dudxyz(t, coef.drdy, coef.dsdy, coef.dtdy)
            dtdz = coef.dudxyz(t, coef.drdz, coef.dsdz, coef.dtdz)
            eps_t_ = (dtdx**2 + dtdy**2 + dtdz**2) * lamb
            
        else:
            # Read the field
            pynekread(fname, comm, data_dtype=sem_dtype, fld=fld)

            eps_k_ = fld.registry[dissipation_keys[0]]
            eps_t_ = fld.registry[dissipation_keys[1]]

        # Update
        eps_k += eps_k_
        eps_t += eps_t_
        fld.clear()

    # Average
    eps_k /= len(file_sequence)
    eps_t /= len(file_sequence)
    
    return eps_k, eps_t

def scale_mesh_kolmogorov(dx, dy, dz, eps_k, eps_t):

    # Read the inputs file
    with open('inputs.json', 'r') as f:
        inputs = json.load(f)

    sem_mesh_fname = inputs['spectral_element_mesh_fname']
    sem_mesh_fname = "../../examples/data/sem_data/instantaneous/cylinder_rbc_nelv_600/field0.f00801"
    sem_dtype_str = inputs.get('spectral_element_mesh_type_in_memory', 'single')
    out_fname = inputs.get('msh_check_output_fname', 'msh_check0.f00001')
    out_fname = f"kolmogorov_{out_fname}"
    pr = inputs.get("pr")
    ra = inputs.get("ra")
    if sem_dtype_str == 'single':
        sem_dtype = np.single
    elif sem_dtype_str == 'double':
        sem_dtype = np.double
    else:
        raise ValueError(f"Invalid spectral element mesh data type: {sem_dtype_str}")

    # Read the mesh
    msh = Mesh(comm)
    pynekread(sem_mesh_fname, comm, data_dtype=sem_dtype, msh=msh)
    coef = Coef(msh, comm)

    # Get kolmogorov length scale
    mu = np.sqrt(pr/ra)
    lamb = 1 / np.sqrt(ra*pr)

    eps_k_global_mean = coef.glsum(eps_k * coef.B, comm, dtype=sem_dtype) / coef.glsum(coef.B, comm, dtype=sem_dtype)
    eps_t_global_mean = coef.glsum(eps_t * coef.B, comm, dtype=sem_dtype) / coef.glsum(coef.B, comm, dtype=sem_dtype)
    eps_k_local_mean = np.sum(eps_k * coef.B, axis= (1,2,3)) / np.sum(coef.B, axis= (1,2,3))

    eta_global = (mu**3 / eps_k_global_mean)**(1/4)
    eta_local = (mu**3 / eps_k_local_mean)**(1/4)

    dx_local = dx / eta_local.reshape(-1,1,1,1)
    dy_local = dy / eta_local.reshape(-1,1,1,1)
    dz_local = dz / eta_local.reshape(-1,1,1,1)
    dx_local_avg = np.sum(dx_local*coef.B, axis = (1,2,3), keepdims=True) / np.sum(coef.B, axis = (1,2,3), keepdims=True) * np.ones_like(dx_local)
    dy_local_avg = np.sum(dy_local*coef.B, axis = (1,2,3), keepdims=True) / np.sum(coef.B, axis = (1,2,3), keepdims=True) * np.ones_like(dy_local)
    dz_local_avg = np.sum(dz_local*coef.B, axis = (1,2,3), keepdims=True) / np.sum(coef.B, axis = (1,2,3), keepdims=True) * np.ones_like(dz_local)

    fld = FieldRegistry(comm)
    fld.add_field(comm, field_name="dx_local", field=dx_local, dtype=sem_dtype)
    fld.add_field(comm, field_name="dy_local", field=dy_local, dtype=sem_dtype)
    fld.add_field(comm, field_name="dz_local", field=dz_local, dtype=sem_dtype)
    fld.add_field(comm, field_name="dx_local_avg", field=dx_local_avg, dtype=sem_dtype)
    fld.add_field(comm, field_name="dy_local_avg", field=dy_local_avg, dtype=sem_dtype)
    fld.add_field(comm, field_name="dz_local_avg", field=dz_local_avg, dtype=sem_dtype)
    pynekwrite(out_fname, comm, msh=msh, fld=fld)

    # Write some data to verify
    nu_eps_k = 1 + eps_k_global_mean/((mu**3)*ra/(pr**2))
    nu_eps_t = 1 / lamb * eps_t_global_mean
    max_dx_local_avg = comm.allreduce(np.max(dx_local_avg), op=MPI.MAX)
    max_dy_local_avg = comm.allreduce(np.max(dy_local_avg), op=MPI.MAX)
    max_dz_local_avg = comm.allreduce(np.max(dz_local_avg), op=MPI.MAX)

    if comm.Get_rank() == 0:
        with open("kolmogorov_data.json", 'w') as f:
            data = {
                "eta_global": eta_global,
                "max_dx_kolmogorov": max_dx_local_avg,
                "max_dy_kolmogorov": max_dy_local_avg,
                "max_dz_kolmogorov": max_dz_local_avg,
                "nu_eps_k": nu_eps_k,
                "nu_eps_t": nu_eps_t
            }
            json.dump(data, f, indent=4)

# ============
# Main program
# ============

dx, dy, dz = msh_spacing()

eps_k, eps_t = calculate_mean_dissipations()

scale_mesh_kolmogorov(dx, dy, dz, eps_k, eps_t)
