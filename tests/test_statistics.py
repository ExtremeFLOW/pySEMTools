# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

import os

# Import general modules
import numpy as np
# Import relevant modules
from pysemtools.io.ppymech.neksuite import pynekread, pynekwrite
from pysemtools.postprocessing.statistics.time_averaging import average_field_files
from pysemtools.postprocessing.file_indexing import index_files_from_folder
from pysemtools.datatypes.field import FieldRegistry, Field
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.msh import Mesh
from pysemtools.postprocessing.statistics.space_averaging import *
from pysemtools.comm.router import Router
from pysemtools.monitoring.logger import Logger

rt = Router(comm)
logger = Logger(comm=comm, module_name="space_average_field_files")
rel_tol = 0.01

def test_time_averaging_1_batches():

    dtype = np.single

    # Read the mesh data
    fname = 'examples/data/rbc0.f00001'    
    # Read the original mesh data
    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype = dtype, msh = msh)

    # Create fiction mean fields
    fld = FieldRegistry(comm)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 10
    u2 = np.ones_like(msh.x) * 20
    u3 = np.ones_like(msh.x) * 30
    s1 = np.ones_like(msh.x) * 40

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 20
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(0).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)
    
    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 13
    u2 = np.ones_like(msh.x) * 23
    u3 = np.ones_like(msh.x) * 33
    s1 = np.ones_like(msh.x) * 43

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 42
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(1).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 17
    u2 = np.ones_like(msh.x) * 27
    u3 = np.ones_like(msh.x) * 37
    s1 = np.ones_like(msh.x) * 47

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 70
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(2).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)


    # Now index the files:
    index_files_from_folder(comm, folder_path=".", run_start_time=0, stat_start_time = 0 )

    # Now average the fields and write them to disk
    average_field_files(comm, field_index_name = "./mean_rbc_index.json", output_folder = "./", output_batch_t_len=70) 


    # Now read the files and see if they are correct
    fld2 = Field(comm)
    pynekread("batch_mean_rbc0.f00000", comm, data_dtype = dtype, fld = fld2)

    mean_u1 = fld2.fields["vel"][0]
    mean_u2 = fld2.fields["vel"][1]
    mean_u3 = fld2.fields["vel"][2]
    mean_s1 = fld2.fields["scal"][0]

    t1 = np.allclose(mean_u1, np.ones_like(msh.x) * (10*20 + 13*22 + 17*28) / 70)
    t2 = np.allclose(mean_u2, np.ones_like(msh.x) * (20*20 + 23*22 + 27*28) / 70)
    t3 = np.allclose(mean_u3, np.ones_like(msh.x) * (30*20 + 33*22 + 37*28) / 70)
    t4 = np.allclose(mean_s1, np.ones_like(msh.x) * (40*20 + 43*22 + 47*28) / 70)

    passed = np.all([t1, t2, t3, t4])

    if comm.Get_rank() == 0:    
        # delete the wirtten files
        os.system("rm ./mean_rbc0.f00000")
        os.system("rm ./mean_rbc0.f00001")
        os.system("rm ./mean_rbc0.f00002")
        os.system("rm ./mean_rbc_index.json")
        os.system("rm ./batches.json")
        os.system("rm ./batch_mean_rbc0.f00000")
    comm.Barrier()

    print(passed)

    assert passed

def test_time_averaging_2_batches():

    dtype = np.single

    # Read the mesh data
    fname = 'examples/data/rbc0.f00001'    
    # Read the original mesh data
    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype = dtype, msh = msh)

    # Create fiction mean fields
    fld = Field(comm)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 10
    u2 = np.ones_like(msh.x) * 20
    u3 = np.ones_like(msh.x) * 30
    s1 = np.ones_like(msh.x) * 40

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 20
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(0).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)
    
    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 13
    u2 = np.ones_like(msh.x) * 23
    u3 = np.ones_like(msh.x) * 33
    s1 = np.ones_like(msh.x) * 43

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 42
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(1).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)

    # Field
    fld.clear()
    u1 = np.ones_like(msh.x) * 17
    u2 = np.ones_like(msh.x) * 27
    u3 = np.ones_like(msh.x) * 37
    s1 = np.ones_like(msh.x) * 47

    fld.fields["vel"].extend([ u1, u2, u3 ])
    fld.fields["scal"].append(s1)
    fld.t = 70
    fld.update_vars()

    # Write it
    out_fname = "mean_" + os.path.basename(fname).split(".")[0][:-1] + "0.f" + str(2).zfill(5)
    pynekwrite(out_fname, comm, msh = msh, fld=fld, wdsz=4, istep=0, write_mesh=True)


    # Now index the files:
    index_files_from_folder(comm, folder_path=".", run_start_time=0, stat_start_time = 0 )

    # Now average the fields and write them to disk
    average_field_files(comm, field_index_name = "./mean_rbc_index.json", output_folder = "./", output_batch_t_len=42) 


    # Now read the files and see if they are correct
    fld2 = Field(comm)
    pynekread("batch_mean_rbc0.f00000", comm, data_dtype = dtype, fld = fld2)

    mean_u1 = fld2.fields["vel"][0]
    mean_u2 = fld2.fields["vel"][1]
    mean_u3 = fld2.fields["vel"][2]
    mean_s1 = fld2.fields["scal"][0]

    t1 = np.allclose(mean_u1, np.ones_like(msh.x) * (10*20 + 13*22) / 42)
    t2 = np.allclose(mean_u2, np.ones_like(msh.x) * (20*20 + 23*22) / 42)
    t3 = np.allclose(mean_u3, np.ones_like(msh.x) * (30*20 + 33*22) / 42)
    t4 = np.allclose(mean_s1, np.ones_like(msh.x) * (40*20 + 43*22) / 42)

    passed1 = np.all([t1, t2, t3, t4])
    
    fld3 = Field(comm)
    pynekread("batch_mean_rbc0.f00001", comm, data_dtype = dtype, fld = fld3)
    
    mean_u1 = fld3.fields["vel"][0]
    mean_u2 = fld3.fields["vel"][1]
    mean_u3 = fld3.fields["vel"][2]
    mean_s1 = fld3.fields["scal"][0]

    t1 = np.allclose(mean_u1, np.ones_like(msh.x) * (17*28) / 28)
    t2 = np.allclose(mean_u2, np.ones_like(msh.x) * (27*28) / 28)
    t3 = np.allclose(mean_u3, np.ones_like(msh.x) * (37*28) / 28)
    t4 = np.allclose(mean_s1, np.ones_like(msh.x) * (47*28) / 28)
    
    passed2 = np.all([t1, t2, t3, t4])

    passed = np.all([passed1, passed2])

    # delete the wirtten files
    if comm.Get_rank() == 0:
        os.system("rm ./mean_rbc0.f00000")
        os.system("rm ./mean_rbc0.f00001")
        os.system("rm ./mean_rbc0.f00002")
        os.system("rm ./mean_rbc_index.json")
        os.system("rm ./batches.json")
        os.system("rm ./batch_mean_rbc0.f00000")
        os.system("rm ./batch_mean_rbc0.f00001")
    comm.Barrier()

    print(passed)

    assert passed

def test_space_averaging():

    homogeneous_dir = "z"    
    dtype = np.single

    # Read the mesh data
    fname = 'examples/data/rbc0.f00001'    
    # Read the original mesh data
    msh = Mesh(comm, create_connectivity=False)
    pynekread(fname, comm, data_dtype = dtype, msh = msh)

    # Initialize coef
    coef = Coef(msh, comm, get_area=False)    
    
    if homogeneous_dir == "z":
        direction = 1
    elif homogeneous_dir == "y":
        direction = 2
    elif homogeneous_dir == "x":
        direction = 3
    else:
        return

    if homogeneous_dir == "z":
        xx = msh.x
        yy = msh.y
        direction = 1
        ax = (2,3)
    elif homogeneous_dir == "y":
        xx = msh.x
        yy = msh.z
        direction = 2
        ax = (1,3)
    elif homogeneous_dir == "x":
        xx = msh.y
        yy = msh.z
        direction = 3
        ax = (1,2)
    else:
        return 
    
    x_bar =  np.min(xx, axis=ax)[:,0]  + (np.max(xx, axis=ax)[:, 0] - np.min(xx, axis=ax)[:, 0])/2
    y_bar =  np.min(yy, axis=ax)[:,0]  + (np.max(yy, axis=ax)[:, 0] - np.min(yy, axis=ax)[:, 0])/2
    centroids = np.zeros((msh.nelv, 2), dtype=dtype)
    centroids[:, 0] = x_bar
    centroids[:, 1] = y_bar

    # Find the unique centroids in the rank
    rank_unique_centroids = np.unique(centroids, axis=0)
    
    # Get el owners
    el_owner = get_el_owner(logger = logger, rank_unique_centroids = rank_unique_centroids, rt = rt, dtype = dtype, rel_tol=rel_tol)

    # Map centroids to unique centroids in this rank
    logger.write("info", f"Mapping the indices of the original elements to those of the unique elements in the specified direction")
    elem_to_unique_map = np.zeros((msh.nelv), dtype=np.int64)
    for i in range(0, msh.nelv):
        elem_to_unique_map[i] = np.where(np.all(np.isclose(rank_unique_centroids, centroids[i, :], rtol=rel_tol), axis=1))[0][0]


    logger.write("info", f"Getting 2D slices")
    # Create a 2D sem arrays to hold averages
    if homogeneous_dir == "z":
        slice_shape = (rank_unique_centroids.shape[0], 1, msh.ly, msh.lx)
        slice_shape_e = (-1, 1, msh.ly, msh.lx)
    elif homogeneous_dir == "y":
        slice_shape = (rank_unique_centroids.shape[0], 1, msh.lz, msh.lx)
        slice_shape_e = (-1, 1, msh.lz, msh.lx)
    elif homogeneous_dir == "x":
        slice_shape = (rank_unique_centroids.shape[0], 1, msh.lz, msh.ly)
        slice_shape_e = (-1, 1, msh.lz, msh.ly)
    x_2d = np.zeros(slice_shape, dtype=dtype)
    y_2d = np.zeros(slice_shape, dtype=dtype)

    # Average the fileds to test
    rank_weights = average_field_in_dir_local(avrg_field = x_2d, field = xx, coef = coef, elem_to_unique_map = elem_to_unique_map, direction = direction)
    global_average, elements_i_own = average_slice_global(avrg_field = x_2d, el_owner = el_owner, router = rt, rank_weights = rank_weights)
    
    _ = average_field_in_dir_local(avrg_field = y_2d, field = yy, coef = coef, elem_to_unique_map = elem_to_unique_map, direction = direction)
    _ = average_slice_global(avrg_field = y_2d, el_owner = el_owner, router = rt, rank_weights = rank_weights)

    logger.write("info", f"Verifying averaging")

    if global_average:
        passed = True
        for e in elements_i_own:
            
            if direction == 1:
                t1 = np.allclose(x_2d[elem_to_unique_map[e], :, :, :], xx[e, 0, :, :])
                t2 = np.allclose(y_2d[elem_to_unique_map[e], :, :, :], yy[e, 0, :, :])
            elif direction == 2:
                t1 = np.allclose(x_2d[elem_to_unique_map[e], :, :, :], xx[e, :, 0, :])
                t2 = np.allclose(y_2d[elem_to_unique_map[e], :, :, :], yy[e, :, 0, :])
            elif direction == 3:
                t1 = np.allclose(x_2d[elem_to_unique_map[e], :, :, :], xx[e, :, :, 0])
                t2 = np.allclose(y_2d[elem_to_unique_map[e], :, :, :], yy[e, :, :, 0])
                 
            passed = np.all([t1, t2])

            if not passed:
                break
        
        if passed:
            logger.write("info", f"Averaging test passed: {passed}")
        else:
            logger.write("error", f"Averaging test passed: {passed}")

    assert passed

test_space_averaging() 
#test_time_averaging_1_batches()
#test_time_averaging_2_batches()
