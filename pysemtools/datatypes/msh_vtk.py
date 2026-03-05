""" Module that contains vtk msh class, which contains relevant data on the domain"""

__all__ = ['VTKMesh']

import numpy as np
from mpi4py import MPI
    
VTK_HEXAHEDRON = 12
VTK_HEXAHEDRON_CELL_POINTS = 8

class VTKMesh:
    """ Class that contains the mesh data in a vtk friendly format
    
    Helper to build connectivity and offsets etc.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator
    x : np.ndarray
        The x coordinates of the mesh points
    y : np.ndarray
        The y coordinates of the mesh points
    z : np.ndarray
        The z coordinates of the mesh points
    cell_type : str, optional
        The type of cells in the mesh, by default "hex". Only "hex" is currently supported.
    global_connectivity : bool, optional
        Whether to use global connectivity or local connectivity in parallel. By default True, but if only one rank is used, it is set to False since the connectivity is already global in that case.
    distributed_axis : int, optional
        The axis along which the mesh is distributed, by default 0 (Only zero allowed now) 
    """

    def __init__(self, comm: MPI.Comm, x: np.ndarray, y: np.ndarray, z: np.ndarray, cell_type: str = "hex", global_connectivity: bool = True, distributed_axis: int = 0):

        self.comm = comm
        self.x = x
        self.y = y
        self.z = z
        self.ndim = x.ndim
        self.cell_type = cell_type
        self.distributed_axis = distributed_axis

        # Outputs
        self.Points = None
        self.Connectivity = None
        self.Offsets = None
        self.Types = None
        self.Sizes = None
        self.NumberOfPoints = None
        self.NumberOfCells = None
        self.NumberOfConnectivityIds = None
        self.shape = None
        self.data = None
        
        if self.distributed_axis != 0:
            raise NotImplementedError("Only distribution along the first axis (0) is currently implemented for VTK meshes")

        if self.comm.Get_size() > 1:
            self.global_conn = global_connectivity
        else:
            self.global_conn = False # In one rank, the serial connectivity is global

        if self.cell_type != "hex":
            raise NotImplementedError("Only hexahedral meshes are currently supported for VTK meshes")
        elif self.cell_type == "hex":
            if self.ndim == 3:
                if not self.global_conn:
                    self.data = set_up_hex_vtk_mesh_local(x, y, z)
                else:
                    self.data = set_up_hex_vtk_mesh_global(self.comm, x, y, z, distributed_axis)
            elif x.ndim == 4:
                if not self.global_conn:
                    self.data = set_up_hex_vtk_mesh_sem_local(x, y, z)
                else:
                    self.data = set_up_hex_vtk_mesh_sem_global(self.comm, x, y, z)
            else:
                raise ValueError("For hex meshes, x, y, and z should be either 3D arrays (for regular hex meshes) or 4D arrays (for SEM-like hex meshes)")

        # Assign attributes
        self.Points = self.data["Points"]
        self.Connectivity = self.data["Connectivity"]
        self.Offsets = self.data["Offsets"]
        self.Types = self.data["Types"]
        self.Sizes = self.data["Sizes"]
        self.NumberOfPoints = self.data["NumberOfPoints"]
        self.NumberOfCells = self.data["NumberOfCells"]
        self.NumberOfConnectivityIds = self.data["NumberOfConnectivityIds"]
        self.shape = self.data["shape"]

def set_up_hex_vtk_mesh_local(X : np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """Set up the mesh in serial"""

    if X.ndim != 3 or Y.ndim != 3 or Z.ndim != 3:
        raise ValueError("X, Y, and Z should be 3D arrays")

    # Get the point list        
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    n_pts_x = X.shape[0]
    n_pts_y = X.shape[1]
    n_pts_z = X.shape[2]
    n_cell_x = n_pts_x - 1
    n_cell_y = n_pts_y - 1
    n_cell_z = n_pts_z - 1
    n_cell_local = n_cell_x * n_cell_y * n_cell_z

    types = np.full(n_cell_local, VTK_HEXAHEDRON, dtype=np.uint8)
    sizes = np.full(n_cell_local, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.uint8)

    connectivity = []
    for i in range(n_cell_x):
        for j in range(n_cell_y):
            for k in range(n_cell_z):
                v0 = ijk_to_id(i,   j,   k,   n_pts_y, n_pts_z)
                v1 = ijk_to_id(i+1, j,   k,   n_pts_y, n_pts_z)
                v2 = ijk_to_id(i+1, j+1, k,   n_pts_y, n_pts_z)
                v3 = ijk_to_id(i,   j+1, k,   n_pts_y, n_pts_z)
                v4 = ijk_to_id(i,   j,   k+1, n_pts_y, n_pts_z)
                v5 = ijk_to_id(i+1, j,   k+1, n_pts_y, n_pts_z)
                v6 = ijk_to_id(i+1, j+1, k+1, n_pts_y, n_pts_z)
                v7 = ijk_to_id(i,   j+1, k+1, n_pts_y, n_pts_z)
                connectivity.extend([v0, v1, v2, v3, v4, v5, v6, v7])

    connectivity = np.asarray(connectivity, dtype=np.int64)   # length 8*ncells
    offsets = np.arange(0, VTK_HEXAHEDRON_CELL_POINTS*n_cell_local + 1, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.int64)   # length ncells+1

    NumberOfPoints = (points.shape[0],)
    NumberOfCells = (n_cell_local,)
    NumberOfConnectivityIds = (connectivity.size,)

    return {"Points": points, "Connectivity": connectivity, 
            "Offsets": offsets, "Types": types,
            "Sizes": sizes, 
            "NumberOfPoints": NumberOfPoints, 
            "NumberOfCells": NumberOfCells, 
            "NumberOfConnectivityIds": NumberOfConnectivityIds,
            "shape": (n_pts_x, n_pts_y, n_pts_z)}


def set_up_hex_vtk_mesh_sem_local(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """Build vtk needed data for SEM-like meshes - serial"""

    if X.ndim != 4 or Y.ndim != 4 or Z.ndim != 4:
        raise ValueError("X, Y, and Z should be 4D arrays with shape (nelv, nz, ny, nx)")

    nelv, nz, ny, nx = X.shape
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Each element must have at least 2 points in each direction")

    # Get the point list        
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    # Determine the number of points and cells per element
    n_cell_x = nx - 1
    n_cell_y = ny - 1
    n_cell_z = nz - 1
    n_cell_per_elem = n_cell_x * n_cell_y * n_cell_z
    n_cell_local = nelv * n_cell_per_elem

    types = np.full(n_cell_local, VTK_HEXAHEDRON, dtype=np.uint8)
    sizes = np.full(n_cell_local, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.uint8)

    connectivity = []
    for e in range(nelv):
        for kz in range(n_cell_z):
            for jy in range(n_cell_y):
                for ix in range(n_cell_x):
                    v0 = elem_kji_to_id(e, ix,   jy,   kz,   nz, ny, nx)
                    v1 = elem_kji_to_id(e, ix+1, jy,   kz,   nz, ny, nx)
                    v2 = elem_kji_to_id(e, ix+1, jy+1, kz,   nz, ny, nx)
                    v3 = elem_kji_to_id(e, ix,   jy+1, kz,   nz, ny, nx)
                    v4 = elem_kji_to_id(e, ix,   jy,   kz+1, nz, ny, nx)
                    v5 = elem_kji_to_id(e, ix+1, jy,   kz+1, nz, ny, nx)
                    v6 = elem_kji_to_id(e, ix+1, jy+1, kz+1, nz, ny, nx)
                    v7 = elem_kji_to_id(e, ix,   jy+1, kz+1, nz, ny, nx)
                    connectivity.extend([v0, v1, v2, v3, v4, v5, v6, v7])

    connectivity = np.asarray(connectivity, dtype=np.int64)   # length 8*ncells
    offsets = np.arange(0, VTK_HEXAHEDRON_CELL_POINTS*n_cell_local + 1, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.int64)   # length ncells+1
    
    NumberOfPoints = (points.shape[0],)
    NumberOfCells = (n_cell_local,)
    NumberOfConnectivityIds = (connectivity.size,)

    return {
        "Points": points,
        "Connectivity": connectivity,
        "Offsets": offsets,
        "Types": types,
        "Sizes": sizes,
        "NumberOfPoints": NumberOfPoints,
        "NumberOfCells": NumberOfCells,
        "NumberOfConnectivityIds": NumberOfConnectivityIds,
        "shape": (nelv, nz, ny, nx),
    }

def set_up_hex_vtk_mesh_global(comm: MPI.Comm, X : np.ndarray, Y: np.ndarray, Z: np.ndarray, distributed_axis: int):
    """Set up the mesh in parallel"""
    
    if X.ndim != 3 or Y.ndim != 3 or Z.ndim != 3:
        raise ValueError("X, Y, and Z should be 3D arrays")
    
    if distributed_axis != 0:
        raise NotImplementedError("Distributed axis other than 0 is not implemented for the set_up_hex_vtk_mesh_parallel function")

    # Get the point list           
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    # Determine the number of points and cells locally 
    n_pts_x = X.shape[0]
    n_pts_y = X.shape[1]
    n_pts_z = X.shape[2]
    
    # For the distributed axis, the number of cells is n_pts_total - 1. The last rank is the one that "loses" one point
    # This happenes because we do not have overlapping boundaries across ranks. That would make it easier.
    n_cell_x = n_pts_x - 1 if comm.Get_rank() == comm.Get_size() - 1 else n_pts_x
    n_cell_y = n_pts_y - 1
    n_cell_z = n_pts_z - 1
    n_cell_local = n_cell_x * n_cell_y * n_cell_z

    # Determine global and offsets for nells
    n_pts_global = comm.allreduce(points.shape[0], op=MPI.SUM)
    n_pts_offset = comm.scan(n_pts_x) - n_pts_x
    n_pts_x_global = comm.allreduce(n_pts_x, op=MPI.SUM)
    n_cell_global = comm.allreduce(n_cell_local, op=MPI.SUM)

    # Set up array with types of cells
    types = np.full(n_cell_local, VTK_HEXAHEDRON, dtype=np.uint8)
    sizes = np.full(n_cell_local, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.uint8)

    # Set up connectivity and offsets 
    connectivity = []
    for i in range(n_cell_x):
        for j in range(n_cell_y):
            for k in range(n_cell_z):
                v0 = ijk_to_id_mpi(i,   j,   k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v1 = ijk_to_id_mpi(i+1, j,   k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v2 = ijk_to_id_mpi(i+1, j+1, k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v3 = ijk_to_id_mpi(i,   j+1, k,   n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v4 = ijk_to_id_mpi(i,   j,   k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v5 = ijk_to_id_mpi(i+1, j,   k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v6 = ijk_to_id_mpi(i+1, j+1, k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                v7 = ijk_to_id_mpi(i,   j+1, k+1, n_pts_y, n_pts_z, distributed_axis, n_pts_offset)
                connectivity.extend([v0, v1, v2, v3, v4, v5, v6, v7])

    connectivity = np.asarray(connectivity, dtype=np.int64)   # length 8*ncells
    conn_start = comm.scan(connectivity.size) - connectivity.size
    offsets_local = (np.arange(0, VTK_HEXAHEDRON_CELL_POINTS*n_cell_local + 1, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.int64) + conn_start)
    offsets = offsets_local[:-1]   # length = n_cell_total - Total offsets need to be n_cell_total + 1, but we fix later

    NumberOfPoints = (n_pts_global,)
    NumberOfCells = (n_cell_global,)
    NumberOfConnectivityIds = (comm.allreduce(connectivity.size, op=MPI.SUM),)

    return {"Points": points, "Connectivity": connectivity, 
            "Offsets": offsets, "Types": types,
            "Sizes": sizes,
            "NumberOfPoints": NumberOfPoints, 
            "NumberOfCells": NumberOfCells, 
            "NumberOfConnectivityIds": NumberOfConnectivityIds,
            "shape": (n_pts_x_global, n_pts_y, n_pts_z)}

def set_up_hex_vtk_mesh_sem_global(comm: MPI.Comm, X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """Build vtk needed data for SEM-like meshes - Parallel"""

    if X.ndim != 4 or Y.ndim != 4 or Z.ndim != 4:
        raise ValueError("X, Y, and Z should be 4D arrays with shape (nelv, nz, ny, nx)")

    nelv, nz, ny, nx = X.shape
    if nx < 2 or ny < 2 or nz < 2:
        raise ValueError("Each element must have at least 2 points in each direction")

    # Get the point list        
    points = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T

    # Determine the number of points and cells per element
    n_cell_x = nx - 1
    n_cell_y = ny - 1
    n_cell_z = nz - 1
    n_cell_per_elem = n_cell_x * n_cell_y * n_cell_z
    n_cell_local = nelv * n_cell_per_elem
    nelv_offset = comm.scan(nelv) - nelv
    
    # Determine global and offsets for nells
    n_pts_global = comm.allreduce(points.shape[0], op=MPI.SUM)
    nelv_global = comm.allreduce(nelv, op=MPI.SUM)
    n_cell_global = comm.allreduce(n_cell_local, op=MPI.SUM)

    types = np.full(n_cell_local, VTK_HEXAHEDRON, dtype=np.uint8)
    sizes = np.full(n_cell_local, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.uint8)

    connectivity = []
    for e in range(nelv):
        for kz in range(n_cell_z):
            for jy in range(n_cell_y):
                for ix in range(n_cell_x):
                    v0 = elem_kji_to_id(e + nelv_offset, ix,   jy,   kz,   nz, ny, nx)
                    v1 = elem_kji_to_id(e + nelv_offset, ix+1, jy,   kz,   nz, ny, nx)
                    v2 = elem_kji_to_id(e + nelv_offset, ix+1, jy+1, kz,   nz, ny, nx)
                    v3 = elem_kji_to_id(e + nelv_offset, ix,   jy+1, kz,   nz, ny, nx)
                    v4 = elem_kji_to_id(e + nelv_offset, ix,   jy,   kz+1, nz, ny, nx)
                    v5 = elem_kji_to_id(e + nelv_offset, ix+1, jy,   kz+1, nz, ny, nx)
                    v6 = elem_kji_to_id(e + nelv_offset, ix+1, jy+1, kz+1, nz, ny, nx)
                    v7 = elem_kji_to_id(e + nelv_offset, ix,   jy+1, kz+1, nz, ny, nx)
                    connectivity.extend([v0, v1, v2, v3, v4, v5, v6, v7])

    connectivity = np.asarray(connectivity, dtype=np.int64)   # length 8*ncells
    conn_start = comm.scan(connectivity.size) - connectivity.size
    offsets_local = (np.arange(0, VTK_HEXAHEDRON_CELL_POINTS*n_cell_local + 1, VTK_HEXAHEDRON_CELL_POINTS, dtype=np.int64) + conn_start)
    offsets = offsets_local[:-1]   # length = n_cell_total - Total offsets need to be n_cell_total + 1, but we fix later
    
    NumberOfPoints = (n_pts_global,)
    NumberOfCells = (n_cell_global,)
    NumberOfConnectivityIds = (comm.allreduce(connectivity.size, op=MPI.SUM),)

    return {
        "Points": points,
        "Connectivity": connectivity,
        "Offsets": offsets,
        "Types": types,
        "Sizes": sizes,
        "NumberOfPoints": NumberOfPoints,
        "NumberOfCells": NumberOfCells,
        "NumberOfConnectivityIds": NumberOfConnectivityIds,
        "shape": (nelv_global, nz, ny, nx),
    }

def ijk_to_id(i, j, k, ny, nz):
    return i*(ny*nz) + j*nz + k

def elem_kji_to_id(e: int, ix: int, jy: int, kz: int, nz: int, ny: int, nx: int) -> int:
    return e * (nz * ny * nx) + kz * (ny * nx) + jy * nx + ix

def ijk_to_id_mpi(i, j, k, ny, nz, parallel_axis, offset):
    if parallel_axis == 0:
        i = i + offset
        return i*(ny*nz) + j*nz + k
    else:
        raise NotImplementedError("Parallel axis other than 0 is not implemented for the ijk_to_id_mpi function")
