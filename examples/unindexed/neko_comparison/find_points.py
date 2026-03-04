# Import the data types
from mpi4py import MPI #equivalent to the use of MPI_init() in C
from pysemtools.io.ppymech.neksuite import preadnek, pwritenek
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field
from pysemtools.interpolation.probes import Probes
import pysemtools.interpolation.utils as interp_utils
import pysemtools.interpolation.pointclouds as pcs
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

probes = Probes(comm, filename="inputs.json", point_interpolator_type='multiple_point_legendre_numpy', max_pts = 256)

number_of_files = (probes.params_file["case"]["interpolate_fields"]["number_of_files"])
for i in range(0, number_of_files):
    fld_data = probes.read_fld_file(i, comm)
    probes.interpolate_from_hexadata_and_writecsv(fld_data, comm)