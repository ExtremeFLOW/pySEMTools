# Import the data types
from mpi4py import MPI #equivalent to the use of MPI_init() in C
from pysemtools.io.ppymech.neksuite import preadnek, pwritenek
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.coef import Coef
from pysemtools.datatypes.field import Field
from pysemtools.interpolation.probes import Probes
import pysemtools.interpolation.utils as interp_utils
import pysemtools.interpolation.pointclouds as pcs
from pysemtools.io.read_probes import ProbesReader
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nekofname = 'output.csv'
pythonfname = 'interpolated_fields_python.csv'


print('Reading neko outputs')
neko  = ProbesReader(nekofname)
print('Reading pysemtools outputs')
pynek = ProbesReader(pythonfname)


print(np.allclose(neko.points, pynek.points, atol=1e-07))

print(neko.fields["w"].shape)
print(pynek.fields["vel2"].shape)

print(np.allclose(neko.fields["w"], pynek.fields["vel2"], atol=1e-07))
