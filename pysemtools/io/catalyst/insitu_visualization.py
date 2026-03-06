""" Module that interfaces with catalyst from a vtk mesh"""

import os
from pathlib import Path
import numpy as np

from mpi4py import MPI
import paraview
import catalyst
import catalyst_conduit as conduit
from ...monitoring import Logger
from ...datatypes.msh_vtk import VTKMesh, VTK_HEXAHEDRON

CATALYST_IMPLEMENTATION_NAME = str(os.getenv("CATALYST_IMPLEMENTATION_NAME", ""))
CATALYST_IMPLEMENTATION_PATH = str(os.getenv("CATALYST_IMPLEMENTATION_PATH", ""))

class CatalystSession:
    """
    Class to interface with catalyst
        
    Interface with catalyst to set up a session and load the pipeline.

    Parameters
    ----------
    comm : MPI.Comm
    pipeline : str
        Path to the saved ParaView Catalyst state file (e.g. "pipeline.py").
    channel : str
        Name of the catalyst channel to use (must match the registrationName in your saved state script, e.g. "rbc00001.vtkhdf").
    implementation_name : str
        Optional name of the catalyst implementation to use. Also set with env variable CATALYST_IMPLEMENTATION_NAME
        e.g. export CATALYST_IMPLEMENTATION_NAME=paraview
    implementation_path : str
        Optional path to the catalyst implementation. Also set with env variable CATALYST_IMPLEMENTATION_PATH
        e.g. export CATALYST_IMPLEMENTATION_PATH=/path/to/paraview/lib/catalyst
    """

    def __init__(self, comm : MPI.Comm, pipeline: str, channel: str, implementation_name = None, implementation_path: str = None):

        self.comm = comm
        self.pipeline = pipeline
        self.channel = channel
        self.implementation_name = implementation_name or CATALYST_IMPLEMENTATION_NAME or ""
        self.implementation_path = implementation_path or CATALYST_IMPLEMENTATION_PATH or ""
        self.log = Logger(comm=self.comm, module_name="CatalystSession")
        self.vtk = None
        self.init_node = None
        self.mesh_node = None
        self.exec_node = None

        if self.implementation_name == "" or self.implementation_path == "":
            raise ValueError("You must specify the catalyst implementation name and path either via environment variables CATALYST_IMPLEMENTATION_NAME and CATALYST_IMPLEMENTATION_PATH, or by passing them to the constructor.")

        # Create conduit node for catalyst initialization
        self.init_node = conduit.Node()
        self.init_node["catalyst/scripts/script/filename"] = str(Path(self.pipeline).resolve())
        self.init_node["catalyst/mpi_comm"] = comm.py2f()

        # Define implementation
        self.init_node["catalyst_load/implementation"] = self.implementation_name
        self.init_node["catalyst_load/search_paths/paraview"] = self.implementation_path

        catalyst.initialize(self.init_node)

        self.log.write("info", f"Initialized CatalystSession - implementation: {self.implementation_name} - path: {self.implementation_path} - pipeline: {self.pipeline}")

    def set_mesh(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, cell_type: str = "hex"):
        """ Set the mesh for the catalyst session / conduit node

        Parameters
        ----------
        x: np.ndarray
            array with local x coordinates on this rank.
        y: np.ndarray
            array with local y coordinates on this rank.
        z: np.ndarray
            array with local z coordinates on this rank.
        cell_type: str
            cell type, currently only "hex" is supported.
        """

        self.log.write("info", f"Setting mesh for Catalyst session")

        # Get the VTK mesh from the coordinates
        # The connectivity and excetera needs to be set up locally
        self.vtk = VTKMesh(self.comm, x, y, z, cell_type=cell_type, global_connectivity=False, distributed_axis=0)

        # Set rank id
        self.mesh_node = conduit.Node()
        self.mesh_node["state/domain_id"] = int(self.comm.Get_rank())

        # Explicit coordinates
        x_local = np.ascontiguousarray(self.vtk.Points[:, 0])
        y_local = np.ascontiguousarray(self.vtk.Points[:, 1])
        z_local = np.ascontiguousarray(self.vtk.Points[:, 2])
        self.mesh_node["coordsets/coords/type"] = "explicit"
        self.mesh_node["coordsets/coords/values/x"].set(x_local)
        self.mesh_node["coordsets/coords/values/y"].set(y_local)
        self.mesh_node["coordsets/coords/values/z"].set(z_local)

        # Indicate the unstructred topology
        self.mesh_node["topologies/mesh/type"] = "unstructured"
        self.mesh_node["topologies/mesh/coordset"] = "coords"
        self.mesh_node["topologies/mesh/elements/shape"] = "mixed"

        # Cell type
        if self.vtk.cell_type == "hex":
            self.mesh_node["topologies/mesh/elements/shape_map/hex"] = VTK_HEXAHEDRON
        else:
            raise ValueError(f"Unsupported cell type {self.vtk.cell_type}, only 'hex' is supported for now.")

        # Set up connectivity and sizes
        self.mesh_node["topologies/mesh/elements/shapes"].set(self.vtk.Types)
        self.mesh_node["topologies/mesh/elements/sizes"].set(self.vtk.Sizes)
        self.mesh_node["topologies/mesh/elements/offsets"].set(self.vtk.Offsets[:self.vtk.Types.size]) # Offsets in vtk are global_number_of_cells + 1 but here we need them to match
        self.mesh_node["topologies/mesh/elements/connectivity"].set(self.vtk.Connectivity)

    def set_field(self, fields: dict[str, np.ndarray]):
        """ Set the fields for the catalyst session / conduit node

        Parameters
        ----------
        fields: dict[str, np.ndarray]
            Dictionary of field name to field values.
        """

        self.log.write("info", f"Setting fields for Catalyst session: {list(fields.keys())}")

        for key in fields.keys():
            self.mesh_node[f"fields/{key}/association"] = "vertex"
            self.mesh_node[f"fields/{key}/topology"] = "mesh"
            self.mesh_node[f"fields/{key}/values"].set(fields[key].flatten())

    def execute(self, timestep=0, time_value=0.0):
        """ Execute the catalyst pipeline for the current mesh and fields

        Parameters
        ----------
        timestep: int
            Current timestep to set in the catalyst state.
        time_value: float
            Current time value to set in the catalyst state.
        """
    
        self.log.write("info", f"Executing Catalyst session - tstep: {timestep} -  t= {time_value}")    
        
        self.exec_node = conduit.Node()
        self.exec_node["catalyst/state/timestep"] = int(timestep)
        self.exec_node["catalyst/state/time"] = float(time_value)

        # MUST match the registrationName in your saved state script.
        self.exec_node[f"catalyst/channels/{self.channel}/type"] = "mesh"
        self.exec_node[f"catalyst/channels/{self.channel}/data"] = self.mesh_node

        catalyst.execute(self.exec_node)


    def finalize(self):
        """ Finalize the catalyst session"""

        self.log.write("info", f"Finalizing Catalyst session")    
        catalyst.finalize(conduit.Node())
        
        self.channel = None
        self.pipeline = None
        self.implementation_name = None
        self.implementation_path = None
        self.init_node = None
        self.mesh_node = None
        self.exec_node = None
        self.vtk = None
