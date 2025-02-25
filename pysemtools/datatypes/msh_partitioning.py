"""
Contains the mesh re partitioner class.
"""

import sys
from typing import Union
import numpy as np
from .msh import Mesh
from .field import Field, FieldRegistry
from ..comm.router import Router
from ..monitoring.logger import Logger


class MeshPartitioner:
    """
    A class that repartitons SEM mesh data using a given partitioning algorithm.

    The idea is to be able to chose subdomains of the mesh and split the elements such that the load is balanced among the ranks.

    One could use this to repartition the data if the condition array is full of True values, but the idea is to be able to use any condition array.

    Parameters
    ----------
    comm : MPI communicator
        MPI communicator

    msh : Mesh
        Mesh object to partition

    conditions : list[np.ndarray]
        List of conditions to apply to the mesh elements. The conditions should be in the form of a list of numpy arrays.
        Each numpy array should have the same length as the number of elements in the mesh. The conditions should be boolean arrays.
    """

    def __init__(
        self, comm, msh: Mesh = None, conditions: list[np.ndarray] = None
    ) -> None:

        self.log = Logger(comm=comm, module_name="Mesh Partitioner")
        self.log.write("info", "Initializing Mesh Partitioner")
        self.rt = Router(comm)

        # Create array that will give true for the elements that are compliant with the conditions
        self.condition = np.all(conditions, axis=0)
        # Create a list with the element index of the compliant elements
        self.compliant_elements = np.unique(np.where(self.condition)[0])
        # Create a new mesh object with the compliant elements
        x_ = msh.x[self.compliant_elements]
        y_ = msh.y[self.compliant_elements]
        z_ = msh.z[self.compliant_elements]

        sub_mesh = Mesh(comm, x=x_, y=y_, z=z_, create_connectivity=False)

        self.partition_nelv = sub_mesh.nelv
        self.partition_glb_nelv = sub_mesh.glb_nelv
        self.partition_global_element_number = sub_mesh.global_element_number
        self.partition_lxyz = sub_mesh.lxyz

    def create_partitioned_mesh(
        self,
        msh: Mesh = None,
        partitioning_algorithm: str = "load_balanced_linear",
        create_conectivity: bool = False,
    ) -> Mesh:
        """
        Create a partitioned mesh object

        Parameters
        ----------
        msh : Mesh
            Mesh object to partition

        partitioning_algorithm : str
            Algorithm to use for partitioning the mesh elements

        Returns
        -------
        partitioned_mesh : Mesh
            Partitioned mesh object
        """

        self.log.write(
            "info",
            f"Partitioning the mesh coordinates with {partitioning_algorithm} algorithm",
        )
        x_ = self.redistribute_field_elements(msh.x, partitioning_algorithm)
        y_ = self.redistribute_field_elements(msh.y, partitioning_algorithm)
        z_ = self.redistribute_field_elements(msh.z, partitioning_algorithm)

        self.log.write("info", "Creating mesh object")
        partitioned_mesh = Mesh(
            self.rt.comm, x=x_, y=y_, z=z_, create_connectivity=create_conectivity
        )

        return partitioned_mesh

    def create_partitioned_field(
        self,
        fld: Union[Field, FieldRegistry] = None,
        partitioning_algorithm: str = "load_balanced_linear",
    ) -> FieldRegistry:
        """
        Create a partitioned field object

        Parameters
        ----------
        fld : Field or FieldRegistry
            Field object to partition

        partitioning_algorithm : str
            Algorithm to use for partitioning the mesh elements

        Returns
        -------
        partitioned_field : FieldRegistry
            Partitioned field object
        """

        self.log.write(
            "info",
            f"Partitioning the field object with {partitioning_algorithm} algorithm",
        )
        
        if isinstance(fld, FieldRegistry):

            partitioned_field = FieldRegistry(self.rt.comm)

            for key in fld.registry.keys():
                field_ = self.redistribute_field_elements(
                        fld.registry[key], partitioning_algorithm
                    )
                partitioned_field.add_field(self.rt.comm, field_name=key, field=field_.copy(), dtype=field_.dtype)

        elif isinstance(fld, Field):

            partitioned_field = FieldRegistry(self.rt.comm)

            for key in fld.fields.keys():
                for i in range(len(fld.fields[key])):
                    field_ = self.redistribute_field_elements(
                        fld.fields[key][i], partitioning_algorithm
                    )
                    partitioned_field.fields[key].append(field_.copy())
     
            partitioned_field.t = fld.t
            partitioned_field.update_vars()

        self.log.write("info", "done")

        return partitioned_field

    def redistribute_field_elements(
        self,
        field: np.ndarray = None,
        partitioning_algorithm: str = "load_balanced_linear",
    ) -> None:
        """
        Redistribute the elements of the mesh object to different ranks

        Parameters
        ----------
        field : np.ndarray
            Field to redistribute based on the conditions at initialization

        partitioning_algorithm : str
            Algorithm to use for partitioning the mesh elements
        """

        if partitioning_algorithm == "load_balanced_linear":

            self.log.write("debug", "Using load balanced linear partitioning algorithm")
            self.log.write(
                "debug", "Determining the number of elements each processor should have"
            )
            nelv, offset_el, n = load_balanced_linear_map(
                self.rt.comm, self.partition_glb_nelv, self.partition_lxyz
            )

            self.log.write("debug", "Partitioning data and redistributing")
            # Get the elements of the field that are compliant with the conditions provided at init
            field_e_shape = (-1, field.shape[1], field.shape[2], field.shape[3])
            field_ = field[self.compliant_elements]

            # Prepare buffers for all other ranks
            destination = []
            data = []
            for rank in range(self.rt.comm.Get_size()):
                condition1 = self.partition_global_element_number >= offset_el[rank]
                condition2 = (
                    self.partition_global_element_number < offset_el[rank] + nelv[rank]
                )
                destination.append(rank)
                data.append(field_[np.where(np.logical_and(condition1, condition2))])

            # Send data to all other ranks
            sources, recvbfs = self.rt.all_to_all(
                destination=destination, data=data, dtype=field.dtype
            )

            self.log.write("debug", "Data received. Reshaping and concatenating")
            # Reshape the datad put it in the field format
            for i in range(0, len(recvbfs)):
                recvbfs[i] = recvbfs[i].reshape(field_e_shape)

            # Put the data in field format
            partitioned_field = np.concatenate(recvbfs, axis=0)

        else:
            self.log.write("error", "Partitioning algorithm not recognized")
            sys.exit(1)

        return partitioned_field


def load_balanced_linear_map(
    comm, glb_nelv: int = None, lxyz: int = None
) -> tuple[list[int], list[int], list[int]]:
    """
    Maps the number of elements each processor has
    in a linearly load balanced manner

    In this case we do it in every rank just to see which data each should have before communicating.

    Parameters
    ----------
    comm : MPI communicator
        MPI communicator

    glb_nelv : int
        Number of global elements

    lxyz : int
        Number of points in an element

    Returns
    -------
    nelv : list[int]
        Number of elements each processor should have

    offset_el : list[int]
        Offset of the elements each processor should have

    n : list[int]
        Number of points each processor should have
    """

    nelv = []
    offset_el = []
    n = []
    for pe_rank in range(comm.Get_size()):

        m = np.int64(glb_nelv)
        pe_rank = np.int64(pe_rank)
        pe_size = np.int64(comm.Get_size())
        l = np.floor(np.double(m) / np.double(pe_size))
        r = np.mod(m, pe_size)
        ip = np.floor(
            (np.double(m) + np.double(pe_size) - np.double(pe_rank) - np.double(1))
            / np.double(pe_size)
        )

        nelv_ = np.int64(ip)
        offset_el_ = np.int64(pe_rank * l + min(pe_rank, r))
        n_ = lxyz * nelv

        nelv.append(nelv_)
        offset_el.append(offset_el_)
        n.append(n_)

    return nelv, offset_el, n
