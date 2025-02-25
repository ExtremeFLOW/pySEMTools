"""Implements the mesh connectivity class"""

from .msh import Mesh
from .coef import Coef
from ..comm.router import Router
from ..monitoring.logger import Logger
import numpy as np
from .element_slicing import fetch_elem_facet_data as fd
from .element_slicing import fetch_elem_edge_data as ed
from .element_slicing import fetch_elem_vertex_data as vd
from .element_slicing import (
    vertex_to_slice_map_2d,
    vertex_to_slice_map_3d,
    edge_to_slice_map_2d,
    edge_to_slice_map_3d,
    facet_to_slice_map,
    edge_to_vertex_map_2d,
    edge_to_vertex_map_3d,
    facet_to_vertex_map,
)
import sys


class MeshConnectivity:
    """
    Class to compute the connectivity of the mesh

    Uses facets and vertices to determine which elements are connected to each other

    Parameters
    ----------
    comm : MPI communicator
        The MPI communicator
    msh : Mesh
        The mesh object
    rel_tol : float
        The relative tolerance to use when comparing the coordinates of the facets/edges
    use_hashtable : bool
        Whether to use a hashtable to define connectivity. This is faster but uses more memory
    max_simultaneous_sends : int
        The maximum number of simultaneous sends to use when sending data to other ranks. A lower number 
        saves memory for buffers but is slower.
    max_elem_per_vertex : int
        The maximum number of elements that share a vertex. The default values are 4 for 2D and 8 for 3D (Works for a structured mesh)
        The default value is selected if this input is left as None
    max_elem_per_edge : int
        The maximum number of elements that share an edge. The default values are 2 for 2D and 4 for 3D (Works for a structured mesh)
        The default value is selected if this input is left as None
    max_elem_per_face : int
        The maximum number of elements that share a face. The default values are 2 for 3D (Works for a structured mesh)
        The default value is selected if this input is left as None
    """

    def __init__(self, comm, msh: Mesh = None, rel_tol=1e-5, use_hashtable=False, max_simultaneous_sends=1, max_elem_per_vertex: int = None, max_elem_per_edge: int = None, max_elem_per_face: int = None):

        self.log = Logger(comm=comm, module_name="MeshConnectivity")
        self.log.write("info", "Initializing MeshConnectivity")
        self.log.tic()
        self.rt = Router(comm)
        self.rtol = rel_tol
        self.use_hashtable = use_hashtable
        self.max_simultaneous_sends = max_simultaneous_sends
        self.max_elem_per_vertex = max_elem_per_vertex
        self.max_elem_per_edge = max_elem_per_edge
        self.max_elem_per_face = max_elem_per_face

        if isinstance(msh, Mesh):
            # Create local connecitivy
            self.log.write("info", "Computing local connectivity")
            self.local_connectivity(msh)

            # Create global connectivity
            self.log.write("info", "Computing global connectivity")
            self.global_connectivity(msh)

            # Get the multiplicity
            self.log.write("info", "Computing multiplicity")
            self.get_multiplicity(msh)

        self.log.write("info", "MeshConnectivity initialized")
        self.log.toc()

    def local_connectivity(self, msh: Mesh):
        """
        Computes the local connectivity of the mesh

        This function checks elements within a rank

        Parameters
        ----------
        msh : Mesh
            The mesh object

        Notes
        -----
        In 3D, the centers of the facets are compared.
        efp means element facet pair.
        One obtains a local_shared_efp_to_elem_map and a local_shared_efp_to_facet_map dictionary.

        local_shared_efp_to_elem_map[(e, f)] = [e1, e2, ...] gives a list with the elements e1, e2 ...
        that share the same facet f of element e.

        local_shared_efp_to_facet_map[(e, f)] = [f1, f2, ...] gives a list with the facets f1, f2 ...
        of the elements e1, e2 ... that share the same facet f of element e.

        In each case, the index of the element list, corresponds to the index of the facet list.
        Therefore, the element list might have repeated element entries.

        Additionally, we create a list of unique_efp_elem and unique_efp_facet, which are the elements and facets
        that are not shared with any other element. These are either boundary elements or elements that are connected to other ranks.
        the unique pairs are the ones that are checked in global connecitivy,
        """

        if msh.gdim >= 1:

            self.log.write("debug", "Computing local connectivity: Using vertices")

            if self.max_elem_per_vertex is None:
                if msh.gdim == 2:
                    min_vertex = 4  # Anything less than 4 means that a vertex might be in another rank
                else:
                    min_vertex = 8  # Anything less than 8 means that a vertex might be in another rank
            else:
                min_vertex = self.max_elem_per_vertex

            (
                self.local_shared_evp_to_elem_map,
                self.local_shared_evp_to_vertex_map,
                self.incomplete_evp_elem,
                self.incomplete_evp_vertex,
            ) = find_local_shared_vef(
                vef_coords=msh.vertices,
                rtol=self.rtol,
                min_shared=min_vertex,
                use_hashtable=self.use_hashtable,
            )

        if msh.gdim >= 2:

            self.log.write("debug", "Computing local connectivity: Using edge centers")

            if self.max_elem_per_edge is None:
                if msh.gdim == 2:
                    min_edges = 2
                else:
                    min_edges = 4
            else:
                min_edges = self.max_elem_per_edge

            (
                self.local_shared_eep_to_elem_map,
                self.local_shared_eep_to_edge_map,
                self.incomplete_eep_elem,
                self.incomplete_eep_edge,
            ) = find_local_shared_vef(
                vef_coords=msh.edge_centers,
                rtol=self.rtol,
                min_shared=min_edges,
                use_hashtable=self.use_hashtable,
            )

        if msh.gdim >= 3:

            self.log.write("debug", "Computing local connectivity: Using facet centers")

            if self.max_elem_per_face is None:
                min_facets = 2
            else:
                min_facets = self.max_elem_per_face

            (
                self.local_shared_efp_to_elem_map,
                self.local_shared_efp_to_facet_map,
                self.unique_efp_elem,
                self.unique_efp_facet,
            ) = find_local_shared_vef(
                vef_coords=msh.facet_centers,
                rtol=self.rtol,
                min_shared=min_facets,
                use_hashtable=self.use_hashtable,
            )

    def global_connectivity(self, msh: Mesh):
        """
        Computes the global connectivity of the mesh

        Currently this function sends data from all to all.

        Parameters
        ----------
        msh : Mesh
            The mesh object

        Notes
        -----
        In 3D. this function sends the facet centers of the unique_efp_elem and unique_efp_facet to all other ranks.
        as well as the element ID and facet ID to be assigned.

        We compare the unique facet centers of our rank to those of others and determine which one matches.
        When we find that one matches, we populate the directories:
        global_shared_efp_to_rank_map[(e, f)] = rank
        global_shared_efp_to_elem_map[(e, f)] = elem
        global_shared_efp_to_facet_map[(e, f)] = facet

        So for each element facet pair we will know which rank has it, and which is their ID in that rank.

        BE MINDFUL: Later when redistributing, send the points, but also send the element and facet ID to the other ranks so the reciever
        can know which is the facet that corresponds.
        """

        if msh.gdim >= 1:

            self.log.write("debug", "Computing global connectivity: Using vertices")
            (
                self.global_shared_evp_to_rank_map,
                self.global_shared_evp_to_elem_map,
                self.global_shared_evp_to_vertex_map,
            ) = find_global_shared_evp(
                self.rt,
                msh.vertices,
                self.incomplete_evp_elem,
                self.incomplete_evp_vertex,
                self.rtol,
                self.use_hashtable,
                self.max_simultaneous_sends,
            )

        if msh.gdim >= 2:

            self.log.write("debug", "Computing global connectivity: Using edge centers")
            (
                self.global_shared_eep_to_rank_map,
                self.global_shared_eep_to_elem_map,
                self.global_shared_eep_to_edge_map,
            ) = find_global_shared_evp(
                self.rt,
                msh.edge_centers,
                self.incomplete_eep_elem,
                self.incomplete_eep_edge,
                self.rtol,
                self.use_hashtable,
                self.max_simultaneous_sends,
            )

        if msh.gdim == 3:

            self.log.write(
                "debug", "Computing global connectivity: Using facet centers"
            )
            (
                self.global_shared_efp_to_rank_map,
                self.global_shared_efp_to_elem_map,
                self.global_shared_efp_to_facet_map,
            ) = find_global_shared_evp(
                self.rt,
                msh.facet_centers,
                self.unique_efp_elem,
                self.unique_efp_facet,
                self.rtol,
                self.use_hashtable,
                self.max_simultaneous_sends,
            )

    def get_multiplicity(self, msh: Mesh):
        """
        Computes the multiplicity of the elements in the mesh

        Parameters
        ----------
        msh : Mesh

        Notes
        -----
        The multiplicity is the number of times a point in a element is shared with its own element or others.
        The minimum multiplicity is 1, since the point is always shared with itself.
        """

        self.multiplicity = np.ones_like(msh.x)

        if msh.gdim == 2:
            vertex_to_slice_map = vertex_to_slice_map_2d
            edge_to_slice_map = edge_to_slice_map_2d
        elif msh.gdim == 3:
            vertex_to_slice_map = vertex_to_slice_map_3d
            edge_to_slice_map = edge_to_slice_map_3d

        for e in range(0, msh.nelv):

            if msh.gdim >= 1:

                # Add number of vertices
                for vertex in range(0, msh.vertices.shape[1]):

                    local_appearances = len(
                        self.local_shared_evp_to_elem_map.get((e, vertex), [])
                    )
                    global_appearances = len(
                        self.global_shared_evp_to_elem_map.get((e, vertex), [])
                    )

                    lz_index = vertex_to_slice_map[vertex][0]
                    ly_index = vertex_to_slice_map[vertex][1]
                    lx_index = vertex_to_slice_map[vertex][2]

                    self.multiplicity[e, lz_index, ly_index, lx_index] = (
                        local_appearances + global_appearances
                    )

            if msh.gdim >= 2:

                # Add number of edges
                for edge in range(0, msh.edge_centers.shape[1]):

                    local_appearances = len(
                        self.local_shared_eep_to_elem_map.get((e, edge), [])
                    )
                    global_appearances = len(
                        self.global_shared_eep_to_elem_map.get((e, edge), [])
                    )

                    lz_index = edge_to_slice_map[edge][0]
                    ly_index = edge_to_slice_map[edge][1]
                    lx_index = edge_to_slice_map[edge][2]

                    # Exclude vertices
                    if lz_index == slice(None):
                        lz_index = slice(1, -1)
                    if ly_index == slice(None):
                        ly_index = slice(1, -1)
                    if lx_index == slice(None):
                        lx_index = slice(1, -1)

                    self.multiplicity[e, lz_index, ly_index, lx_index] = (
                        local_appearances + global_appearances
                    )

            if msh.gdim >= 3:

                # Add number of facets
                for facet in range(0, 6):

                    local_appearances = len(
                        self.local_shared_efp_to_elem_map.get((e, facet), [])
                    )
                    global_appearances = len(
                        self.global_shared_efp_to_elem_map.get((e, facet), [])
                    )

                    lz_index = facet_to_slice_map[facet][0]
                    ly_index = facet_to_slice_map[facet][1]
                    lx_index = facet_to_slice_map[facet][2]

                    # Exclude edges
                    if lz_index == slice(None):
                        lz_index = slice(1, -1)
                    if ly_index == slice(None):
                        ly_index = slice(1, -1)
                    if lx_index == slice(None):
                        lx_index = slice(1, -1)

                    self.multiplicity[e, lz_index, ly_index, lx_index] = (
                        local_appearances + global_appearances
                    )

    def dssum(
        self, field: np.ndarray = None, msh: Mesh = None, average: str = "multiplicity"
    ):
        """
        Computes the dssum of the field

        Parameters
        ----------
        field : np.ndarray
            The field to compute the dssum
        msh : Mesh
            The mesh object
        average : str
            The averaging weights to use. Can be "multiplicity"

        Returns
        -------
        np.ndarray
            The dssum of the field
        """

        # Always compute local dssum
        dssum_field = self.dssum_local(field=field, msh=msh)

        # If running in parallel, compute the global ds sum
        if self.rt.comm.Get_size() > 1:
            iferror = False
            try:
                dssum_field = self.dssum_global(
                    local_dssum_field=dssum_field, field=field, msh=msh
                )
            
            except KeyError as e:
                iferror = True
                self.log.write("error", f"Error in rank {self.rt.comm.Get_rank()} - Key: {e} does not exist in global connectivity dictionaries - dssum not completed succesfully")
                self.log.write("error", f"This error happens when using unstructured meshes. Input the max number of elements that can share a vertex, edge or face when initializing the mesh conectivity object and try again.")
            
            self.rt.comm.Barrier()

            if iferror:
                sys.exit(1)

        if average == "multiplicity":
            self.log.write("info", "Averaging using the multiplicity")
            dssum_field = dssum_field / self.multiplicity

        return dssum_field

    def dssum_local(self, field: np.ndarray = None, msh: Mesh = None):
        """
        Computes the local dssum of the field

        Parameters
        ----------
        field : np.ndarray
            The field to compute the dssum
        msh : Mesh
            The mesh object

        Returns
        -------
        np.ndarray
            The local dssum of the field
        """

        self.log.write("info", "Computing local dssum")
        self.log.tic()

        local_dssum_field = np.copy(field)

        if msh.gdim == 2:
            vertex_to_slice_map = vertex_to_slice_map_2d
            edge_to_slice_map = edge_to_slice_map_2d
            edge_to_vertex_map = edge_to_vertex_map_2d
        elif msh.gdim == 3:
            vertex_to_slice_map = vertex_to_slice_map_3d
            edge_to_slice_map = edge_to_slice_map_3d
            edge_to_vertex_map = edge_to_vertex_map_3d

        if msh.gdim >= 1:
            self.log.write("debug", "Adding vertices")
            for e in range(0, msh.nelv):

                # Vertex data is pointwise and can be summed directly
                for vertex in range(0, msh.vertices.shape[1]):

                    if (e, vertex) in self.local_shared_evp_to_elem_map.keys():

                        # Get the data from other elements
                        shared_elements_ = list(
                            self.local_shared_evp_to_elem_map[(e, vertex)]
                        )
                        shared_vertices_ = list(
                            self.local_shared_evp_to_vertex_map[(e, vertex)]
                        )

                        # Filter out my own element from the list
                        shared_elements = [
                            shared_elements_[ii]
                            for ii in range(0, len(shared_elements_))
                            if shared_elements_[ii] != e
                        ]
                        shared_vertices = [
                            shared_vertices_[ii]
                            for ii in range(0, len(shared_elements_))
                            if shared_elements_[ii] != e
                        ]

                        if shared_vertices == []:
                            continue

                        # Get the vertex data from the other elements of the field.
                        shared_vertex_data = vd(
                            field=field, elem=shared_elements, vertex=shared_vertices
                        )

                        # Get the vertex location on my own elemenet
                        lz_index = vertex_to_slice_map[vertex][0]
                        ly_index = vertex_to_slice_map[vertex][1]
                        lx_index = vertex_to_slice_map[vertex][2]

                        local_dssum_field[e, lz_index, ly_index, lx_index] += np.sum(
                            shared_vertex_data
                        )

        if msh.gdim >= 2:
            self.log.write("debug", "Adding edges")
            for e in range(0, msh.nelv):
                # Edge data is provided as a line that might be flipped, we must compare values of the mesh
                for edge in range(0, msh.edge_centers.shape[1]):

                    if (e, edge) in self.local_shared_eep_to_elem_map.keys():

                        # Get the data from other elements
                        shared_elements_ = list(
                            self.local_shared_eep_to_elem_map[(e, edge)]
                        )
                        shared_edges_ = list(
                            self.local_shared_eep_to_edge_map[(e, edge)]
                        )

                        # Filter out my own element from the list
                        shared_elements = [
                            shared_elements_[ii]
                            for ii in range(0, len(shared_elements_))
                            if shared_elements_[ii] != e
                        ]
                        shared_edges = [
                            shared_edges_[ii]
                            for ii in range(0, len(shared_elements_))
                            if shared_elements_[ii] != e
                        ]

                        if shared_edges == []:
                            continue

                        # Get the shared edge coordinates from the other elements
                        shared_edge_coord_x = ed(
                            field=msh.x, elem=shared_elements, edge=shared_edges
                        )
                        shared_edge_coord_y = ed(
                            field=msh.y, elem=shared_elements, edge=shared_edges
                        )
                        shared_edge_coord_z = ed(
                            field=msh.z, elem=shared_elements, edge=shared_edges
                        )

                        # Get the shared edge data from the other elements of the field.
                        shared_edge_data = ed(
                            field=field, elem=shared_elements, edge=shared_edges
                        )

                        # Get the edge location on my own elemenet
                        lz_index = edge_to_slice_map[edge][0]
                        ly_index = edge_to_slice_map[edge][1]
                        lx_index = edge_to_slice_map[edge][2]

                        # Get my own edge data and coordinates
                        my_edge_coord_x = msh.x[e, lz_index, ly_index, lx_index]
                        my_edge_coord_y = msh.y[e, lz_index, ly_index, lx_index]
                        my_edge_coord_z = msh.z[e, lz_index, ly_index, lx_index]
                        my_edge_data = np.copy(field[e, lz_index, ly_index, lx_index])

                        # Figure out if the edges are flipped.
                        ## First find the vertices of my edge
                        my_edge_vertices = edge_to_vertex_map[edge]

                        ## Now check how they are actually aligned to see if they are flipped
                        ### Find which are the shared vertices of my own edge vertices that are in each entry of shared element
                        ### Note that in general, each vertex in one element will have 1 matching vertex in another... otherwise something is weird
                        shared_vertex_idx_of_my_edge_vertex_0 = [
                            self.local_shared_evp_to_vertex_map[
                                (e, my_edge_vertices[0])
                            ][
                                np.where(
                                    np.array(
                                        self.local_shared_evp_to_elem_map[
                                            (e, my_edge_vertices[0])
                                        ]
                                    )
                                    == se
                                )
                            ][
                                0
                            ]
                            for se in shared_elements
                        ]
                        shared_vertex_idx_of_my_edge_vertex_1 = [
                            self.local_shared_evp_to_vertex_map[
                                (e, my_edge_vertices[1])
                            ][
                                np.where(
                                    np.array(
                                        self.local_shared_evp_to_elem_map[
                                            (e, my_edge_vertices[1])
                                        ]
                                    )
                                    == se
                                )
                            ][
                                0
                            ]
                            for se in shared_elements
                        ]

                        ### create a list of how the shared vetices are actually oriented
                        shared_vertex_orientation = [
                            (
                                int(shared_vertex_idx_of_my_edge_vertex_0[i]),
                                int(shared_vertex_idx_of_my_edge_vertex_1[i]),
                            )
                            for i in range(len(shared_elements))
                        ]

                        ### Now compare, if they are not the same, then you must flip the edge data
                        flip_edge = []
                        for i in range(len(shared_elements)):
                            # if vertex_matching_if_aligned[i] != actual_vertex_matching[i]:
                            if (
                                shared_vertex_orientation[i][1]
                                - shared_vertex_orientation[i][0]
                                < 0
                            ):
                                flip_edge.append(True)
                            else:
                                flip_edge.append(False)

                        # Sum the data
                        for idx in range(0, len(shared_elements)):
                            if flip_edge[idx]:
                                shared_edge_data[idx] = np.flip(shared_edge_data[idx])

                            my_edge_data += shared_edge_data[idx]

                        # Do not assing at the vertices
                        if lz_index == slice(None):
                            lz_index = slice(1, -1)
                        if ly_index == slice(None):
                            ly_index = slice(1, -1)
                        if lx_index == slice(None):
                            lx_index = slice(1, -1)
                        slice_copy = slice(1, -1)
                        local_dssum_field[e, lz_index, ly_index, lx_index] = np.copy(
                            my_edge_data[slice_copy]
                        )

        if msh.gdim >= 3:
            self.log.write("debug", "Adding faces")
            for e in range(0, msh.nelv):

                # Facet data might be flipper or rotated so better check coordinates
                for facet in range(0, 6):

                    if (e, facet) in self.local_shared_efp_to_elem_map.keys():

                        # Get the data from other elements
                        shared_elements_ = list(
                            self.local_shared_efp_to_elem_map[(e, facet)]
                        )
                        shared_facets_ = list(
                            self.local_shared_efp_to_facet_map[(e, facet)]
                        )

                        # Filter out my own element from the list
                        shared_elements = [
                            shared_elements_[ii]
                            for ii in range(0, len(shared_elements_))
                            if shared_elements_[ii] != e
                        ]
                        shared_facets = [
                            shared_facets_[ii]
                            for ii in range(0, len(shared_elements_))
                            if shared_elements_[ii] != e
                        ]

                        if shared_facets == []:
                            continue

                        # Get the shared facet coordinates from the other elements
                        shared_facet_coord_x = fd(
                            field=msh.x, elem=shared_elements, facet=shared_facets
                        )
                        shared_facet_coord_y = fd(
                            field=msh.y, elem=shared_elements, facet=shared_facets
                        )
                        shared_facet_coord_z = fd(
                            field=msh.z, elem=shared_elements, facet=shared_facets
                        )

                        # Get the shared facet data from the other elements of the field.
                        shared_facet_data = fd(
                            field=field, elem=shared_elements, facet=shared_facets
                        )

                        # Get the facet location on my own elemenet
                        lz_index = facet_to_slice_map[facet][0]
                        ly_index = facet_to_slice_map[facet][1]
                        lx_index = facet_to_slice_map[facet][2]

                        # Get my own facet data and coordinates
                        my_facet_coord_x = msh.x[e, lz_index, ly_index, lx_index]
                        my_facet_coord_y = msh.y[e, lz_index, ly_index, lx_index]
                        my_facet_coord_z = msh.z[e, lz_index, ly_index, lx_index]
                        my_facet_data = np.copy(field[e, lz_index, ly_index, lx_index])

                        # Figure out if the facets are flipped.
                        ## First find the vertices of my facet
                        my_facet_vertices = facet_to_vertex_map[facet]

                        ## Now check how my vertices pair with the ones from the shared element
                        ### Find which are the shared vertices of my own facet vertices that are in each entry of shared element
                        ### Note that in general, each vertex in one element will have 1 matching vertex in another... otherwise something is weird
                        ### For axis 1
                        shared_vertex_idx_of_my_facet_vertex_0_ax1 = [
                            self.local_shared_evp_to_vertex_map[
                                (e, my_facet_vertices[0][0])
                            ][
                                np.where(
                                    np.array(
                                        self.local_shared_evp_to_elem_map[
                                            (e, my_facet_vertices[0][0])
                                        ]
                                    )
                                    == se
                                )
                            ][
                                0
                            ]
                            for se in shared_elements
                        ]
                        shared_vertex_idx_of_my_facet_vertex_1_ax1 = [
                            self.local_shared_evp_to_vertex_map[
                                (e, my_facet_vertices[0][1])
                            ][
                                np.where(
                                    np.array(
                                        self.local_shared_evp_to_elem_map[
                                            (e, my_facet_vertices[0][1])
                                        ]
                                    )
                                    == se
                                )
                            ][
                                0
                            ]
                            for se in shared_elements
                        ]
                        ### For axis 0
                        shared_vertex_idx_of_my_facet_vertex_0_ax0 = [
                            self.local_shared_evp_to_vertex_map[
                                (e, my_facet_vertices[2][0])
                            ][
                                np.where(
                                    np.array(
                                        self.local_shared_evp_to_elem_map[
                                            (e, my_facet_vertices[2][0])
                                        ]
                                    )
                                    == se
                                )
                            ][
                                0
                            ]
                            for se in shared_elements
                        ]
                        shared_vertex_idx_of_my_facet_vertex_1_ax0 = [
                            self.local_shared_evp_to_vertex_map[
                                (e, my_facet_vertices[2][1])
                            ][
                                np.where(
                                    np.array(
                                        self.local_shared_evp_to_elem_map[
                                            (e, my_facet_vertices[2][1])
                                        ]
                                    )
                                    == se
                                )
                            ][
                                0
                            ]
                            for se in shared_elements
                        ]

                        ### Create a list of how the shared vetices are actually oriented
                        shared_vertex_orientation_axis_1 = [
                            (
                                int(shared_vertex_idx_of_my_facet_vertex_0_ax1[i]),
                                int(shared_vertex_idx_of_my_facet_vertex_1_ax1[i]),
                            )
                            for i in range(len(shared_elements))
                        ]
                        shared_vertex_orientation_axis_0 = [
                            (
                                int(shared_vertex_idx_of_my_facet_vertex_0_ax0[i]),
                                int(shared_vertex_idx_of_my_facet_vertex_1_ax0[i]),
                            )
                            for i in range(len(shared_elements))
                        ]

                        ### Now compare. If the vertex index of the shared facet orientation are not increasing, then it is flipped.
                        ### Note: A more general way to do this is to check the sign of the difference of the vertex indices. If it is the same as the sign of the difference of the vertex indices of my facet, then it is not flipped.
                        ### however since my_facet_vertices is always increasing by construction in the dictionary, we can just check if the shared vertex indices are increasing.
                        flip_facet_axis_1 = []
                        flip_facet_axis_0 = []
                        for i in range(len(shared_elements)):

                            if (
                                shared_vertex_orientation_axis_1[i][1]
                                - shared_vertex_orientation_axis_1[i][0]
                                < 0
                            ):
                                # if np.sign(shared_vertex_orientation_axis_1[i][1] - shared_vertex_orientation_axis_1[i][0]) != np.sign(my_facet_vertices[2][1] - my_facet_vertices[2][0]):
                                flip_facet_axis_1.append(True)
                            else:
                                flip_facet_axis_1.append(False)
                            if (
                                shared_vertex_orientation_axis_0[i][1]
                                - shared_vertex_orientation_axis_0[i][0]
                                < 0
                            ):
                                # if np.sign(shared_vertex_orientation_axis_0[i][1] - shared_vertex_orientation_axis_0[i][0]) != np.sign(my_facet_vertices[0][1] - my_facet_vertices[0][0]):
                                flip_facet_axis_0.append(True)
                            else:
                                flip_facet_axis_0.append(False)

                        # Sum the data
                        for idx in range(0, len(shared_elements)):

                            if flip_facet_axis_1[idx]:
                                shared_facet_data[idx] = np.flip(
                                    shared_facet_data[idx], axis=1
                                )
                            if flip_facet_axis_0[idx]:
                                shared_facet_data[idx] = np.flip(
                                    shared_facet_data[idx], axis=0
                                )

                            my_facet_data += shared_facet_data[idx]

                        # Do not assing at the edges
                        if lz_index == slice(None):
                            lz_index = slice(1, -1)
                        if ly_index == slice(None):
                            ly_index = slice(1, -1)
                        if lx_index == slice(None):
                            lx_index = slice(1, -1)
                        slice_copy = slice(1, -1)
                        local_dssum_field[e, lz_index, ly_index, lx_index] = np.copy(
                            my_facet_data[slice_copy, slice_copy]
                        )

        self.log.write("info", "Local dssum computed")
        self.log.toc()

        return local_dssum_field

    def dssum_global(
        self,
        local_dssum_field: np.ndarray = None,
        field: np.ndarray = None,
        msh: Mesh = None,
    ):
        """
        Computes the global dssum of the field

        Parameters
        ----------
        local_dssum_field : np.ndarray
            The local dssum of the field, computed with dssum_local
        field : np.ndarray
            The field to compute the dssum
        msh : Mesh
            The mesh object

        Returns
        -------
        np.ndarray
            The global dssum of the field
        """

        self.log.write("info", "Computing global dssum")
        self.log.tic()

        global_dssum_field = np.copy(local_dssum_field)

        if msh.gdim == 2:
            vertex_to_slice_map = vertex_to_slice_map_2d
            edge_to_slice_map = edge_to_slice_map_2d
            edge_to_vertex_map = edge_to_vertex_map_2d
        elif msh.gdim == 3:
            vertex_to_slice_map = vertex_to_slice_map_3d
            edge_to_slice_map = edge_to_slice_map_3d
            edge_to_vertex_map = edge_to_vertex_map_3d

        if msh.gdim >= 1:
            self.log.write("debug", "Adding vertices")
            # Prepare data to send to other ranks:
            vertex_send_buff = prepare_send_buffers(
                msh=msh,
                field=field,
                vef_to_rank_map=self.global_shared_evp_to_rank_map,
                data_to_fetch="vertex",
            )

            # Send and receive the data
            (
                vertex_sources,
                source_vertex_el_id,
                source_vertex_id,
                source_vertex_x_coords,
                source_vertex_y_coords,
                source_vertex_z_coords,
                source_vertex_data,
            ) = send_recv_data(
                rt=self.rt, send_buff=vertex_send_buff, data_to_send="vertex", lx=msh.lx
            )

            # Summ vertices:
            for e in range(0, msh.nelv):

                # Vertex data is pointwise and can be summed directly
                for vertex in range(0, msh.vertices.shape[1]):

                    if (e, vertex) in self.global_shared_evp_to_elem_map.keys():

                        # Check which other rank has this vertex
                        shared_ranks = list(
                            self.global_shared_evp_to_rank_map[(e, vertex)]
                        )
                        shared_elements = list(
                            self.global_shared_evp_to_elem_map[(e, vertex)]
                        )
                        shared_vertices = list(
                            self.global_shared_evp_to_vertex_map[(e, vertex)]
                        )

                        # Get the vertex data from the different ranks
                        for sv in range(0, len(shared_vertices)):

                            source_index = list(vertex_sources).index(shared_ranks[sv])

                            # Get the data from this source
                            shared_vertex_el_id = source_vertex_el_id[source_index]
                            shared_vertex_id = source_vertex_id[source_index]
                            shared_vertex_coord_x = source_vertex_x_coords[source_index]
                            shared_vertex_coord_y = source_vertex_y_coords[source_index]
                            shared_vertex_coord_z = source_vertex_z_coords[source_index]
                            shared_vertex_data = source_vertex_data[source_index]

                            # find the data that matches the element and vertex id dictionary
                            el = shared_elements[sv]
                            vertex_id = shared_vertices[sv]
                            same_el = shared_vertex_el_id == el
                            same_vertex = shared_vertex_id == vertex_id
                            matching_index = np.where(same_el & same_vertex)

                            matching_vertex_coord_x = shared_vertex_coord_x[
                                matching_index
                            ]
                            matching_vertex_coord_y = shared_vertex_coord_y[
                                matching_index
                            ]
                            matching_vertex_coord_z = shared_vertex_coord_z[
                                matching_index
                            ]
                            matching_vertex_data = shared_vertex_data[matching_index]

                            # Get my own values (It is not really needed for the vertices)
                            my_vertex_coord_x = vd(field=msh.x, elem=e, vertex=vertex)
                            my_vertex_coord_y = vd(field=msh.y, elem=e, vertex=vertex)
                            my_vertex_coord_z = vd(field=msh.z, elem=e, vertex=vertex)
                            my_vertex_data = np.copy(
                                vd(field=field, elem=e, vertex=vertex)
                            )

                            # Get the vertex location on my own elemenet
                            lz_index = vertex_to_slice_map[vertex][0]
                            ly_index = vertex_to_slice_map[vertex][1]
                            lx_index = vertex_to_slice_map[vertex][2]

                            # Add the data from this rank, element, vertex triad.
                            global_dssum_field[
                                e, lz_index, ly_index, lx_index
                            ] += matching_vertex_data

        if msh.gdim >= 2:
            self.log.write("debug", "Adding edges")
            # Prepare data to send to other ranks:
            edge_send_buff = prepare_send_buffers(
                msh=msh,
                field=field,
                vef_to_rank_map=self.global_shared_eep_to_rank_map,
                data_to_fetch="edge",
            )

            # Send and receive the data
            (
                edges_sources,
                source_edge_el_id,
                source_edge_id,
                source_edge_x_coords,
                source_edge_y_coords,
                source_edge_z_coords,
                source_edge_data,
            ) = send_recv_data(
                rt=self.rt, send_buff=edge_send_buff, data_to_send="edge", lx=msh.lx
            )

            # Summ edges:
            for e in range(0, msh.nelv):

                # Edge data is provided as a line that might be flipped, we must compare values of the mesh
                for edge in range(0, msh.edge_centers.shape[1]):

                    if (e, edge) in self.global_shared_eep_to_elem_map.keys():

                        # Check which other rank has this edge
                        shared_ranks = list(
                            self.global_shared_eep_to_rank_map[(e, edge)]
                        )
                        shared_elements = list(
                            self.global_shared_eep_to_elem_map[(e, edge)]
                        )
                        shared_edges = list(
                            self.global_shared_eep_to_edge_map[(e, edge)]
                        )

                        # Get the edge data from the different ranks
                        shared_edge_index = 0
                        for se in range(0, len(shared_edges)):

                            source_index = list(edges_sources).index(shared_ranks[se])

                            # Get the data from this source
                            shared_edge_el_id = source_edge_el_id[source_index]
                            shared_edge_id = source_edge_id[source_index]
                            shared_edge_coord_x = source_edge_x_coords[source_index]
                            shared_edge_coord_y = source_edge_y_coords[source_index]
                            shared_edge_coord_z = source_edge_z_coords[source_index]
                            shared_edge_data = source_edge_data[source_index]

                            # find the data that matches the element and edge id dictionary
                            el = shared_elements[se]
                            edge_id = shared_edges[se]
                            same_el = shared_edge_el_id == el
                            same_edge = shared_edge_id == edge_id
                            matching_index = np.where(same_el & same_edge)

                            matching_element = shared_edge_el_id[matching_index]
                            matching_edge = shared_edge_id[matching_index]
                            matching_edge_coord_x = shared_edge_coord_x[matching_index]
                            matching_edge_coord_y = shared_edge_coord_y[matching_index]
                            matching_edge_coord_z = shared_edge_coord_z[matching_index]
                            matching_edge_data = shared_edge_data[matching_index]

                            # Get the edge location on my own elemenet
                            lz_index = edge_to_slice_map[edge][0]
                            ly_index = edge_to_slice_map[edge][1]
                            lx_index = edge_to_slice_map[edge][2]

                            # Get my own edge data and coordinates
                            my_edge_coord_x = msh.x[e, lz_index, ly_index, lx_index]
                            my_edge_coord_y = msh.y[e, lz_index, ly_index, lx_index]
                            my_edge_coord_z = msh.z[e, lz_index, ly_index, lx_index]
                            # Since edge can be in many elements and ranks, only copy the clean data once
                            if shared_edge_index == 0:
                                my_edge_data = np.copy(
                                    local_dssum_field[e, lz_index, ly_index, lx_index]
                                )
                            else:
                                my_edge_data = np.copy(
                                    global_dssum_field[e, lz_index, ly_index, lx_index]
                                )

                            # Figure out if the edges are flipped.
                            ## First find the vertices of my edge
                            my_edge_vertices = edge_to_vertex_map[edge]

                            ## Now check how they are actually aligned to see if they are flipped
                            ### Find which are the shared vertices of my own edge vertices that are in each entry of shared element
                            ### Note that in general, each vertex in one element will have 1 matching vertex in another... otherwise something is weird
                            shared_vertex_idx_of_my_edge_vertex_0 = [
                                self.global_shared_evp_to_vertex_map[
                                    (e, my_edge_vertices[0])
                                ][
                                    np.where(
                                        np.array(
                                            self.global_shared_evp_to_elem_map[
                                                (e, my_edge_vertices[0])
                                            ]
                                        )
                                        == se
                                    )
                                ][
                                    0
                                ]
                                for se in matching_element
                            ]
                            shared_vertex_idx_of_my_edge_vertex_1 = [
                                self.global_shared_evp_to_vertex_map[
                                    (e, my_edge_vertices[1])
                                ][
                                    np.where(
                                        np.array(
                                            self.global_shared_evp_to_elem_map[
                                                (e, my_edge_vertices[1])
                                            ]
                                        )
                                        == se
                                    )
                                ][
                                    0
                                ]
                                for se in matching_element
                            ]

                            ### Create a list of how the shared vetices are actually oriented
                            shared_vertex_orientation = [
                                (
                                    int(shared_vertex_idx_of_my_edge_vertex_0[i]),
                                    int(shared_vertex_idx_of_my_edge_vertex_1[i]),
                                )
                                for i in range(len(matching_element))
                            ]

                            ### Now compare, if they are not the same, then you must flip the edge data
                            flip_edge = []
                            for i in range(len(matching_element)):
                                # if vertex_matching_if_aligned[i] != actual_vertex_matching[i]:
                                if (
                                    shared_vertex_orientation[i][1]
                                    - shared_vertex_orientation[i][0]
                                    < 0
                                ):
                                    flip_edge.append(True)
                                else:
                                    flip_edge.append(False)

                            # Sum the data
                            for idx in range(0, len(matching_element)):
                                if flip_edge[idx]:
                                    matching_edge_data[idx] = np.flip(
                                        matching_edge_data[idx]
                                    )

                                my_edge_data += matching_edge_data[idx]

                            # Do not assing at the vertices
                            if lz_index == slice(None):
                                lz_index = slice(1, -1)
                            if ly_index == slice(None):
                                ly_index = slice(1, -1)
                            if lx_index == slice(None):
                                lx_index = slice(1, -1)
                            slice_copy = slice(1, -1)
                            global_dssum_field[e, lz_index, ly_index, lx_index] = (
                                np.copy(my_edge_data[slice_copy])
                            )

                            shared_edge_index += 1

        if msh.gdim >= 3:
            self.log.write("debug", "Adding faces")
            # Prepare data to send to other ranks:
            facet_send_buff = prepare_send_buffers(
                msh=msh,
                field=field,
                vef_to_rank_map=self.global_shared_efp_to_rank_map,
                data_to_fetch="facet",
            )

            # Send and receive the data
            (
                facet_sources,
                source_facet_el_id,
                source_facet_id,
                source_facet_x_coords,
                source_facet_y_coords,
                source_facet_z_coords,
                source_facet_data,
            ) = send_recv_data(
                rt=self.rt, send_buff=facet_send_buff, data_to_send="facet", lx=msh.lx
            )

            # Summ facets:
            for e in range(0, msh.nelv):

                # Facet data might be flipped or rotated so better check coordinates
                for facet in range(0, 6):

                    if (e, facet) in self.global_shared_efp_to_elem_map.keys():

                        # Check which other rank has this facet
                        shared_ranks = list(
                            self.global_shared_efp_to_rank_map[(e, facet)]
                        )
                        shared_elements = list(
                            self.global_shared_efp_to_elem_map[(e, facet)]
                        )
                        shared_facets = list(
                            self.global_shared_efp_to_facet_map[(e, facet)]
                        )

                        # Get the facet data from the different ranks
                        shared_facet_index = 0
                        for sf in range(0, len(shared_facets)):

                            source_index = list(facet_sources).index(shared_ranks[sf])

                            # Get the data from this source
                            shared_facet_el_id = source_facet_el_id[source_index]
                            shared_facet_id = source_facet_id[source_index]
                            shared_facet_coord_x = source_facet_x_coords[source_index]
                            shared_facet_coord_y = source_facet_y_coords[source_index]
                            shared_facet_coord_z = source_facet_z_coords[source_index]
                            shared_facet_data = source_facet_data[source_index]

                            # find the data that matches the element and facet id dictionary
                            el = shared_elements[sf]
                            facet_id = shared_facets[sf]
                            same_el = shared_facet_el_id == el
                            same_facet = shared_facet_id == facet_id
                            matching_index = np.where(same_el & same_facet)

                            matching_element = shared_facet_el_id[matching_index]
                            matching_facet = shared_facet_id[matching_index]
                            matching_facet_coord_x = shared_facet_coord_x[
                                matching_index
                            ]
                            matching_facet_coord_y = shared_facet_coord_y[
                                matching_index
                            ]
                            matching_facet_coord_z = shared_facet_coord_z[
                                matching_index
                            ]
                            matching_facet_data = shared_facet_data[matching_index]

                            # Get the facet location on my own elemenet
                            lz_index = facet_to_slice_map[facet][0]
                            ly_index = facet_to_slice_map[facet][1]
                            lx_index = facet_to_slice_map[facet][2]

                            # Get my own facet data and coordinates
                            my_facet_coord_x = msh.x[e, lz_index, ly_index, lx_index]
                            my_facet_coord_y = msh.y[e, lz_index, ly_index, lx_index]
                            my_facet_coord_z = msh.z[e, lz_index, ly_index, lx_index]
                            if shared_facet_index == 0:
                                my_facet_data = np.copy(
                                    local_dssum_field[e, lz_index, ly_index, lx_index]
                                )
                            else:
                                my_facet_data = np.copy(
                                    global_dssum_field[e, lz_index, ly_index, lx_index]
                                )

                            # Figure out if the facets are flipped.
                            ## First find the vertices of my facet
                            my_facet_vertices = facet_to_vertex_map[facet]

                            ## Now check how they are actually aligned to see if they are flipped
                            ### Find which are the shared vertices of my own facet vertices that are in each entry of shared element
                            ### Note that in general, each vertex in one element will have 1 matching vertex in another... otherwise something is weird
                            ### For axis 1
                            shared_vertex_idx_of_my_facet_vertex_0_ax1 = [
                                self.global_shared_evp_to_vertex_map[
                                    (e, my_facet_vertices[0][0])
                                ][
                                    np.where(
                                        np.array(
                                            self.global_shared_evp_to_elem_map[
                                                (e, my_facet_vertices[0][0])
                                            ]
                                        )
                                        == se
                                    )
                                ][
                                    0
                                ]
                                for se in matching_element
                            ]
                            shared_vertex_idx_of_my_facet_vertex_1_ax1 = [
                                self.global_shared_evp_to_vertex_map[
                                    (e, my_facet_vertices[0][1])
                                ][
                                    np.where(
                                        np.array(
                                            self.global_shared_evp_to_elem_map[
                                                (e, my_facet_vertices[0][1])
                                            ]
                                        )
                                        == se
                                    )
                                ][
                                    0
                                ]
                                for se in matching_element
                            ]
                            ### For axis 0
                            shared_vertex_idx_of_my_facet_vertex_0_ax0 = [
                                self.global_shared_evp_to_vertex_map[
                                    (e, my_facet_vertices[2][0])
                                ][
                                    np.where(
                                        np.array(
                                            self.global_shared_evp_to_elem_map[
                                                (e, my_facet_vertices[2][0])
                                            ]
                                        )
                                        == se
                                    )
                                ][
                                    0
                                ]
                                for se in matching_element
                            ]
                            shared_vertex_idx_of_my_facet_vertex_1_ax0 = [
                                self.global_shared_evp_to_vertex_map[
                                    (e, my_facet_vertices[2][1])
                                ][
                                    np.where(
                                        np.array(
                                            self.global_shared_evp_to_elem_map[
                                                (e, my_facet_vertices[2][1])
                                            ]
                                        )
                                        == se
                                    )
                                ][
                                    0
                                ]
                                for se in matching_element
                            ]

                            ### Create a list of how the shared vetices are actually oriented
                            shared_vertex_orientation_axis_1 = [
                                (
                                    int(shared_vertex_idx_of_my_facet_vertex_0_ax1[i]),
                                    int(shared_vertex_idx_of_my_facet_vertex_1_ax1[i]),
                                )
                                for i in range(len(matching_element))
                            ]
                            shared_vertex_orientation_axis_0 = [
                                (
                                    int(shared_vertex_idx_of_my_facet_vertex_0_ax0[i]),
                                    int(shared_vertex_idx_of_my_facet_vertex_1_ax0[i]),
                                )
                                for i in range(len(matching_element))
                            ]

                            ### Now compare. If the vertex index of the shared facet orientation are not increasing, then it is flipped.
                            ### Note: A more general way to do this is to check the sign of the difference of the vertex indices. If it is the same as the sign of the difference of the vertex indices of my facet, then it is not flipped.
                            ### however since my_facet_vertices is always increasing by construction in the dictionary, we can just check if the shared vertex indices are increasing.
                            flip_facet_axis_1 = []
                            flip_facet_axis_0 = []
                            for i in range(len(matching_element)):

                                if (
                                    shared_vertex_orientation_axis_1[i][1]
                                    - shared_vertex_orientation_axis_1[i][0]
                                    < 0
                                ):
                                    # if np.sign(shared_vertex_orientation_axis_1[i][1] - shared_vertex_orientation_axis_1[i][0]) != np.sign(my_facet_vertices[2][1] - my_facet_vertices[2][0]):
                                    flip_facet_axis_1.append(True)
                                else:
                                    flip_facet_axis_1.append(False)
                                if (
                                    shared_vertex_orientation_axis_0[i][1]
                                    - shared_vertex_orientation_axis_0[i][0]
                                    < 0
                                ):
                                    # if np.sign(shared_vertex_orientation_axis_0[i][1] - shared_vertex_orientation_axis_0[i][0]) != np.sign(my_facet_vertices[0][1] - my_facet_vertices[0][0]):
                                    flip_facet_axis_0.append(True)
                                else:
                                    flip_facet_axis_0.append(False)

                            # Sum the data
                            for idx in range(0, len(matching_element)):
                                if flip_facet_axis_1[idx]:
                                    matching_facet_data[idx] = np.flip(
                                        matching_facet_data[idx], axis=1
                                    )
                                if flip_facet_axis_0[idx]:
                                    matching_facet_data[idx] = np.flip(
                                        matching_facet_data[idx], axis=0
                                    )

                                my_facet_data += matching_facet_data[idx]

                            # Do not assing at the edges
                            if lz_index == slice(None):
                                lz_index = slice(1, -1)
                            if ly_index == slice(None):
                                ly_index = slice(1, -1)
                            if lx_index == slice(None):
                                lx_index = slice(1, -1)
                            slice_copy = slice(1, -1)
                            global_dssum_field[e, lz_index, ly_index, lx_index] = (
                                np.copy(my_facet_data[slice_copy, slice_copy])
                            )

                            shared_facet_index += 1

        self.log.write("info", "Global dssum computed")
        self.log.toc()

        return global_dssum_field


def find_local_shared_vef(  #
    vef_coords: np.ndarray = None,
    rtol: float = 1e-5,
    min_shared: int = 0,
    use_hashtable: bool = False,
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    list[int],
    list[int],
]:
    """
    Find the shared vertices/edges/facets in the local rank

    Here we will compare vertices, edge centers and facet centers to find the shared vertices/edges/facets in the local rank.

    Parameters
    ----------
    vef_coords : np.ndarray
        The coordinates of the vertices/edges/facets
    rtol : float
        The relative tolerance to use when comparing the coordinates
    min_shared : int
        The minimum number of shared vertices/edges/facets to consider them found in a rank.
        In 2D, min vertices = 4, min edges = 2
        In 3D, min vertices = 8, min edges = 4, min facets = 2
    use_hashtable : bool
        If True, use a hash table to speed up the search
        The has table will compare exact values, so only do if you are sure that the coordinates are exact

    Returns
    -------
    tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], list[int], list[int]
        The shared vertices/edges/facets to element map, the shared vertices/edges/facets to vertex/edge/facet map, the incomplete vertices/edges/facets element list and the incomplete vertices/edges/facets vertex/edge/facet list
    """

    # Define the maps
    shared_e_vef_p_to_elem_map = {}
    shared_e_vef_p_to_vef_map = {}

    if use_hashtable:
        # Iterate over each element and vertex/edge/face adding to the hash table
        hash_table_e = {}
        hash_table_vef = {}
        for e in range(0, vef_coords.shape[0]):
            for vef in range(0, vef_coords.shape[1]):
                hash_key = tuple(vef_coords[e, vef])
                if hash_key in hash_table_e.keys():
                    hash_table_e[hash_key].append(e)
                    hash_table_vef[hash_key].append(vef)
                else:
                    hash_table_e[hash_key] = [e]
                    hash_table_vef[hash_key] = [vef]

        # Iterate over each element and vertex/edge/face again, and now populate the shared maps
        for e in range(0, vef_coords.shape[0]):
            for vef in range(0, vef_coords.shape[1]):
                hash_key = tuple(vef_coords[e, vef])
                shared_e_vef_p_to_elem_map[(e, vef)] = np.array(hash_table_e[hash_key])
                shared_e_vef_p_to_vef_map[(e, vef)] = np.array(hash_table_vef[hash_key])

    else:
        # Iterate over each element
        for e in range(0, vef_coords.shape[0]):
            # Iterate over each vertex/edge/facet
            for vef in range(0, vef_coords.shape[1]):
                same_x = np.isclose(
                    vef_coords[e, vef, 0], vef_coords[:, :, 0], rtol=rtol
                )
                same_y = np.isclose(
                    vef_coords[e, vef, 1], vef_coords[:, :, 1], rtol=rtol
                )
                same_z = np.isclose(
                    vef_coords[e, vef, 2], vef_coords[:, :, 2], rtol=rtol
                )
                same_geometric_entity = np.where(same_x & same_y & same_z)

                matching_elem = same_geometric_entity[0]
                matching_geometric_entity = same_geometric_entity[1]

                # Assig the matching element and vertex/edge/facet to the dictionary
                shared_e_vef_p_to_elem_map[(e, vef)] = matching_elem
                shared_e_vef_p_to_vef_map[(e, vef)] = matching_geometric_entity

    # If the number of shared vertices/edges/facets is less than min_shared, then the vertex/edge/facet is incomplete
    # and the rest might be in anothe rank
    incomplete_e_vef_p_elem = []
    incomplete_e_vef_p_vef = []
    for elem_vef_pair in shared_e_vef_p_to_elem_map.keys():
        if len(shared_e_vef_p_to_elem_map[elem_vef_pair]) < min_shared:
            incomplete_e_vef_p_elem.append(elem_vef_pair[0])
            incomplete_e_vef_p_vef.append(elem_vef_pair[1])

    return (
        shared_e_vef_p_to_elem_map,
        shared_e_vef_p_to_vef_map,
        incomplete_e_vef_p_elem,
        incomplete_e_vef_p_vef,
    )


def find_global_shared_evp(
    rt: Router,
    vef_coords: np.ndarray,
    incomplete_e_vef_p_elem: list[int],
    incomplete_e_vef_p_vef: list[int],
    rtol: float = 1e-5,
    use_hashtable: bool = False,
    max_simultaneous_sends: int = 1,
) -> tuple[
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
]:
    """
    Find the shared vertices/edges/facets in the global ranks

    Here we will compare the incomplete vertices/edges/facets in the local rank with the vertices/edges/facets in the global ranks.

    Parameters
    ----------
    rt : Router
        The router object
    vef_coords : np.ndarray
        The coordinates of the vertices/edges/facets
    incomplete_e_vef_p_elem : list[int]
        The incomplete vertices/edges/facets element list
    incomplete_e_vef_p_vef : list[int]
        The incomplete vertices/edges/facets vertex/edge/facet list
    rtol : float
        The relative tolerance to use when comparing the coordinates
    use_hashtable : bool
        If True, use a hash table to speed up the search
        The has table will compare exact values, so only do if you are sure that the coordinates are exact

    Returns
    -------
    tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray], dict[tuple[int, int], np.ndarray]
        The shared vertices/edges/facets to rank map, the shared vertices/edges/facets to element map and the shared vertices/edges/facets to vertex/edge/facet map
    """

    # Set up send buffers
    local_incomplete_el_id = np.array(incomplete_e_vef_p_elem)
    local_incomplete_vef_id = np.array(incomplete_e_vef_p_vef)
    local_incomplete_vef_coords = vef_coords[
        incomplete_e_vef_p_elem, incomplete_e_vef_p_vef
    ]
    
    # Create global dictionaries
    global_shared_e_vef_p_to_rank_map = {}
    global_shared_e_vef_p_to_elem_map = {}
    global_shared_e_vef_p_to_vef_map = {}

    # Set up the iterations
    n_iterations = int(np.ceil(rt.comm.Get_size() / max_simultaneous_sends))
    ranks_sent_to = []

    for it in range(0, n_iterations):

        # Use round_robin to get the ranks to send to in this iteration
        schedule = round_robin_schedule(rt.comm.Get_rank(), rt.comm.Get_size(), it, max_simultaneous_sends)
        destinations = [sche for sche in schedule if (sche not in ranks_sent_to) and (sche != rt.comm.Get_rank())]
        ranks_sent_to.extend(destinations)

        # Send and recieve
        sources, source_incomplete_el_id = rt.all_to_all(
            destination=destinations,
            data=local_incomplete_el_id,
            dtype=local_incomplete_el_id.dtype,
        )
        _, source_incomplete_vef_id = rt.all_to_all(
            destination=destinations,
            data=local_incomplete_vef_id,
            dtype=local_incomplete_vef_id.dtype,
        )
        _, source_incomplete_vef_coords = rt.all_to_all(
            destination=destinations,
            data=local_incomplete_vef_coords,
            dtype=local_incomplete_vef_coords.dtype,
        )

        # Reshape flattened arrays
        for i in range(0, len(source_incomplete_vef_coords)):
            source_incomplete_vef_coords[i] = source_incomplete_vef_coords[i].reshape(-1, 3)

        # Go through the data in each other rank.
        for source_idx, source_vef in enumerate(source_incomplete_vef_coords):

            if use_hashtable:
                # Iterate over each element and vertex/edge/face from the source rank and add to the hash table
                hash_table_rank = {}
                hash_table_e = {}
                hash_table_vef = {}
                for idx in range(0, source_vef.shape[0]):
                    hash_key = tuple(source_vef[idx])
                    if hash_key in hash_table_e.keys():
                        hash_table_rank[hash_key].append(sources[source_idx])
                        hash_table_e[hash_key].append(
                            source_incomplete_el_id[source_idx][idx]
                        )
                        hash_table_vef[hash_key].append(
                            source_incomplete_vef_id[source_idx][idx]
                        )
                    else:
                        hash_table_rank[hash_key] = [sources[source_idx]]
                        hash_table_e[hash_key] = [source_incomplete_el_id[source_idx][idx]]
                        hash_table_vef[hash_key] = [
                            source_incomplete_vef_id[source_idx][idx]
                        ]

                # Now Iterate over my own incomplete points and check if they are in the hash table, then poulate the shared maps
                for e_vef_pair in range(0, len(incomplete_e_vef_p_elem)):

                    e = incomplete_e_vef_p_elem[e_vef_pair]
                    vef = incomplete_e_vef_p_vef[e_vef_pair]
                    hash_key = tuple(vef_coords[e, vef])

                    if hash_key in hash_table_e.keys():

                        if (e, vef) in global_shared_e_vef_p_to_rank_map.keys():
                            global_shared_e_vef_p_to_rank_map[(e, vef)] = np.append(
                                global_shared_e_vef_p_to_rank_map[(e, vef)],
                                np.array(hash_table_rank[hash_key]),
                            )
                            global_shared_e_vef_p_to_elem_map[(e, vef)] = np.append(
                                global_shared_e_vef_p_to_elem_map[(e, vef)],
                                np.array(hash_table_e[hash_key]),
                            )
                            global_shared_e_vef_p_to_vef_map[(e, vef)] = np.append(
                                global_shared_e_vef_p_to_vef_map[(e, vef)],
                                np.array(hash_table_vef[hash_key]),
                            )
                        else:
                            global_shared_e_vef_p_to_rank_map[(e, vef)] = np.array(
                                hash_table_rank[hash_key]
                            )
                            global_shared_e_vef_p_to_elem_map[(e, vef)] = np.array(
                                hash_table_e[hash_key]
                            )
                            global_shared_e_vef_p_to_vef_map[(e, vef)] = np.array(
                                hash_table_vef[hash_key]
                            )

            else:
                # Loop through all my own incomplete element vertex pairs
                for e_vef_pair in range(0, len(incomplete_e_vef_p_elem)):

                    # Check where my incomplete vertex pair coordinates match with the incomplete ...
                    # ... vertex pair coordinates of the other rank
                    e = incomplete_e_vef_p_elem[e_vef_pair]
                    vef = incomplete_e_vef_p_vef[e_vef_pair]
                    same_x = np.isclose(vef_coords[e, vef, 0], source_vef[:, 0], rtol=rtol)
                    same_y = np.isclose(vef_coords[e, vef, 1], source_vef[:, 1], rtol=rtol)
                    same_z = np.isclose(vef_coords[e, vef, 2], source_vef[:, 2], rtol=rtol)
                    same_vef = np.where(same_x & same_y & same_z)

                    # If we find a match assign it in the global dictionaries
                    if len(same_vef[0]) > 0:
                        matching_id = same_vef[0]
                        sources_list = (
                            np.ones_like(source_incomplete_vef_id[source_idx][matching_id])
                            * sources[source_idx]
                        )
                        if (e, vef) in global_shared_e_vef_p_to_rank_map.keys():
                            global_shared_e_vef_p_to_rank_map[(e, vef)] = np.append(
                                global_shared_e_vef_p_to_rank_map[(e, vef)], sources_list
                            )
                            global_shared_e_vef_p_to_elem_map[(e, vef)] = np.append(
                                global_shared_e_vef_p_to_elem_map[(e, vef)],
                                source_incomplete_el_id[source_idx][matching_id],
                            )
                            global_shared_e_vef_p_to_vef_map[(e, vef)] = np.append(
                                global_shared_e_vef_p_to_vef_map[(e, vef)],
                                source_incomplete_vef_id[source_idx][matching_id],
                            )
                        else:
                            global_shared_e_vef_p_to_rank_map[(e, vef)] = sources_list
                            global_shared_e_vef_p_to_elem_map[(e, vef)] = (
                                source_incomplete_el_id[source_idx][matching_id]
                            )
                            global_shared_e_vef_p_to_vef_map[(e, vef)] = (
                                source_incomplete_vef_id[source_idx][matching_id]
                            )

    return (
        global_shared_e_vef_p_to_rank_map,
        global_shared_e_vef_p_to_elem_map,
        global_shared_e_vef_p_to_vef_map,
    )


def prepare_send_buffers(
    msh: Mesh = None,
    field: np.ndarray = None,
    vef_to_rank_map: dict[tuple[int, int], np.ndarray] = None,
    data_to_fetch: str = None,
) -> dict:
    """
    Prepare the data to send to other ranks

    Parameters
    ----------
    msh : Mesh
        The mesh object
    field : np.ndarray
        The field to send
    vef_to_rank_map : dict[tuple[int, int], np.ndarray]
        The map of vertices/edges/facets to ranks
    data_to_fetch : str
        The data to fetch, either vertex, edge or facet

    Returns
    -------
    dict
        The data to send to other ranks
    """

    # Prepare vertices to send to other ranks:
    send_buff = {}

    # Select the data to fetch
    if data_to_fetch == "vertex":
        n_vef = msh.vertices.shape[1]
        df = vd
    elif data_to_fetch == "edge":
        n_vef = msh.edge_centers.shape[1]
        df = ed
    elif data_to_fetch == "facet":
        n_vef = 6
        df = fd

    # Iterate over all elements
    for e in range(0, msh.nelv):

        # Iterate over the vertex/edge/facet of the element
        for vef in range(0, n_vef):

            if (e, vef) in vef_to_rank_map.keys():

                # Check which other rank has this vertex/edge/facet
                shared_ranks = list(vef_to_rank_map[(e, vef)])

                # Get the data for this vertex/edge/facet on this element
                my_vef_coord_x = df(msh.x, e, vef)
                my_vef_coord_y = df(msh.y, e, vef)
                my_vef_coord_z = df(msh.z, e, vef)
                my_vef_data = df(field, e, vef)

                # Go over all ranks that share this vertex/edge/facet
                # and for that rank, append the data to the send buffer
                for rank in list(np.unique(shared_ranks)):

                    if rank not in send_buff.keys():
                        send_buff[rank] = {}
                        send_buff[rank]["e"] = []
                        send_buff[rank][data_to_fetch] = []
                        send_buff[rank]["x_coords"] = []
                        send_buff[rank]["y_coords"] = []
                        send_buff[rank]["z_coords"] = []
                        send_buff[rank]["data"] = []

                    send_buff[rank]["e"].append(e)
                    send_buff[rank][data_to_fetch].append(vef)
                    send_buff[rank]["x_coords"].append(my_vef_coord_x)
                    send_buff[rank]["y_coords"].append(my_vef_coord_y)
                    send_buff[rank]["z_coords"].append(my_vef_coord_z)
                    send_buff[rank]["data"].append(my_vef_data)

    return send_buff


def send_recv_data(
    rt: Router = None, send_buff: dict = None, data_to_send: str = None, lx: int = None
) -> tuple[
    list[int],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
]:
    """
    Send and receive the data

    Parameters
    ----------
    rt : Router
        The router object
    send_buff : dict
        The data to send to other ranks
    data_to_send : str
        The data to send, either vertex, edge or facet
    lx : int
        The number of gll points in one direction of the element

    Returns
    -------
    tuple[list[int], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]
        The source ranks, the source element id, the source vertex/edge/facet id, the source x coordinates, the source y coordinates, the source z coordinates and the source data
    """

    # Populate a list with the destinations
    destinations = [rank for rank in send_buff.keys()]

    # Populate individual list for each data that must be sent to each destination
    # 1. The element id
    local_vef_el_id = [np.array(send_buff[rank]["e"]) for rank in destinations]
    # 2. The vertex/edge/facet id inside the element
    local_vef_id = [np.array(send_buff[rank][data_to_send]) for rank in destinations]
    # 3. The x coordinates if the vertex/edge/facet
    local_vef_x_coords = [
        np.array(send_buff[rank]["x_coords"]) for rank in destinations
    ]
    # 4. The y coordinates if the vertex/edge/facet
    local_vef_y_coords = [
        np.array(send_buff[rank]["y_coords"]) for rank in destinations
    ]
    # 5. The z coordinates if the vertex/edge/facet
    local_vef_z_coords = [
        np.array(send_buff[rank]["z_coords"]) for rank in destinations
    ]
    # 6. The data of the vertex/edge/facet that will be summed
    local_vef_data = [np.array(send_buff[rank]["data"]) for rank in destinations]

    # Send and recieve the data
    vef_sources, source_vef_el_id = rt.all_to_all(
        destination=destinations, data=local_vef_el_id, dtype=local_vef_el_id[0].dtype
    )
    _, source_vef_id = rt.all_to_all(
        destination=destinations, data=local_vef_id, dtype=local_vef_id[0].dtype
    )
    _, source_vef_x_coords = rt.all_to_all(
        destination=destinations,
        data=local_vef_x_coords,
        dtype=local_vef_x_coords[0].dtype,
    )
    _, source_vef_y_coords = rt.all_to_all(
        destination=destinations,
        data=local_vef_y_coords,
        dtype=local_vef_y_coords[0].dtype,
    )
    _, source_vef_z_coords = rt.all_to_all(
        destination=destinations,
        data=local_vef_z_coords,
        dtype=local_vef_z_coords[0].dtype,
    )
    _, source_vef_data = rt.all_to_all(
        destination=destinations, data=local_vef_data, dtype=local_vef_data[0].dtype
    )

    # Reshape the flattened arrays
    if data_to_send == "vertex":
        for i in range(0, len(source_vef_x_coords)):
            # No need to reshape the vertex data
            pass
    elif data_to_send == "edge":
        for i in range(0, len(source_vef_x_coords)):
            source_vef_x_coords[i] = source_vef_x_coords[i].reshape(-1, lx)
            source_vef_y_coords[i] = source_vef_y_coords[i].reshape(-1, lx)
            source_vef_z_coords[i] = source_vef_z_coords[i].reshape(-1, lx)
            source_vef_data[i] = source_vef_data[i].reshape(-1, lx)
    elif data_to_send == "facet":
        for i in range(0, len(source_vef_x_coords)):
            source_vef_x_coords[i] = source_vef_x_coords[i].reshape(-1, lx, lx)
            source_vef_y_coords[i] = source_vef_y_coords[i].reshape(-1, lx, lx)
            source_vef_z_coords[i] = source_vef_z_coords[i].reshape(-1, lx, lx)
            source_vef_data[i] = source_vef_data[i].reshape(-1, lx, lx)

    return (
        vef_sources,
        source_vef_el_id,
        source_vef_id,
        source_vef_x_coords,
        source_vef_y_coords,
        source_vef_z_coords,
        source_vef_data,
    )

def round_robin_schedule(rank, size, iteration, chunksize):
    if chunksize >= size:
        chunksize = size
    return [(rank + iteration + i) % size for i in range(1, chunksize + 1)]