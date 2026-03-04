"""
Contains method for slicing elements

 All comments based on a typical reference element
 See the drawings below:
 Current mapping:

 2D Mapping
  v2----e1----v3
  |            |
  |            |
  e2           e3
  |            |
  |            |
  v0----e0----v1

 3D Mapping
                             f3 (back)
                            / 

                v6---------e5---------v7
               /|                    /|
              / |                   / |
            e6  e10     f5         e7 e11
            /   |                 /   |
           v4---------e4---------v5   |  
      f0-- |    |                |    |    --f1
           |    |                |    |     
           |   v2---------e1-----|----v3
           |   /                 |   /
           |  e2        f4       |  e3
          e8 /                  e9 /
           |/                    |/
           v0---------e0---------v1

                / 
               f2(front)

"""

from typing import Union
import numpy as np

## For 2D we have vertices
vertex_to_slice_map_2d = {"vertex": ("lz", "ly", "lx")}
vertex_to_slice_map_2d[0] = (0, 0, 0)  # Equivalent to [0, 0, 0] # bottom left vertex
vertex_to_slice_map_2d[1] = (0, 0, -1)  # Equivalent to [0, 0, -1] # bottom right vertex
vertex_to_slice_map_2d[2] = (0, -1, 0)  # Equivalent to [0, -1, 0] # top left vertex
vertex_to_slice_map_2d[3] = (0, -1, -1)  # Equivalent to [0, -1, -1] # top right vertex

## For 3D we have vertices
vertex_to_slice_map_3d = {"vertex": ("lz", "ly", "lx")}
vertex_to_slice_map_3d[0] = (
    0,
    0,
    0,
)  # Equivalent to [0, 0, 0] # bottom left front vertex
vertex_to_slice_map_3d[1] = (
    0,
    0,
    -1,
)  # Equivalent to [0, 0, -1] # bottom right front vertex
vertex_to_slice_map_3d[2] = (
    0,
    -1,
    0,
)  # Equivalent to [0, -1, 0] # bottom left back vertex
vertex_to_slice_map_3d[3] = (
    0,
    -1,
    -1,
)  # Equivalent to [0, -1, -1] # bottom right back vertex
vertex_to_slice_map_3d[4] = (
    -1,
    0,
    0,
)  # Equivalent to [-1, 0, 0] # top left front vertex
vertex_to_slice_map_3d[5] = (
    -1,
    0,
    -1,
)  # Equivalent to [-1, 0, -1] # top right front vertex
vertex_to_slice_map_3d[6] = (
    -1,
    -1,
    0,
)  # Equivalent to [-1, -1, 0] # top left back vertex
vertex_to_slice_map_3d[7] = (
    -1,
    -1,
    -1,
)  # Equivalent to [-1, -1, -1] # top right back vertex

## For 2D we have edges
edge_to_slice_map_2d = {"edge": ("lz", "ly", "lx")}
edge_to_slice_map_2d[0] = (0, 0, slice(None))  # Equivalent to [0,0,:]  # bottom edge
edge_to_slice_map_2d[1] = (0, -1, slice(None))  # Equivalent to [0,-1,:]  # top edge
edge_to_slice_map_2d[2] = (0, slice(None), 0)  # Equivalent to [0,:,0]  # left edge
edge_to_slice_map_2d[3] = (0, slice(None), -1)  # Equivalent to [0,:,-1]  # right edge
## For 2D we can get edges from 2 vertices connected
vertex_pair_to_edge_map_2d = {"vertex_pair": "edge_id"}
vertex_pair_to_edge_map_2d[(0, 1)] = 0  # Equivalent to bottom edge
vertex_pair_to_edge_map_2d[(2, 3)] = 1  # Equivalent to top edge
vertex_pair_to_edge_map_2d[(0, 2)] = 2  # Equivalent to left edge
vertex_pair_to_edge_map_2d[(1, 3)] = 3  # Equivalent to right edge
### Allow to get the same result from the reverse order
vertex_pair_to_edge_map_2d[(1, 0)] = 0  # Equivalent to bottom edge
vertex_pair_to_edge_map_2d[(3, 2)] = 1  # Equivalent to top edge
vertex_pair_to_edge_map_2d[(2, 0)] = 2  # Equivalent to left edge
vertex_pair_to_edge_map_2d[(3, 1)] = 3  # Equivalent to right edge
### We also have the edge to vertex map
edge_to_vertex_map_2d = {"edge": ("vertex0", "vertex1")}
edge_to_vertex_map_2d[0] = (0, 1)  # Equivalent to bottom edge
edge_to_vertex_map_2d[1] = (2, 3)  # Equivalent to top edge
edge_to_vertex_map_2d[2] = (0, 2)  # Equivalent to left edge
edge_to_vertex_map_2d[3] = (1, 3)  # Equivalent to right edge

## For 3D we have edges
edge_to_slice_map_3d = {"edge": ("lz", "ly", "lx")}
edge_to_slice_map_3d[0] = (
    0,
    0,
    slice(None),
)  # Equivalent to [0,0,:]  # front face bottom edge
edge_to_slice_map_3d[1] = (
    0,
    -1,
    slice(None),
)  # Equivalent to [0,-1,:]  # back face bottom edge
edge_to_slice_map_3d[2] = (
    0,
    slice(None),
    0,
)  # Equivalent to [0,:,0]  # left face bottom edge
edge_to_slice_map_3d[3] = (
    0,
    slice(None),
    -1,
)  # Equivalent to [0,:,-1]  # right face bottom edge
edge_to_slice_map_3d[4] = (
    -1,
    0,
    slice(None),
)  # Equivalent to [0,0,:]  # front face top edge
edge_to_slice_map_3d[5] = (
    -1,
    -1,
    slice(None),
)  # Equivalent to [0,-1,:]  # back face top edge
edge_to_slice_map_3d[6] = (
    -1,
    slice(None),
    0,
)  # Equivalent to [0,:,0]  # left face top edge
edge_to_slice_map_3d[7] = (
    -1,
    slice(None),
    -1,
)  # Equivalent to [0,:,-1]  # right face top edge
edge_to_slice_map_3d[8] = (
    slice(None),
    0,
    0,
)  # Equivalent to [:,0,0]  # front face left edge
edge_to_slice_map_3d[9] = (
    slice(None),
    0,
    -1,
)  # Equivalent to [:,0,-1]  # front face right edge
edge_to_slice_map_3d[10] = (
    slice(None),
    -1,
    0,
)  # Equivalent to [:,-1,0]  # back face left edge
edge_to_slice_map_3d[11] = (
    slice(None),
    -1,
    -1,
)  # Equivalent to [:,-1,-1]  # back face right edge
### For 3D we can get edges from 2 vertices connected
vertex_pair_to_edge_map_3d = {"vertex_pair": "edge_id"}
vertex_pair_to_edge_map_3d[(0, 1)] = 0  # Equivalent to front face bottom edge
vertex_pair_to_edge_map_3d[(2, 3)] = 1  # Equivalent to back face bottom edge
vertex_pair_to_edge_map_3d[(0, 2)] = 2  # Equivalent to left face bottom edge
vertex_pair_to_edge_map_3d[(1, 3)] = 3  # Equivalent to right face bottom edge
vertex_pair_to_edge_map_3d[(4, 5)] = 4  # Equivalent to front face top edge
vertex_pair_to_edge_map_3d[(6, 7)] = 5  # Equivalent to back face top edge
vertex_pair_to_edge_map_3d[(4, 6)] = 6  # Equivalent to left face top edge
vertex_pair_to_edge_map_3d[(5, 7)] = 7  # Equivalent to right face top edge
vertex_pair_to_edge_map_3d[(0, 4)] = 8  # Equivalent to front face left edge
vertex_pair_to_edge_map_3d[(1, 5)] = 9  # Equivalent to front face right edge
vertex_pair_to_edge_map_3d[(2, 6)] = 10  # Equivalent to back face left edge
vertex_pair_to_edge_map_3d[(3, 7)] = 11  # Equivalent to back face right edge
### Allow to get the same result from the reverse order
vertex_pair_to_edge_map_3d[(1, 0)] = 0  # Equivalent to front face bottom edge
vertex_pair_to_edge_map_3d[(3, 2)] = 1  # Equivalent to back face bottom edge
vertex_pair_to_edge_map_3d[(2, 0)] = 2  # Equivalent to left face bottom edge
vertex_pair_to_edge_map_3d[(3, 1)] = 3  # Equivalent to right face bottom edge
vertex_pair_to_edge_map_3d[(5, 4)] = 4  # Equivalent to front face top edge
vertex_pair_to_edge_map_3d[(7, 6)] = 5  # Equivalent to back face top edge
vertex_pair_to_edge_map_3d[(6, 4)] = 6  # Equivalent to left face top edge
vertex_pair_to_edge_map_3d[(7, 5)] = 7  # Equivalent to right face top edge
vertex_pair_to_edge_map_3d[(4, 0)] = 8  # Equivalent to front face left edge
vertex_pair_to_edge_map_3d[(5, 1)] = 9  # Equivalent to front face right edge
vertex_pair_to_edge_map_3d[(6, 2)] = 10  # Equivalent to back face left edge
vertex_pair_to_edge_map_3d[(7, 3)] = 11  # Equivalent to back face right edge
### We also have the edge to vertex map
edge_to_vertex_map_3d = {"edge": ("vertex0", "vertex1")}
edge_to_vertex_map_3d[0] = (0, 1)  # Equivalent to front face bottom edge
edge_to_vertex_map_3d[1] = (2, 3)  # Equivalent to back face bottom edge
edge_to_vertex_map_3d[2] = (0, 2)  # Equivalent to left face bottom edge
edge_to_vertex_map_3d[3] = (1, 3)  # Equivalent to right face bottom edge
edge_to_vertex_map_3d[4] = (4, 5)  # Equivalent to front face top edge
edge_to_vertex_map_3d[5] = (6, 7)  # Equivalent to back face top edge
edge_to_vertex_map_3d[6] = (4, 6)  # Equivalent to left face top edge
edge_to_vertex_map_3d[7] = (5, 7)  # Equivalent to right face top edge
edge_to_vertex_map_3d[8] = (0, 4)  # Equivalent to front face left edge
edge_to_vertex_map_3d[9] = (1, 5)  # Equivalent to front face right edge
edge_to_vertex_map_3d[10] = (2, 6)  # Equivalent to back face left edge
edge_to_vertex_map_3d[11] = (3, 7)  # Equivalent to back face right edge

## For 3D and above we have facets
facet_to_slice_map = {"facet": ("lz", "ly", "lx")}
facet_to_slice_map[0] = (
    slice(None),
    slice(None),
    0,
)  # Equivalent to [:, :, 0] # Left facet
facet_to_slice_map[1] = (
    slice(None),
    slice(None),
    -1,
)  # Equivalent to [:, :, -1] # Right facet
facet_to_slice_map[2] = (
    slice(None),
    0,
    slice(None),
)  # Equivalent to [:, 0, :] # Front facet
facet_to_slice_map[3] = (
    slice(None),
    -1,
    slice(None),
)  # Equivalent to [:, -1, :] # Back facet
facet_to_slice_map[4] = (
    0,
    slice(None),
    slice(None),
)  # Equivalent to [0, :, :] # Bottom facet
facet_to_slice_map[5] = (
    -1,
    slice(None),
    slice(None),
)  # Equivalent to [-1, :, :] # Top facet
### For 3D we can get facets from 4 pairs of vertices connected,
### each pair representing one axis of the 2D data facet data
facet_to_vertex_map = {
    "facet": (
        ("vertex0", "vertex1", "axis"),
        ("vertex2", "vertex3", "axis"),
        ("vertex4", "vertex5", "axis"),
        ("vertex6", "vertex7", "axis"),
    )
}
facet_to_vertex_map[0] = (
    (0, 2, 1),
    (4, 6, 1),
    (0, 4, 0),
    (2, 6, 0),
)  # Equivalent to left facet
facet_to_vertex_map[1] = (
    (1, 3, 1),
    (5, 7, 1),
    (1, 5, 0),
    (3, 7, 0),
)  # Equivalent to right facet
facet_to_vertex_map[2] = (
    (0, 1, 1),
    (4, 5, 1),
    (0, 4, 0),
    (1, 5, 0),
)  # Equivalent to front facet
facet_to_vertex_map[3] = (
    (2, 3, 1),
    (6, 7, 1),
    (2, 6, 0),
    (3, 7, 0),
)  # Equivalent to back facet
facet_to_vertex_map[4] = (
    (0, 1, 1),
    (2, 3, 1),
    (0, 2, 0),
    (1, 3, 0),
)  # Equivalent to bottom facet
facet_to_vertex_map[5] = (
    (4, 5, 1),
    (6, 7, 1),
    (4, 6, 0),
    (5, 7, 0),
)  # Equivalent to top facet


def fetch_elem_vertex_data(
    field: np.ndarray = None,
    elem: Union[int, list[int]] = None,
    vertex: Union[int, list[int]] = None,
):
    """

    Fetch the data from the field for the given element and vertex pair

    Parameters
    ----------
    field : np.ndarray
        The field data  to be sliced.
        The field shape should be (nelv, lz, ly, lx) for 3D and 2D. 2D has lz = 1.

    elem : Union[int, list[int]]
        The element index or list of element indices

    vertex : Union[int, list[int]]
        The vertex index or list of vertex indices

    Notes
    -----

    The restriction is that if vertex is a list, elem must be a list of the same length.
    This because then we retrieve the data for vertex[i] from elem[i].

    If vertex is an integer, then that means that we will retieve the same vertex from all.
    In this case:
    elem can be an interger, which returns the data for the elem and vertex pair.
    elem can be a list of integers, which returns the data for the list of elem and the same vertex.
    elem can be None, which returns the data for all elements and the same vertex.

    Returns
    -------
    np.ndarray
        The data for the given element and vertex pair
    """

    if field.shape[1] > 1:
        vertex_to_slice_map = vertex_to_slice_map_3d
    else:
        vertex_to_slice_map = vertex_to_slice_map_2d

    # Get the slice tuple for each element in the list
    if isinstance(elem, list):
        elem_index = [e for e in elem]
    elif isinstance(elem, int):
        elem_index = elem
    else:
        elem_index = slice(None)

    if isinstance(vertex, list):
        lz_index = [vertex_to_slice_map[v][0] for v in vertex]
        ly_index = [vertex_to_slice_map[v][1] for v in vertex]
        lx_index = [vertex_to_slice_map[v][2] for v in vertex]
    elif isinstance(vertex, int):
        lz_index = vertex_to_slice_map[vertex][0]
        ly_index = vertex_to_slice_map[vertex][1]
        lx_index = vertex_to_slice_map[vertex][2]
    else:
        raise TypeError("Vertex must be an integer or a list of integers")

    # Retrieve the data if inputs are list of corresponding element and facet pairs
    if isinstance(elem, list) and isinstance(vertex, list):
        # Create slices
        slices = [
            (elem_index[i], lz_index[i], ly_index[i], lx_index[i])
            for i in range(len(elem_index))
        ]
        # Retrieve the data
        vertex_data = np.array([field[s] for s in slices])

    # Retrieve the same facet from one element, elements in a list, or all elements
    elif (
        isinstance(elem, list) or isinstance(elem, int) or isinstance(elem, type(None))
    ) and isinstance(vertex, int):
        slices = (elem_index, lz_index, ly_index, lx_index)
        vertex_data = field[slices]

    else:
        raise TypeError(
            "Invalid input: If vertex is a list, elem must be a list of the same length"
        )

    return vertex_data


def fetch_elem_edge_data(
    field: np.ndarray = None,
    elem: Union[int, list[int]] = None,
    edge: Union[int, list[int], list[tuple[int, int]]] = None,
):
    """
    Fetch the data from the field for the given element and edge pair

    Parameters
    ----------
    field : np.ndarray
        The field data  to be sliced.
        The field shape should be (nelv, lz, ly, lx) for 3D and 2D. 2D has lz = 1.

    elem : Union[int, list[int]]
        The element index or list of element indices

    edge : Union[int, list[int], list[tuple[int, int]]]
        The edge index or list of edge indices or list of tuples of 2 vertex indices

    Notes
    -----
    The restriction is that if edge is a list, elem must be a list of the same length.
    This because then we retrieve the data for edge[i] from elem[i].

    If edge is an integer, then that means that we will retieve the same edge from all.
    In this case:
    elem can be an interger, which returns the data for the elem and edge pair.
    elem can be a list of integers, which returns the data for the list of elem and the same edge.
    elem can be None, which returns the data for all elements and the same edge.

    If edge is a list of tuples, then we retrieve the data for the edge between the two vertices in the tuple.

    Returns
    -------
    np.ndarray
        The data for the given element and edge pair
    """

    if field.shape[1] > 1:
        edge_to_slice_map = edge_to_slice_map_3d
        vertex_pair_to_edge_map = vertex_pair_to_edge_map_3d
    else:
        edge_to_slice_map = edge_to_slice_map_2d
        vertex_pair_to_edge_map = vertex_pair_to_edge_map_2d

    # Get the slice tuple for each element in the list
    if isinstance(elem, list):
        elem_index = [e for e in elem]
    elif isinstance(elem, int):
        elem_index = elem
    else:
        elem_index = slice(None)

    if isinstance(edge, list):
        if isinstance(edge[0], tuple):
            lz_index = [edge_to_slice_map[vertex_pair_to_edge_map[e]][0] for e in edge]
            ly_index = [edge_to_slice_map[vertex_pair_to_edge_map[e]][1] for e in edge]
            lx_index = [edge_to_slice_map[vertex_pair_to_edge_map[e]][2] for e in edge]
        elif (
            isinstance(edge[0], int)
            or isinstance(edge[0], np.int64)
            or isinstance(edge[0], np.int32)
        ):
            lz_index = [edge_to_slice_map[e][0] for e in edge]
            ly_index = [edge_to_slice_map[e][1] for e in edge]
            lx_index = [edge_to_slice_map[e][2] for e in edge]
        else:
            raise TypeError(
                "Edge must be an integer, a list of integers, or a list of tuples of integers"
            )
    elif isinstance(edge, int):
        lz_index = edge_to_slice_map[edge][0]
        ly_index = edge_to_slice_map[edge][1]
        lx_index = edge_to_slice_map[edge][2]
    else:
        raise TypeError(
            "Edge must be an integer, a list of integers or a list of tuples of integers"
        )

    # Retrieve the data if inputs are list of corresponding element and facet pairs
    if isinstance(elem, list) and isinstance(edge, list):
        # Create slices
        slices = [
            (elem_index[i], lz_index[i], ly_index[i], lx_index[i])
            for i in range(len(elem_index))
        ]
        # Retrieve the data
        edge_data = np.array([field[s] for s in slices])

    # Retrieve the same facet from one element, elements in a list, or all elements
    elif (
        isinstance(elem, list) or isinstance(elem, int) or isinstance(elem, type(None))
    ) and isinstance(edge, int):
        slices = (elem_index, lz_index, ly_index, lx_index)
        edge_data = field[slices]

    else:
        raise TypeError(
            "Invalid input: If edge is a list, elem must be a list of the same length"
        )

    return edge_data


def fetch_elem_facet_data(
    field: np.ndarray = None,
    elem: Union[int, list[int]] = None,
    facet: Union[int, list[int]] = None,
):
    """
    Fetch the data from the field for the given element and facet pair

    Parameters
    ----------
    field : np.ndarray
        The field data  to be sliced.
        The field shape should be (nelv, lz, ly, lx) for 3D and 2D.

    elem : Union[int, list[int]]
        The element index or list of element indices

    facet : Union[int, list[int]]
        The facet index or list of facet indices

    Notes
    -----
    The restriction is that if facet is a list, elem must be a list of the same length.
    This because then we retrieve the data for facet[i] from elem[i].

    If facet is an integer, then that means that we will retieve the same facet from all.
    In this case:
    elem can be an interger, which returns the data for the elem and facet pair.
    elem can be a list of integers, which returns the data for the list of elem and the same facet.
    elem can be None, which returns the data for all elements and the same facet.

    Returns
    -------
    np.ndarray
        The data for the given element and facet pair
    """

    # Get the slice tuple for each element in the list
    if isinstance(elem, list):
        elem_index = [e for e in elem]
    elif isinstance(elem, int):
        elem_index = elem
    else:
        elem_index = slice(None)

    if isinstance(facet, list):
        lz_index = [facet_to_slice_map[f][0] for f in facet]
        ly_index = [facet_to_slice_map[f][1] for f in facet]
        lx_index = [facet_to_slice_map[f][2] for f in facet]
    elif isinstance(facet, int):
        lz_index = facet_to_slice_map[facet][0]
        ly_index = facet_to_slice_map[facet][1]
        lx_index = facet_to_slice_map[facet][2]
    else:
        raise TypeError("Facet must be an integer or a list of integers")

    # Retrieve the data if inputs are list of corresponding element and facet pairs
    if isinstance(elem, list) and isinstance(facet, list):
        # Create slices
        slices = [
            (elem_index[i], lz_index[i], ly_index[i], lx_index[i])
            for i in range(len(elem_index))
        ]
        # Retrieve the data
        facet_data = np.array([field[s] for s in slices])

    # Retrieve the same facet from one element, elements in a list, or all elements
    elif (
        isinstance(elem, list) or isinstance(elem, int) or isinstance(elem, type(None))
    ) and isinstance(facet, int):
        slices = (elem_index, lz_index, ly_index, lx_index)
        facet_data = field[slices]

    else:
        raise TypeError(
            "Invalid input: If facet is a list, elem must be a list of the same length"
        )

    return facet_data
