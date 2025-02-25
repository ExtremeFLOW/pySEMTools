""" Functions to plot data """

import numpy as np
import pyvista as pv
from typing import Union

def isosurfaces(mesh: dict[str, np.ndarray], fields:dict[str, np.ndarray], isosurfaces = dict[str, np.ndarray], shape:tuple[int, int] = (1, 1), window_size: list[int] = [1920, 1080], rows_frist: bool = True, colormap: str = "coolwarm", opacity: float= 1.0) -> list[pv.Plotter]:
    """ Create isosurfaces of the fields in the mesh
    
    Parameters
    ----------
    mesh : dict[str, np.ndarray]
        The mesh data. Should be a dictionary with "x", "y" and "z" keys
    fields : dict[str, np.ndarray]
        The fields data. Should be a dictionary with the field names as keys and the field data as values
    isosurfaces : dict[str, np.ndarray]
        The isosurfaces to be plotted. Should be a dictionary with the field names as keys and the isosurface values as values
    shape : tuple[int, int], optional
        The shape of the plot, by default (1, 1)
    window_size : list[int], optional
        The size of the window, by default [1920, 1080]
    rows_frist : bool, optional
        If the plots should be done by rows first, by default True
    """


    # Create the grid
    mesh = pv.StructuredGrid(mesh["x"], mesh["y"], mesh["z"])
    # Assign the data
    for key in fields.keys():
        mesh.point_data[key] = fields[key].ravel(order='F')

    # Get the number of plots you will need.
    requested_plots = len(fields.keys())
    plots_per_plotter = shape[0] * shape[1]
    n_plotter = requested_plots // plots_per_plotter + (requested_plots % plots_per_plotter > 0)

    # Create the plotter
    pl = [pv.Plotter(shape=shape, window_size=window_size) for _ in range(n_plotter)]

    i_plotter = 0
    i_plot = 0
    i_plot_row = 0
    i_plot_col = 0

    for key in fields.keys():
        # Plot the data
        pl[i_plotter].add_axes()
        pl[i_plotter].subplot(i_plot_row, i_plot_col)
        isos = mesh.contour(scalars = key, isosurfaces = isosurfaces[key])
        pl[i_plotter].add_mesh(mesh.outline(), color="k")
        pl[i_plotter].add_mesh(isos, opacity=opacity, cmap=colormap)
        i_plot += 1

        if not rows_frist:
            i_plot_row += 1
            if i_plot_row == shape[0]:
                i_plot_row = 0
                i_plot_col += 1
        else:
            i_plot_col += 1
            if i_plot_col == shape[1]:
                i_plot_col = 0
                i_plot_row += 1

        # Update the indexes
        if i_plot == plots_per_plotter:
            i_plot = 0
            i_plot_row = 0
            i_plot_col = 0
            i_plotter += 1

    return pl