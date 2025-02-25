"""contains p refiner class"""

import numpy as np
from .point_interpolator.single_point_legendre_interpolator import (
    LegendreInterpolator as element_interpolator_c,
)
from ..datatypes.msh import Mesh
from ..datatypes.field import Field, FieldRegistry
from typing import Union


class PRefiner:
    """Class to perform p-refinement on a sem mesh"""

    def __init__(self, n_old=8, n_new=8, dtype=None):

        # Order of the element
        self.n = n_new
        self.dtype = dtype

        # Initialize the element interpolators
        self.ei_old = element_interpolator_c(n_old)
        self.ei_new = element_interpolator_c(n_new)

        # Define some dummy variables
        self.lx = None
        self.ly = None
        self.lz = None
        self.nelv = None

    def create_refined_mesh(self, comm, msh=None):
        """Obtained refined/coarsened mesh"""

        # See the points per element in the new mesh
        self.lx = self.n
        self.ly = self.n
        if msh.lz > 1:
            self.lz = self.n
        self.nelv = msh.nelv

        if isinstance(self.dtype, type(None)):
            dtype = msh.x.dtype
        else:
            dtype = self.dtype

        # Allocate the new coordinates
        x = np.zeros((msh.nelv, self.lz, self.ly, self.lx), dtype=dtype)
        y = np.zeros((msh.nelv, self.lz, self.ly, self.lx), dtype=dtype)
        z = np.zeros((msh.nelv, self.lz, self.ly, self.lx), dtype=dtype)

        # Loop over the elements and perform the interpolation
        x_gll = self.ei_new.x_gll
        y_gll = self.ei_new.x_gll
        w_gll = self.ei_new.x_gll

        for e in range(0, msh.nelv):
            x[e, :, :, :] = self.ei_old.interpolate_field_at_rst_vector(
                x_gll, y_gll, w_gll, msh.x[e, :, :, :]
            )
            y[e, :, :, :] = self.ei_old.interpolate_field_at_rst_vector(
                x_gll, y_gll, w_gll, msh.y[e, :, :, :]
            )
            z[e, :, :, :] = self.ei_old.interpolate_field_at_rst_vector(
                x_gll, y_gll, w_gll, msh.z[e, :, :, :]
            )

        # Create the msh object
        new_msh = Mesh(comm, x=x, y=y, z=z)

        return new_msh
    
    def create_refined_field(self, comm, fld: Union[Field, FieldRegistry]=None) -> FieldRegistry:
        
        
        if isinstance(fld, FieldRegistry):
            refined_field = FieldRegistry(comm)

            for key in fld.registry.keys():
                field_ = self.interpolate_from_field_list(
                        comm, field_list=[fld.registry[key]]
                    )
                refined_field.add_field(comm, field_name=key, field=field_[0].copy(), dtype=field_[0].dtype)

        elif isinstance(fld, Field):
            
            refined_field = FieldRegistry(comm)

            for key in fld.fields.keys():
                for i in range(len(fld.fields[key])):
                    field_ = self.interpolate_from_field_list(
                        comm, field_list=[fld.fields[key][i]]
                    )
                    refined_field.fields[key].append(field_[0].copy())
     
            refined_field.t = fld.t
            refined_field.update_vars()
        
        return refined_field


    def interpolate_from_field_list(self, comm, field_list=[]):
        """Interpolate any field that was in the old mesh onto the refined/coarsened one"""
        # check the number of fields to interpolate
        number_of_fields = len(field_list)
        
        if isinstance(self.dtype, type(None)):
            dtype = field_list[0].dtype
        else:
            dtype = self.dtype

        # Allocate the result of the interpolation
        interpolated_fields = []
        for _ in range(0, number_of_fields):
            interpolated_fields.append(
                np.zeros((self.nelv, self.lz, self.ly, self.lx), dtype=dtype)
            )

        # Get the RST coordinates of the new points
        x_gll = self.ei_new.x_gll
        y_gll = self.ei_new.x_gll
        w_gll = self.ei_new.x_gll

        ff = 0
        for field in field_list:

            for e in range(0, self.nelv):
                interpolated_fields[ff][e, :, :, :] = (
                    self.ei_old.interpolate_field_at_rst_vector(
                        x_gll, y_gll, w_gll, field[e, :, :, :]
                    )
                )

            ff += 1

        return interpolated_fields


class PMapper:
    """Class to map points from one point distribution to other."""

    def __init__(self, n=8, n_new = None, distribution=["GLL", "GLL", "GLL"]):

        self.n_old = n
        if type(n_new) is type(None):
            self.n_new = n
        else:
            self.n_new = n_new
        
        # Order of the element
        self.n = self.n_new

        # Initialize the element interpolators
        self.ei = element_interpolator_c(self.n_old)        
        self.ei_new = element_interpolator_c(self.n_new)

        # Define some dummy variables
        self.lx = None
        self.ly = None
        self.lz = None
        self.nelv = None

        self.distribution = distribution

        # Select the distribution per direction
        if distribution[0] == "GLL":
            self.r_dist = self.ei_new.x_gll
        elif distribution[0] == "EQ":
            self.r_dist = np.linspace(-1, 1, self.n_new)

        if distribution[1] == "GLL":
            self.s_dist = self.ei_new.x_gll
        elif distribution[1] == "EQ":
            self.s_dist = np.linspace(-1, 1, self.n_new)

        if distribution[2] == "GLL":
            self.t_dist = self.ei_new.x_gll
        elif distribution[2] == "EQ":
            self.t_dist = np.linspace(-1, 1, self.n_new)

    def create_mapped_mesh(self, comm, msh=None):
        """Obtained refined/coarsened mesh"""

        # See the points per element in the new mesh
        self.lx = self.n_new
        self.ly = self.n_new
        if msh.lz > 1:
            self.lz = self.n_new
        self.nelv = msh.nelv

        # Allocate the new coordinates
        x = np.zeros((msh.nelv, self.lz, self.ly, self.lx), dtype=msh.x.dtype)
        y = np.zeros((msh.nelv, self.lz, self.ly, self.lx), dtype=msh.x.dtype)
        z = np.zeros((msh.nelv, self.lz, self.ly, self.lx), dtype=msh.x.dtype)

        # Loop over the elements and perform the interpolation
        x_gll = self.r_dist
        y_gll = self.s_dist
        w_gll = self.t_dist

        for e in range(0, msh.nelv):
            x[e, :, :, :] = self.ei.interpolate_field_at_rst_vector(
                x_gll, y_gll, w_gll, msh.x[e, :, :, :]
            )
            y[e, :, :, :] = self.ei.interpolate_field_at_rst_vector(
                x_gll, y_gll, w_gll, msh.y[e, :, :, :]
            )
            z[e, :, :, :] = self.ei.interpolate_field_at_rst_vector(
                x_gll, y_gll, w_gll, msh.z[e, :, :, :]
            )

        # Create the msh object
        new_msh = Mesh(comm, x=x, y=y, z=z, create_connectivity=msh.create_connectivity_bool)

        return new_msh
    
    def create_mapped_field(self, comm, fld: Union[Field, FieldRegistry]=None) -> FieldRegistry:
        
        
        if isinstance(fld, FieldRegistry):
            mapped_field = FieldRegistry(comm)

            for key in fld.registry.keys():
                field_ = self.interpolate_from_field_list(
                        comm, field_list=[fld.registry[key]]
                    )
                mapped_field.add_field(comm, field_name=key, field=field_[0].copy(), dtype=field_[0].dtype)

        elif isinstance(fld, Field):
            
            mapped_field = FieldRegistry(comm)

            for key in fld.fields.keys():
                for i in range(len(fld.fields[key])):
                    field_ = self.interpolate_from_field_list(
                        comm, field_list=[fld.fields[key][i]]
                    )
                    mapped_field.fields[key].append(field_[0].copy())
     
            mapped_field.t = fld.t
            mapped_field.update_vars()
        
        return mapped_field

    def interpolate_from_field_list(self, comm, field_list=[]):
        """Interpolate any field that was in the old mesh onto the refined/coarsened one"""
        # check the number of fields to interpolate
        number_of_fields = len(field_list)

        # Allocate the result of the interpolation
        interpolated_fields = []
        for _ in range(0, number_of_fields):
            interpolated_fields.append(np.zeros((self.nelv, self.lz, self.ly, self.lx), dtype=field_list[0].dtype))

        # Get the RST coordinates of the new points
        x_gll = self.r_dist
        y_gll = self.s_dist
        w_gll = self.t_dist

        ff = 0
        for field in field_list:

            for e in range(0, self.nelv):
                interpolated_fields[ff][e, :, :, :] = (
                    self.ei.interpolate_field_at_rst_vector(
                        x_gll, y_gll, w_gll, field[e, :, :, :]
                    )
                )

            ff += 1

        return interpolated_fields
