import numpy as np
from .sem import element_interpolator_c
from ..datatypes.msh import msh_c


class p_refiner_c:
    def __init__(self, n_old=8, n_new=8):

        # Order of the element
        self.n = n_new

        # Initialize the element interpolators
        self.ei_old = element_interpolator_c(n_old)
        self.ei_new = element_interpolator_c(n_new)

        return

    def get_new_mesh(self, comm, msh=None):

        # See the points per element in the new mesh
        self.lx = self.n
        self.ly = self.n
        if msh.lz > 1:
            self.lz = self.n
        self.nelv = msh.nelv

        # Allocate the new coordinates
        x = np.zeros((msh.nelv, self.lz, self.ly, self.lx))
        y = np.zeros((msh.nelv, self.lz, self.ly, self.lx))
        z = np.zeros((msh.nelv, self.lz, self.ly, self.lx))

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
        new_msh = msh_c(comm, x=x, y=y, z=z)

        return new_msh

    def interpolate_from_field_list(self, comm, field_list=[]):

        # check the number of fields to interpolate
        number_of_fields = len(field_list)

        # Allocate the result of the interpolation
        interpolated_fields = []
        for i in range(0, number_of_fields):
            interpolated_fields.append(np.zeros((self.nelv, self.lz, self.ly, self.lx)))

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
