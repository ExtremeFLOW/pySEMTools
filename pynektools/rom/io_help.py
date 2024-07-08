from mpi4py import MPI  # equivalent to the use of MPI_init() in C
import numpy as np
from .math_ops import math_ops_c
from .logger import logger_c
import logging
import os

NoneType = type(None)


class io_help_c:

    def __init__(
        self,
        comm,
        number_of_fields=1,
        batch_size=1,
        field_size=1,
        data_type=np.double,
        module_name="io_helper",
    ):

        self.number_of_fields = number_of_fields
        self.batch_size = batch_size
        self.field_size = field_size
        self.nf = int(field_size * number_of_fields)

        # Allocate the buffers
        self.xi = np.zeros((int(field_size * number_of_fields), 1), dtype=data_type)
        self.buff = np.zeros(
            (int(field_size * number_of_fields), self.batch_size), dtype=data_type
        )

        # Allocate the mass matrix
        self.bm1 = np.zeros((int(field_size * number_of_fields), 1), dtype=data_type)

        # Allocate the square root of the mass matrix
        self.bm1sqrt = np.zeros(
            (int(field_size * number_of_fields), 1), dtype=data_type
        )

        # Instance object
        self.log = logger_c(level=logging.DEBUG, comm=comm, module_name=module_name)
        self.math = math_ops_c()

        # Set the control indices
        self.buffer_index = 0
        self.buffer_max_index = batch_size - 1
        self.update_from_buffer = True

        self.log.write("info", "io_helper object initialized")

        return

    def copy_fieldlist_to_xi(self, field_list=[]):

        field_size = field_list[0].size

        for i in range(0, len(field_list)):
            self.xi[i * field_size : (i + 1) * field_size, 0] = field_list[i].reshape(
                (field_size)
            )[:]

        return

    def split_narray_to_1dfields(self, array):

        field_size = self.field_size
        number_of_fields = self.number_of_fields
        field_list1d = []

        for i in range(0, number_of_fields):
            temp_field = array[i * field_size : (i + 1) * field_size]
            field_list1d.append(np.copy(temp_field))

        return field_list1d

    def load_buffer(self, scale_snapshot=True):

        if self.buffer_index > self.buffer_max_index:
            self.buffer_index = 0

        if scale_snapshot:
            # Scale the data with the mass matrix
            self.math.scale_data(self.xi, self.bm1sqrt, self.nf, 1, "mult")

        # Fill the buffer
        self.buff[:, self.buffer_index] = np.copy(self.xi[:, 0])

        if self.buffer_index == self.buffer_max_index:
            self.log.write(
                "info", "Loaded snapshot in buffer in pos: " + repr(self.buffer_index)
            )
            self.log.write("info", "Buffer one is full, proceed to update")
            self.buffer_index += 1
            self.update_from_buffer = True
        else:
            self.log.write(
                "info", "Loaded snapshot in buffer in pos: " + repr(self.buffer_index)
            )
            self.buffer_index += 1
            self.update_from_buffer = False

        return
