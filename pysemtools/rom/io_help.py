""" Module used to aid in IO operations. Contains classes
that help keep the data loaded or streamed during POD"""

import logging
import numpy as np
from .math_ops import MathOps as math_ops_c
from ..monitoring.logger import Logger as logger_c

NoneType = type(None)


class IoHelp:
    """Class used to help with IO buffering

    This class contains buffers to be used in the carrying out, for example, POD
    
    Parameters
    ----------
    comm : MPI communicator
        MPI communicator object.
    number_of_fields : int
        Number of fields to be stored in the buffer.
    batch_size : int
        Size of the buffer.
    field_size : int
        Size of each field.
    field_data_type : data-type, optional
        Data type of the fields. The default is np.double.
    mass_matrix_data_type : data-type, optional
        Data type of the mass matrix. The default is np.double.
    module_name : str, optional
        Name of the module for logging purposes. The default is "io_helper".
    """

    def __init__(
        self,
        comm,
        number_of_fields=1,
        batch_size=1,
        field_size=1,
        field_data_type=np.double,
        mass_matrix_data_type=np.double,
        module_name="io_helper",
    ):

        self.number_of_fields = number_of_fields
        self.batch_size = batch_size
        self.field_size = field_size
        self.nf = int(field_size * number_of_fields)

        # Allocate the buffers
        self.xi = np.zeros(
            (int(field_size * number_of_fields), 1), dtype=field_data_type
        )
        self.buff = np.zeros(
            (int(field_size * number_of_fields), self.batch_size), dtype=field_data_type
        )

        # Allocate the mass matrix
        self.bm1 = np.zeros(
            (int(field_size * number_of_fields), 1), dtype=mass_matrix_data_type
        )

        # Allocate the square root of the mass matrix
        self.bm1sqrt = np.zeros(
            (int(field_size * number_of_fields), 1), dtype=mass_matrix_data_type
        )

        # Instance object
        self.log = logger_c(comm=comm, module_name=module_name)
        self.math = math_ops_c()

        # Set the control indices
        self.buffer_index = 0
        self.buffer_max_index = batch_size - 1
        self.update_from_buffer = True

        self.log.write("info", "io_helper object initialized")

    def copy_fieldlist_to_xi(self, field_list=None):
        """Copy a field list into the buffer xi position 0

        This is used to make multi dimensional data into one big
        column vector that works as a snapshot for the POD
        
        Parameters
        ----------
        field_list : list of np.ndarray
            List of fields to be copied into xi.
        
        Returns
        -------
        None
        
        """
        if field_list is None:
            field_list = []

        field_size = field_list[0].size

        for i in range(0, len(field_list)):
            self.xi[i * field_size : (i + 1) * field_size, 0] = field_list[i].reshape(
                (field_size)
            )[:]

    def split_narray_to_1dfields(self, array):
        """Split a snapshot into a set of fields.

        This is somewhat an inverse of copy_fieldlist_to_xi, where from a big column
        vector we split it into a list of fields.
        
        Parameters
        ----------
        array : np.ndarray
            Array to be split into fields.

        Returns
        -------
        field_list1d : list of np.ndarray
            List of fields obtained from splitting the array.
        
        """
        field_size = self.field_size
        number_of_fields = array.shape[0] // field_size
        if number_of_fields != self.number_of_fields:
            self.log.write("debug", f"Number of fields passed: {number_of_fields} != number of fields expected: {self.number_of_fields} from inputs. Dividing the array into {number_of_fields} fields of size {field_size}. Be mindful that this may not be what you want.")
        field_list1d = []

        for i in range(0, number_of_fields):
            temp_field = array[i * field_size : (i + 1) * field_size]
            field_list1d.append(np.copy(temp_field))

        return field_list1d

    def load_buffer(self, scale_snapshot=True):
        """Function to load snapshot into the allocated buffer.

        This transfer the data from xi into the buffer at the current buffer index.
        
        If the buffer is full, it sets the flag to update from buffer to True.

        This is the buffer that the POD class uses to compute.
        
        Parameters
        ----------
        scale_snapshot : bool, optional
            If True, the snapshot is scaled with the mass matrix before being
            loaded into the buffer. The default is True.
        
        Returns
        -------
        None

        """
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
