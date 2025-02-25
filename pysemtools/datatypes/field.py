""" Contians class that contains information associated to fields"""

import numpy as np
from ..monitoring.logger import Logger
from ..io.ppymech.neksuite import pynekread_field

NoneType = type(None)


class Field:
    """
    Class that contains fields.

    This is the main class used to contain data that can be used for post processing.
    The data does not generarly need to be present in this class, as it is typically enough to have
    the data as ndarrays of shape (nelv, lz, ly, lx) for each field. However this class
    provides a easy interface to collect data tha is somehow associated.

    It also allows to easily write data to disk. As all the data in this class will be stored in the same file.

    Parameters
    ----------
    comm : Comm
        MPI comminicator object.
    data : HexaData, optional
        HexaData object that contains the coordinates of the domain.

    Attributes
    ----------
    fields : dict
        Dictionary that contains the fields. The keys are the field names and the values are lists of ndarrays.
        The keys for these dictionaries are the same as for Hexadata objects, i.e. vel, pres, temp, scal.
    vel_fields : int
        Number of velocity fields.
    pres_fields : int
        Number of pressure fields.
    temp_fields : int
        Number of temperature fields.
    scal_fields : int
        Number of scalar fields.
    t : float
        Time of the data.

    Returns
    -------

    Examples
    --------
    If a hexadata object: data is read from disk, the field object can be created directly from it.

    >>> from pysemtools.datatypes.field import Field
    >>> fld = Mesh(comm, data = data)

    If one wishes to use the data in the fields. It is possible to reference it with a ndarray of shape (nelv, lz, ly, lx)
    as follows:

    >>> u = fld.fields["vel"][0]
    >>> v = fld.fields["vel"][1]
    >>> w = fld.fields["vel"][2]
    >>> vel_magnitude = np.sqrt(u**2 + v**2 + w**2)

    A field object can be created empty and then fields can be added to it. Useful to write data to disk.
    if a ndarray u is created with shape (nelv, lz, ly, lx) it can be added to the field object as follows:

    >>> from pysemtools.datatypes.field import Field
    >>> fld = Field(comm)
    >>> fld.fields["vel"].append(u)
    >>> fld.update_vars()

    This fld object can then be used to write fld files from field u created in the code.
    """

    def __init__(self, comm, data=None):

        self.log = Logger(comm=comm, module_name="Field")

        self.fields = {}
        self.fields["vel"] = []
        self.fields["pres"] = []
        self.fields["temp"] = []
        self.fields["scal"] = []
        self.t = 0.0
        self.vel_fields = 0
        self.pres_fields = 0
        self.temp_fields = 0
        self.scal_fields = 0

        if not isinstance(data, NoneType):

            self.log.tic()
            self.log.write("info", "Initializing Field object from HexaData")

            vars_ = data.var
            self.vel_fields = vars_[1]
            self.pres_fields = vars_[2]
            self.temp_fields = vars_[3]
            self.scal_fields = vars_[4]

            # Read the full fields
            for qoi in range(0, self.vel_fields):
                prefix = "vel"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.pres_fields):
                prefix = "pres"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.temp_fields):
                prefix = "temp"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            for qoi in range(0, self.scal_fields):
                prefix = "scal"
                self.fields[prefix].append(get_field_from_hexadata(data, prefix, qoi))

            self.t = data.time

            self.log.write("info", "Field object initialized")
            self.log.toc()
        else:
            self.log.write("info", "Initializing empty Field object")

    def update_vars(self):
        """
        Update number of fields.

        Update the number of fields in the class in the event that
        it has been modified. This is needed for writing data properly if more arrays are added to the class.

        Examples
        --------
        A field object can be created empty and then fields can be added to it. Useful to write data to disk.
        if a ndarray u is created with shape (nelv, lz, ly, lx) it can be added to the field object as follows:

        >>> from pysemtools.datatypes.field import Field
        >>> fld = Field(comm)
        >>> fld.fields["vel"].append(u)
        >>> fld.update_vars()

        This fld object can then be used to write fld files from field u created in the code.
        """
        self.vel_fields = len(self.fields["vel"])
        self.pres_fields = len(self.fields["pres"])
        self.temp_fields = len(self.fields["temp"])
        self.scal_fields = len(self.fields["scal"])

        self.log.write("debug", "Field variables updated")
        self.log.write(
            "debug",
            f"Velocity fields: {self.vel_fields}, Pressure fields: {self.pres_fields}, Temperature fields: {self.temp_fields}, Scalar fields: {self.scal_fields}",
        )

    def clear(self):
        self.log.write("debug", "Clearing field variables")
        self.fields["vel"] = []
        self.fields["pres"] = []
        self.fields["temp"] = []
        self.fields["scal"] = []
        self.t = 0.0
        self.vel_fields = 0
        self.pres_fields = 0
        self.temp_fields = 0
        self.scal_fields = 0


def get_field_from_hexadata(data, prefix, qoi):
    """
    Extract a field from the hexadata object and return it as a numpy array

    This way the hexadata can be more readily used for computations.

    Parameters
    ----------
    data : hexadata
        The hexadata object that contains the field data.
    prefix : str
        The prefix of the field to extract. Options are "vel", "pres", "temp", "scal"
    qoi : int
        The quantity of interest to extract, e.g. if prefix is "vel" and qoi is 0, the velocity field in x durection will be extracted.

    Returns
    -------
    ndarray
        The field data extracted from the hexadata object
    """
    nelv = data.nel
    lx = data.lr1[0]
    ly = data.lr1[1]
    lz = data.lr1[2]

    if prefix == "vel":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].vel.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].vel[qoi, :, :, :]

    if prefix == "pres":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].pres.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].pres[0, :, :, :]

    if prefix == "temp":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].temp.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].temp[0, :, :, :]

    if prefix == "scal":
        field = np.zeros((nelv, lz, ly, lx), dtype=data.elem[0].scal.dtype)
        for e in range(0, nelv):
            field[e, :, :, :] = data.elem[e].scal[qoi, :, :, :]

    return field

class NoOverwriteDict(dict):
    def __setitem__(self, key, value):
        if key in self and isinstance(value, np.ndarray):
            raise KeyError(f"Key '{key}' already exists. Cannot overwrite existing array without the add field method")
        super().__setitem__(key, value)

class FieldRegistry(Field):
    """
    Class that contains fields.

    This class extends the main field class, as it contains a registry that allows to easily reference the fields.

    Parameters
    ----------
    comm : Comm
        MPI comminicator object.
    data : HexaData, optional
        HexaData object that contains the coordinates of the domain.
    """

    def __init__(self, comm, data=None):

        super().__init__(comm, data=data)

        #self.registry = {}
        self.registry = NoOverwriteDict()
        self.registry_pos = {}
        self.scal_fields_names = []

        self.update_vars()

    def update_vars(self):
        """
        Update the registry with the fields that are present in the fields dictionary.
        """

        super().update_vars()

        if self.vel_fields > 0:
            self.registry["u"] = self.fields["vel"][0]
            self.registry_pos["u"] = "vel_0"
            self.registry["v"] = self.fields["vel"][1]
            self.registry_pos["v"] = "vel_1"
            if self.vel_fields > 2:
                self.registry["w"] = self.fields["vel"][2]
                self.registry_pos["w"] = "vel_2"

        if self.pres_fields > 0:
            self.registry["p"] = self.fields["pres"][0]
            self.registry_pos["p"] = "pres_0"

        if self.temp_fields > 0:
            self.registry["t"] = self.fields["temp"][0]
            self.registry_pos["t"] = "temp_0"

        if self.scal_fields > 0:
            for i in range(0, self.scal_fields):
                self.registry[f"s{i}"] = self.fields["scal"][i]
                self.registry_pos[f"s{i}"] = f"scal_{i}"

    def clear(self):
        """
        Clear the registry and the fields.
        """

        super().clear()

        #self.registry = {}
        self.registry = NoOverwriteDict()
        registry_pos = {}
        self.scal_fields_names = []

    def rename_registry_key(self, old_key="", new_key=""):
        """
        Rename a key in the registry.

        Parameters
        ----------
        old_key : str
            Old key to be renamed.

        new_key : str
            New key to be used.

        Notes
        -----

        If you update the registry, some keys might be overwritten or multiple keys might reference the same data.

        """

        if old_key in self.registry:
            self.registry[new_key] = self.registry.pop(old_key)

    def add_field(
        self,
        comm,
        field_name="",
        field=None,
        file_type=None,
        file_name=None,
        file_key=None,
        dtype=np.double,
    ):
        """
        Add fields to the registry. They will be stored in the fields dictionary to easily write them.


        Parameters
        ----------

        comm : Comm
            MPI comminicator object.

        field_name : str
            Name of the field to be added. where the field is added thepends on the name

        field : ndarray
            Field to be added to the registry. If this is provided, it is assumed to be a ndarray.
            It will then be added to the registry.

        file_type : str
            Type of the file to be added. If this is provided, it is assumed to be a file.
            Currently, only "fld" supported.

        file_name : str
            File name of the field to be added. If this is provided, it is assumed to be a file.
            It will then be added to the registry.

        file_key : str
            File key. This will be search in the file.
            For nek file, the key have the following format:
            "vel_0", "vel_1", "pres", "temp", "scal_0", "scal_1", etc.
            Only for "vel" we read the 2/3 components at the same time

        dtype : np.dtype
            Data type of the field. Default is np.double.
        """

        if (
            not isinstance(file_name, type(None))
            or not isinstance(file_key, type(None))
            or not isinstance(file_type, type(None))
        ):
            # Assign a field from a file

            if file_type != "fld":
                self.log.write("error", f"File type {file_type} not supported")
                return

            if file_type == "fld":

                key_prefix = file_key.split("_")[0]
                try:
                    key_suffix = int(file_key.split("_")[1])
                except IndexError:
                    self.log.write(
                        "warning",
                        f"File key {file_key} has no suffix (key: prefix_sufix). Assuming suffix 0",
                    )
                    key_suffix = 0

                self.log.write("debug", f"Reading field {file_name} from file")

                field_list = pynekread_field(
                    file_name, comm, data_dtype=dtype, key=file_key
                )

                if key_prefix == "vel" or key_prefix == "pos":
                    field = field_list[key_suffix]
                else:
                    field = field_list[0]

        if not isinstance(field, type(None)):
            # Assign a field from a ndarray

            if field_name == "u" or field_name == "v" or field_name == "w":
                prefix = "vel"
            elif field_name == "p":
                prefix = "pres"
            elif field_name == "t":
                prefix = "temp"
            else:
                prefix = "scal"
                self.scal_fields_names.append(field_name)

            if field.dtype != dtype:
                self.log.write(
                    "warning",
                    f"Field {field_name} has dtype {field.dtype} but expected {dtype}",
                )
                self.log.write(
                    "warning", f"Field {field_name} will be casted to {dtype}"
                )
                # cast it
                field = field.astype(dtype)


            if field_name in self.registry:
                self.log.write(
                    "warning",
                    f"Field {field_name} already in registry. Overwriting",
                )

                # Find the position of the field in the list
                pos = self.registry_pos[field_name]
                prefix = pos.split("_")[0]
                pos = int(pos.split("_")[1])

                # Overwrite the field
                self.registry[field_name][:,:,:,:] = field 
            
            else:

                # Append it to the field list
                self.fields[prefix].append(field)
                # Register it in the dictionary
                self.registry[field_name] = field
                self.registry_pos[field_name] = f"{prefix}_{len(self.fields[prefix])-1}"

            super().update_vars()
