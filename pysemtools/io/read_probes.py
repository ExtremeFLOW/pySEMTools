"""Contains the ProbesReader class"""

import csv
import numpy as np

__all__ = ["ProbesReader"]


class ProbesReader:
    """
    A class to read probed fields from a Neko simulation.

    Attributes
    ----------
    points : np.ndarray of floats
        The probe locations.
    times : np.array of floats
        The times at each time step.
    field_names : list of strings
        The names of the fields.
    fields : dictionary of np.array
        The fields at each time step.
    """

    def __init__(self, file_name: str):
        """
        Read in the probes file and return the points, fields, times and field
        names.

        Parameters
        ----------
        file_name : str
            The name of the file to read in.
            Currently only supports .csv files.
        """

        if file_name.endswith(".csv"):
            points, fields, times, field_names = read_probes_csv(file_name)
        else:
            raise ValueError("Unsupported probe format.")

        # Save the attributes
        self.points = points
        self.times = times
        self.field_names = field_names
        self.fields = fields


def read_probes_csv(file_name: str) -> tuple:
    """
    Read in the probes file and return the points, fields, times and field
    names.

    Parameters
    ----------
    file_name : str
        The name of the file to read in.

    Returns
    -------
    points : np.array
        The points in the domain.
    fields : np.array
        The fields at each time step.
    times : np.array
        The times at each time step.
    field_names : list
        The names of the fields.
    """

    with open(file_name) as file:
        N_lines = sum(1 for _ in file)

    with open(file_name, "r") as file:
        reader = csv.reader(file)

        # Read in the header
        header = next(reader)
        N_p = int(header[0])
        N_s = int((N_lines - 1 - N_p) / (N_p)) * N_p
        field_names = header[2:]

        points = np.asarray([next(reader) for _ in range(N_p)], dtype=float)
        tmp = np.asarray([next(reader) for _ in range(N_s)], dtype=float)

        times = tmp[0::N_p, 0]
        fields = dict()
        for i, field_name in enumerate(field_names):
            fields[field_name] = np.reshape(tmp[:, i + 1], (N_p, -1), order="F")

    return (points, fields, times, field_names)
