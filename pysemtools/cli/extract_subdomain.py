""" Extract a subdomain from an existing file. """
import pysemtools
from mpi4py import MPI
import numpy as np
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.utils import write_fld_subdomain_from_list
import argparse


def parse_comma_separated(value):
    '''
    Internal function for parsing text inputs
    '''
    # Split the string by commas, strip spaces, and return a list
    return [item.strip() for item in value.split(",")]


def parse_bounds(value):
    '''
    Internal function for parsing bounds input
    '''
    # Split the string by commas, strip spaces from each part, and convert to float
    return [float(x.strip()) for x in value.split(",")]


def main():
    '''
    Extract a subdomain from an existing file.

    This script extracts a box-shaped subdomain from a given field file and saves it to a new file.

    Parameters
    ----------
    - -input_file: Path to the input field file.
    - -output_file: Path to the output field file where the extracted subdomain will be saved
    - -bounds: Comma-separated list of 6 floats defining the subdomain bounds: xmin,xmax,ymin,ymax,zmin,zmax
    - -fields: (Optional) Comma-separated list of field names to extract. If not provided, all fields will be extracted.

    Examples
    --------
    To use this script, run it with MPI using the following command:
    
    >>> mpirun -n <num_processes> python pysemtools_extract_subdomain --input_file <input.fld> --output_file <output.fld> --bounds=<xmin,xmax,ymin,ymax,zmin,zmax> [--fields=<field1,field2,...>]
    
    Replacing the placeholders with actual values. Observe that you have to remove the angle brackets.
    '''
    comm = MPI.COMM_WORLD
    parser = argparse.ArgumentParser(
        description="Extract a box-shaped subdomain from a field file",
        epilog="""Example usage:

    mpirun -n 4 python pysemtools_extract_subdomain --input_file input.fld --output_file output.fld --bounds=1.0,2.0,3.0,4.0,5.0,6.0 --fields=field1,field2,field3
    """,
    )

    # Define command-line arguments
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the field file"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to the output field file with the extracted subdomain",
    )
    parser.add_argument(
        "--bounds",
        type=parse_bounds,
        required=True,
        help="Comma-separated list of 6 floats",
    )
    parser.add_argument(
        "--fields",
        type=parse_comma_separated,
        default=None,
        required=False,
        help="Comma-separated list of field names to save in the extraction (optional, all fields by default)",
    )

    # Parse the arguments
    args = parser.parse_args()
    bounds = args.bounds

    # Mesh and field registry to populate on read
    mesh = Mesh(comm, create_connectivity=True)
    fld = FieldRegistry(comm)

    # Read
    pynekread(args.input_file, comm, data_dtype=np.single, msh=mesh, fld=fld)

    fields = args.fields
    if not fields:
        fields = list(fld.registry.keys())

    # Write the data in a subdomain
    write_fld_subdomain_from_list(
        args.output_file,
        comm,
        mesh,
        field_list=[fld.registry[i] for i in fields],
        subdomain=[
            [bounds[0], bounds[1]],
            [bounds[2], bounds[3]],
            [bounds[4], bounds[5]],
        ],
    )

if __name__ == "__main__":
    main()
