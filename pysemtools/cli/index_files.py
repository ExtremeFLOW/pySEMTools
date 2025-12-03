#!/usr/bin/env python3
""" Create an index of files in a folder and save it as a JSON file. """
import argparse
from mpi4py import MPI
from ..postprocessing.file_indexing import index_files_from_folder

def main():
    '''
    Create a json file that contains and index of files in a folder

    This script creates an index of files in a specified folder and saves it as a JSON file. The index can include file contents and time intervals based on user preferences.
    
    Parameters
    ----------
    - -folder_path: Path to the folder containing files to index.
    - -output_folder: Path to the output folder.
    - -file_type: (Optional) List of file types to index.
    - -include_file_contents: (Optional) Boolean flag to include file contents in the index
    - -include_time_interval: (Optional) Boolean flag to include time interval in the index.
    - -run_start_time: (Optional) Float indicating the start time of the run.
    - -stat_start_time: (Optional) Float indicating the start time of the statistics.

    Examples
    --------
    To use this script, run the following command:

    >>> pysemtools_index_files --folder_path <path_to_folder> --output_folder <output_folder> [--file_type <file_type1,file_type2,...>] [--include_file_contents] [--include_time_interval] [--run_start_time <float>] [--stat_start_time <float>]

    Replacing the placeholders with actual values. Observe that you have to remove the angle brackets.
    '''
    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Index files from a folder.")
    
    # Define command-line arguments
    parser.add_argument("--folder_path", type=str, default="./", help="Path to the folder containing files to index.")
    parser.add_argument("--output_folder", type=str, default="./", help="Path to the output folder.")
    parser.add_argument("--file_type", type=str, nargs='+', default="", help="List of file types to index.")
    parser.add_argument("--include_file_contents", action="store_true", default=True, help="Include file contents in the index.")
    parser.add_argument("--include_time_interval", action="store_true", default=False, help="Include time interval in the index.")
    parser.add_argument("--run_start_time", type=float, default=0.0, help="Start time of the run.")
    parser.add_argument("--stat_start_time", type=float, default=0.0, help="Start time of the statistics.")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Call the function with parsed arguments
    index_files_from_folder(
        comm,
        folder_path=args.folder_path,
        output_folder=args.output_folder,
        file_type=args.file_type,
        include_file_contents=args.include_file_contents,
        include_time_interval=args.include_time_interval,
        run_start_time= args.run_start_time,
        stat_start_time= args.stat_start_time
    )

if __name__ == "__main__":
    main()
