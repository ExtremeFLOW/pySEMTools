#!/usr/bin/env python3
''' Create a nek5000 file template from existing files. Used to visualize nek5000 files in VisIt. '''

import argparse
import glob
import sys

def main():
    '''
    Create a nek5000 file template from existing files. Used to visualize nek5000 files in VisIt.

    This script generates a nek5000 file template based on existing files in the current directory. The generated file contains information about the file naming pattern, first timestep, and number of timesteps.

    Parameters
    ----------
    filename_pattern: Pattern to match the nek5000 files.

    Examples
    --------
    To use this script, run it with the following command:

    >>> pysemtools_visnek <filename_pattern>

    Replacing the placeholder with the actual filename pattern. Observe that you have to remove the angle brackets.
    '''

    if len(sys.argv) == 1:
        print("No arguments provided. Exiting...")
        sys.exit(1)
    
    if len(sys.argv) > 2:
        print("Too many arguments provided. Exiting...")
        sys.exit(1)

    files = sorted(glob.glob(f"*{sys.argv[1]}0*"))

    # Get the first index of the file
    first_index = int(files[0].split(".")[1][-5:]) 
    last_index = int(files[-1].split(".")[1][-5:]) 

    number_of_files = len(files)

    if number_of_files == 0:
        print("No files found. Exiting...")
        sys.exit(1)

    with open(f"{sys.argv[1]}.nek5000", "w") as f:
        f.write(f"filetemplate: {sys.argv[1]}%01d.f%05d\n")
        f.write(f"firsttimestep: {first_index}\n")
        f.write(f"numtimesteps: {number_of_files}\n")
        
if __name__ == "__main__":
    main()
