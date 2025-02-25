#!/usr/bin/env python3

import argparse
import glob
import sys

def main():
    
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
