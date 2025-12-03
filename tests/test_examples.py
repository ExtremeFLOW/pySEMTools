# Initialize MPI
import os
import subprocess
import pytest
import glob

@pytest.fixture(scope="session")
def sem_data_path():
    
    sem_data_path = "examples/data/sem_data"

    if not os.path.exists(sem_data_path):
        print("Sem data not found, clioning repository...")
        os.system(f"git clone https://github.com/adperezm/sem_data.git {sem_data_path}")
    else:
        print("Sem data found.")
    
    print("I AM RUNNING")
    assert False, "Fixture reached!"


def test_data_types():


    examples_path = "examples/1-datatypes_and_io/"
    notebook_files = ["1-datatypes_io.ipynb", "2-sem_subdomains.ipynb", "3-data_compression.ipynb"]

    passed = []
    for notebook in notebook_files:
        notebook_path = os.path.join(examples_path, notebook)

        print(f"Executing notebook: {notebook_path}")
        _path=os.path.abspath(notebook_path)
        _name=os.path.basename(notebook_path)
        print(f"Notebook name: {_name}") 
        command = f"jupyter nbconvert --to notebook --execute {notebook_path} --output ./executed_{_name}"

        process = subprocess.run(command, capture_output=True, shell=True)
        if process.returncode != 0:
            print("Error executing notebook:")
            print(process.stdout.decode())
            print(process.stderr.decode())
            _passed = False
        else:
            print("Notebook executed successfully.")
            _passed = True

        passed.append(_passed)

    assert all(passed)