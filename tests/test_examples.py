# Initialize MPI
import os
import subprocess
import pytest
import glob

@pytest.fixture(scope="session")
def sem_data_path():
    
    sem_data_path = "examples/data/sem_data"

    if not os.path.exists(sem_data_path):
        print("Sem data not found, cloning repository...")
        os.system(f"git clone https://github.com/adperezm/sem_data.git {sem_data_path}")
    else:
        print("Sem data found.")
    
    return sem_data_path

def test_data_types():


    examples_path = "examples/1-datatypes_and_io/"
    notebook_files = ["1-datatypes_io.ipynb", 
                      "2-sem_subdomains.ipynb"]

    passed = []
    for notebook in notebook_files:
        notebook_path = os.path.join(examples_path, notebook)

        print(f"Executing notebook: {notebook_path}")
        _path=os.path.abspath(notebook_path)
        _name=os.path.basename(notebook_path)
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

def test_calculus():


    examples_path = "examples/3-calculus_in_sem_mesh/"
    notebook_files = ["1-differentiation.ipynb", 
                      "1.5-differentiation_torch.ipynb", 
                      "2-integration.ipynb"]

    passed = []
    for notebook in notebook_files:
        notebook_path = os.path.join(examples_path, notebook)

        print(f"Executing notebook: {notebook_path}")
        _path=os.path.abspath(notebook_path)
        _name=os.path.basename(notebook_path)
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

def test_interpolation(sem_data_path):


    examples_path = "examples/4-interpolation/"
    notebook_files = ["1-interpolation_to_query_points.ipynb", 
                      "2-element_interpolation.ipynb", 
                      "3-interpolation_from_2d_sem_mesh.ipynb", 
                      "5-structured_mesh.ipynb", 
                      "6-interpolating_file_sequences.ipynb", 
                      "8-mirroring_fields.ipynb"]

    passed = []
    for notebook in notebook_files:
        notebook_path = os.path.join(examples_path, notebook)

        print(f"Executing notebook: {notebook_path}")
        _path=os.path.abspath(notebook_path)
        _name=os.path.basename(notebook_path)
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

def test_reduced_order_modelling():


    examples_path = "examples/5-reduced_order_modelling/"
    notebook_files = ["1-POD_from_pointclouds.ipynb", 
                      "2-POD_fft_from_pointclouds.ipynb"]

    passed = []
    for notebook in notebook_files:
        notebook_path = os.path.join(examples_path, notebook)

        print(f"Executing notebook: {notebook_path}")
        _path=os.path.abspath(notebook_path)
        _name=os.path.basename(notebook_path)
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