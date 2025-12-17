# pySEMTools
A package for post-processing data obtained using a spectral-element method (SEM), on hexahedral high-order elements.

The most prominent features of the packages are the following:
* **Parallel IO**: A set of routines to perform distributed IO on Nek5000/Neko field files and directly keep the data in memory on NumPy arrays or PyMech data objects.
* **Parallel data interfaces**: A set of objects that aim to facilitate the transfer of messages among processors. Done to ease the use of MPI functions for more inexperienced users.
* **Calculus**:  Objects to calculate the derivation and integration matrices based on the geometry, which allows to perform calculus operations on the spectral element mesh.
* **Mesh connectivity and partitioning**: Objects to determine the connectivity based on the geometry and mesh repartitioning tools for tasks such as global summation, among others.
* **Interpolation**: Routines to perform high-order interpolation from an SEM mesh into any arbitrary query point. A crucial functionality when performing post-processing.
* **Reduced-order modeling**: Objects to perform parallel and streaming proper orthogonal decomposition (POD).
* **Data compression/streaming**: Through the use of ADIOS2 [@adios2], a set of interfaces is available to perform data compression or to connect Python scripts to running simulations to perform in-situ data processing. 
* **Visualization**: Given that the data is available in Python, visualizations can be performed from readily available packages. 


**Documentation** is available [here](https://extremeflow.github.io/pySEMTools/).

If you wish to **contribute** to PySEMTools, need assistance or to report a bug, please check `CONTRIBUTING.md` for the community guidelines on the best way to do it.

In case you find the tools useful, please cite as:
* Perez, A., Toosi, S., Olsen, T.F., Markidis, S., Schlatter, P., 2025. Pysemtools: A library for post-processing hexahedral spectral element data. [https://doi.org/10.48550/arXiv.2504.12301](https://doi.org/10.48550/arXiv.2504.12301)

The work was partially funded by the “Adaptive multi-tier intelligent data manager for Exascale (ADMIRE)” project,
which is funded by the European Union’s Horizon 2020 JTI-EuroHPC research and innovation program under grant
Agreement number: 956748. 

# Installation

To install in your pip environment (requires Python 3.10-3.12), clone this repository and execute:
```
pip install  --editable .
```

The `--editable` flag is optional, and will allow changes in the code of the package to be used
directly without reinstalling.

## Dependencies

### Mandatory

You can install dependencies as follow:

```
pip install numpy
pip install scipy
pip install pymech
pip install tdqm
```
#### mpi4py
`mpi4py` is needed even when running in serial, as the library is built with communication in mind. It can typically be installed with: 
```
pip install mpi4py
```

In some instances, such as in supercomputers, it is typically necesary that the mpi of the system is used. If `mpi4py` is not available as a module, we have found (so far) that installing it as follows works:
```
export MPICC=$(which CC)
pip install mpi4py --no-cache-dir
```
where CC should be replaced by the correct C wrappers of the system (In a workstation you would probably need mpicc or so). It is always a good idea to contact support or check the specific documentation if things do not work.

### Optional

#### ADIOS2

Some functionalities such as data streaming require the use of adios2. You can check how the installation is performed [here](https://adios2.readthedocs.io/en/latest/setting_up/setting_up.html)

#### PyTorch

Some classed are compatible with the pytorch module in case you have GPUs and want to use them in the process. We note that we only use pytorch optionally. There are versions that work exclusively with numpy on CPUs so pytorch can be avoided.

To install pytorch, you can check [here](https://pytorch.org/get-started/locally/). A simple installation for CUDA v12.1 on linux would look like this (following the instructions from the link):
```
pip3 install torch torchvision torchaudio
```
The process of installing pytorch in supercomputers is more intricate. In this case it is best to use the documentation of the specific cluster or contact support.


# Use

To get an idea on how the codes are used, feel free to check the examples we have provided. Please note that most of the routines included here work in prallalel. In fact, python scripts are encouraged rather than notebooks to take advantage of this capability.

# Tests

You can use the provided tests to check if your installation is complete (Not all functionallities are currently tested but more to come).

The tests rely on `pytest`. To install it in your pip enviroment simply execute `pip install pytest`. To run the tests, execute the `pytest tests/` command from the root directory of the repository.
