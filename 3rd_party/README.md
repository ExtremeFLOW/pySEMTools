# 3rd party

In this folder we provide cmake instructions to build the dependencies. This is not really guaranteed to work in all cases, but we thought to add it to reduce the barrier of entry.

Please note that we have provided some instructions in the wiki, which use bash scripts. In this case we decided to use cmake to try to automate the process. This is by no means necessary. You can simply install everything yourself following the specific library developers instructions.

**Important:** 
- ADIOS2 and Paraview take some time and are very situational. 
- Parallel HDF5 and H5PY are very useful and recommended.
- MPI4PY here is only really needed if you did not have it before. Note that this installation will make use of the systems' MPI, which is preferred in supercomputers

To configure the dependencies use the following command (Select `OFF` or remove from the command if there is a library you do not want):

```bash
cmake -S . -B build-superbuild \
  -DCMAKE_C_COMPILER=$(which mpicc) \
  -DCMAKE_CXX_COMPILER=$(which mpicxx) \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DINSTALL_MPI4PY=ON \
  -DMPI4PY_VERSION=4.1.1 \
  -DINSTALL_ADIOS2=ON \
  -DADIOS2_VERSION=2.10.1 \
  -DINSTALL_HDF5=ON \
  -DHDF5_VERSION=1.14.6 \
  -DINSTALL_H5PY=ON \
  -DINSTALL_CATALYST=ON \
  -DCATALYST_VERSION=2.0.0 \
  -DINSTALL_PARAVIEW=ON \
  -DPARAVIEW_VERSION=6.0.1 \
  -DPARAVIEW_VARIANT=SERVER
```

> On supercomputers you will very likely need to change your mpi wrappers

Here we have set defaults for things that work. But you can overwrite things if you see fit.

After configuring, build with:
```bash
cmake --build ./build-superbuild -j10
```

The libraries will be installed in this folder. You can choose different options from cmake.

> Setting the number of jobs in `-jxx` too high has caused paraview installations to run out of memory. 
> keep this in mind if you encounter compilers breaking down at the vtk build stage.

## Instructions known to work.
We will try to periodically update [here](https://github.com/ExtremeFLOW/pySEMTools/discussions) instructions that work when installing in some platforms
