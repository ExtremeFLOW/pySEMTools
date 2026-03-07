# 3rd party

In this folder we provide cmake instructions to build the dependencies. This is not really guaranteed to work in all cases, but we thought to add it to reduce the barrier of entry.

Please note that we have provided some instructions in the wiki, which use bash scripts. In this case we decided to use cmake to try to automate the process. This is by no means necesary. You can simply install everything yourself following the library developers instructions.

To build the dependencies use the following command:

```bash
cmake -S . -B build-superbuild \
  -DCMAKE_C_COMPILER=$(which mpicc) \
  -DCMAKE_CXX_COMPILER=$(which mpicxx) \
  -DINSTALL_ADIOS2=On \
  -DADIOS2_VERSION=2.10.2
```

Here we have set defaults for things that work. But you can overwrite things if you see fit.