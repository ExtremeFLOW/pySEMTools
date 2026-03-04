
# Data transfer

These tools are designed to be run in parallel. While the use of MPI is aimed to be mostly internal, we provide a class called Router that aims to simplify data movement in case the user needs it.

For this case we do not provide notebooks, as the examples are meant to be run at least with 2 MPI ranks.

We heavily comment the python notebook and provide here some take aways.

1. In pySEMTools we have implemented the Router class
    - The router class contains wrappers for mpi all_to_all, Scatterv, GatherV, AllgatherV and a all_to_all implementation using isend/irecv
    - In all cases the data is returned as an array of 1 dimension, meaning that it is always necesary to reshape data upon recieveing it.
    - gathering and scattering operations accept data directly
    - the all to all operations require a list with the destination ranks and a list with the data to be sent to each destination
    - if the all to all operations do not get a list as an input, then the same data will be sent to all ranks.

More information can be found in the docs for the Router class.

If you wish to see how this class can be used, you can check the source code for the interpolator class and for the mesh partitioner class, where data movement between ranks is needed.

You can of course use mpi4py directly instead of these routines, but we beleive that we have simplified it for the general user.