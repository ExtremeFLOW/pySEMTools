import numpy as np


def gather_in_root(sendbuf, root, dtype, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Collect local array sizes using the high-level mpi4py gather
    sendcounts = np.array(comm.allgather(sendbuf.size))

    if rank == root:
        # print("sendcounts: {}, total: {}".format(sendcounts, sum(sendcounts)))
        recvbuf = np.empty(sum(sendcounts), dtype=dtype)
    else:
        recvbuf = None

    comm.Gatherv(sendbuf=sendbuf, recvbuf=(recvbuf, sendcounts), root=root)

    return recvbuf, sendcounts


def scatter_from_root(sendbuf, sendcounts, root, dtype, comm):

    rank = comm.Get_rank()
    size = comm.Get_size()

    recvbuf = np.ones(sendcounts[rank], dtype=dtype) * -100

    comm.Scatterv(sendbuf=(sendbuf, sendcounts), recvbuf=recvbuf, root=root)

    return recvbuf
