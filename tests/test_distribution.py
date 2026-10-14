# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD

# Import general modules
import numpy as np

# Import relevant modules
from pysemtools.comm.distribution import linear_distribution, linear_owner


class FakeComm:
    """Stand-in communicator with a chosen size and rank."""

    def __init__(self, size, rank=0):
        self.size, self.rank = size, rank

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank


def reference(m, size, rank):
    """Neko's linear_dist in the floating point form used across the package before."""
    l = np.floor(np.double(m) / np.double(size))
    r = np.mod(m, size)
    ip = np.floor((np.double(m) + np.double(size) - np.double(rank) - np.double(1)) / np.double(size))
    return int(ip), int(rank * l + min(rank, r))


def test_linear_distribution_matches_reference():

    for m in (0, 1, 7, 64, 1000, 12345):
        for size in (1, 2, 3, 8, 64, 100):
            offsets = []
            total = 0
            for rank in range(size):
                n, off = linear_distribution(m, FakeComm(size, rank))
                assert (n, off) == reference(m, size, rank)
                assert linear_distribution(m, FakeComm(size), rank=rank) == (n, off)
                offsets.append(off)
                total += n
            # the blocks tile 0..m-1 contiguously
            assert total == m
            assert offsets[0] == 0
            for rank in range(1, size):
                n_prev, off_prev = linear_distribution(m, FakeComm(size, rank - 1))
                assert offsets[rank] == off_prev + n_prev


def test_linear_owner_matches_distribution():

    for m in (1, 7, 64, 1000):
        for size in (1, 3, 8, 100):
            owner = linear_owner(np.arange(m), m, size)
            for rank in range(size):
                n, off = linear_distribution(m, FakeComm(size, rank))
                assert (owner[off : off + n] == rank).all()


def test_linear_distribution_world():

    n, off = linear_distribution(10, comm)
    assert n == comm.allreduce(n) // comm.Get_size() + (1 if comm.Get_rank() < 10 % comm.Get_size() else 0)
    assert comm.allreduce(n) == 10
