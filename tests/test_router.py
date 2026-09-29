# Initialize MPI
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank, size = comm.Get_rank(), comm.Get_size()

# Import general modules
import numpy as np

# Import relevant modules
from pysemtools.comm.router import Router, record_datatype

REC_DT = np.dtype([("id", "<i4"), ("v", "<f8", (3,)), ("tag", "S4")])


def local_records(n, offset):
    rec = np.zeros(n, dtype=REC_DT)
    rec["id"] = offset + np.arange(n)
    rec["v"] = rec["id"][:, None] * np.array([1.0, 0.5, 0.25])
    rec["tag"] = b"r%03d" % rank
    return rec


def test_record_datatype():

    rec_t = record_datatype(REC_DT)
    assert rec_t.Get_size() == REC_DT.itemsize
    rec_t.Free()


def test_all_gather_records():

    rt = Router(comm)
    n = 2 + rank
    offset = sum(2 + r for r in range(rank))
    gathered, counts = rt.all_gather(data=local_records(n, offset), dtype=REC_DT)
    assert gathered.dtype == REC_DT
    assert np.array_equal(counts, [2 + r for r in range(size)])
    # concatenated in rank order: ids are contiguous and every rank's tag appears
    assert np.array_equal(gathered["id"], np.arange(gathered.size))
    assert set(gathered["tag"].tolist()) == {b"r%03d" % r for r in range(size)}
    assert np.allclose(gathered["v"][:, 1], 0.5 * gathered["id"])


def test_gather_in_root_records():

    rt = Router(comm)
    gathered, counts = rt.gather_in_root(data=local_records(3, 3 * rank), root=0, dtype=REC_DT)
    assert np.array_equal(counts, np.full(size, 3))
    if rank == 0:
        assert np.array_equal(gathered["id"], np.arange(3 * size))
    else:
        assert gathered is None


def test_all_to_all_records():

    rt = Router(comm)
    # every rank sends r + 1 records to every rank r
    destinations = list(range(size))
    data = [local_records(r + 1, 100 * rank) for r in destinations]
    sources, chunks = rt.all_to_all(destination=destinations, data=data, dtype=REC_DT)
    assert np.array_equal(sources, destinations)
    for src, chunk in zip(sources, chunks):
        assert chunk.dtype == REC_DT
        assert chunk.size == rank + 1
        assert (chunk["tag"] == b"r%03d" % src).all()
        assert np.array_equal(chunk["id"], 100 * src + np.arange(rank + 1))


def test_scatter_from_root_records():

    rt = Router(comm)
    counts = np.array([r + 1 for r in range(size)], dtype=np.int64)
    data = local_records(int(counts.sum()), 0) if rank == 0 else None
    received = rt.scatter_from_root(data=data, sendcounts=counts, root=0, dtype=REC_DT)
    assert received.size == rank + 1
    assert np.array_equal(received["id"], int(counts[:rank].sum()) + np.arange(rank + 1))


def test_scalar_data_still_works():

    rt = Router(comm)
    gathered, _ = rt.all_gather(data=np.full(2, rank, dtype=np.double), dtype=np.double)
    assert np.array_equal(gathered, np.repeat(np.arange(size, dtype=np.double), 2))
