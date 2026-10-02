"""Helpers for checking the communicator a routine runs on."""

__all__ = ["require_single_rank"]


def require_single_rank(comm, name):
    """
    Refuse to run a serial routine on more than one rank.

    Some routines work on the whole mesh at once, for example a converter
    that numbers points in order of first appearance. Running them under
    several ranks would only replicate the work, so they require a single
    rank instead of gathering distributed data.

    Parameters
    ----------
    comm : MPI.Comm
        MPI communicator of the run.
    name : str
        Name of the routine, used in the error message.

    Raises
    ------
    RuntimeError
        If ``comm`` has more than one rank.
    """
    if comm.Get_size() > 1:
        raise RuntimeError(
            f"{name} runs on a single rank; "
            f"launch it with one MPI rank (got {comm.Get_size()})"
        )
