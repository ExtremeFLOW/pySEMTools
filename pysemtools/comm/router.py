"This module contains the class router"

import numpy as np

NoneType = type(None)

int32_limit = np.int64(2 ** 31 - 1)

class Router:
    """
    This class can be used to handle communication between ranks in a MPI communicator.

    With this one can send and recieve data to any rank that is specified in the destination list.

    Parameters
    ----------
    comm : MPI communicator
        The MPI communicator that is used for the communication.

    Attributes
    ----------
    comm : MPI communicator
        The MPI communicator that is used for the communication.

    destination_count : ndarray
        Specifies a buffer to see how many points I send to each rank

    source_count : ndarray
        Specifies a buffer to see how many points I recieve from each rank

    Notes
    -----
    The data is always flattened before sending and recieved data is always flattened.
    The user must reshape the data after recieving it.

    Examples
    --------
    To initialize simply use the communicator

    >>> from mpi4py import MPI
    >>> from pysemtools.comm.router import Router
    >>> comm = MPI.COMM_WORLD
    >>> rt = Router(comm)
    """

    def __init__(self, comm):

        self.comm = comm
        # Specifies a buffer to see how many points I send to each rank
        self.destination_count = np.zeros((comm.Get_size()), dtype=np.int64)
        # Specifies a buffer to see how many points I recieve from each rank
        self.source_count = np.zeros((comm.Get_size()), dtype=np.int64)
        # Displacements for all to all communication
        self.destination_displacement = np.zeros((comm.Get_size()), dtype=np.int64)
        self.source_displacement = np.zeros((comm.Get_size()), dtype=np.int64)

    def transfer_data(self, comm_pattern, **kwargs):
        """
        Moves data between ranks in the specified patthern.

        This method wraps others in this class.

        Parameters
        ----------
        keyword : str
            The keyword that specifies the pattern of the data movement.
            current options are:
                - distribute_p2p: sends data to specified destinations.
                  and recieves data from whoever sent. point to point comm
                - distribute_a2a: sends data to specified destinations.
                  and recieves data from whoever sent. all to all comm
                - gather: gathers data from all processes to the root process.
                - scatter: scatters data from the root process to all other processes.
        kwargs : dict
            The arguments that are passed to the specified pattern.
            One needs to check the pattern documentation for the required arguments.
            For each scenario

        Returns
        -------
        tuple
            The output of the specified pattern.
        """

        router_factory = {
            "point_to_point": self.send_recv,
            "collective": self.all_to_all,
        }

        if comm_pattern not in router_factory:
            raise ValueError(f"Method '{comm_pattern}' not recognized.")

        return router_factory[comm_pattern](**kwargs)

    def send_recv(self, destination=None, data=None, dtype=None, tag=None):
        """
        Sends data to specified destinations and recieves data from whoever sent.

        Typically, when a rank needs to send some data, it also needs to recieve some.
        In this method this is done by using non blocking communication.
        We note, however, that when the method returns, the data is already recieved.

        Parameters
        ----------
        destination : list
            A list with the rank ids that the data should be sent to.
        data : list or ndarray
            The data that will be sent. If it is a list,
            the data will be sent to the corresponding destination.
            if the data is an ndarray, the same data will be sent to all destinations.
        dtype : dtype
            The data type of the data that is sent.
        tag : int
            Tag used to identify the messages.

        Returns
        -------
        sources : list
            A list with the rank ids that the data was recieved from.
        recvbuff : list
            A list with the recieved data. The data is stored in the same order as the sources.

        Examples
        --------
        To send and recieve data between ranks, do the following:

        local_data = np.zeros(((rank+1)*10, 3), dtype=np.double)

        >>> rt = Router(comm)
        >>> destination = [rank + 1, rank + 2]
        >>> for i, dest in enumerate(destination):
        >>>     if dest >= size:
        >>>         destination[i] = dest - size
        >>> sources, recvbf = rt.send_recv(destination = destination,
        >>>                   data = local_data, dtype=np.double, tag = 0)
        >>> for i in range(0, len(recvbf)):
        >>>     recvbf[i] = recvbf[i].reshape((-1, 3))
        """

        # ===========================
        # Fill the destination count
        # ===========================

        self.destination_count[:] = 0
        # Check if the data to send is a list
        if isinstance(data, list):
            # If it is a list, match each destination with its data
            for dest_ind, dest in enumerate(destination):
                self.destination_count[dest] = data[dest_ind].size
        else:
            # If it is not a list, send the same data to all destinations
            self.destination_count[destination] = data.size

        # ======================
        # Fill the source count
        # ======================
        self.source_count[:] = 0
        self.comm.Alltoall(sendbuf=self.destination_count, recvbuf=self.source_count)
        sources = np.where(self.source_count != 0)[0]

        # Check if any message is too large
        check_sendrecv_counts(self.comm, self.source_count)

        # =========================
        # Allocate recieve buffers
        # =========================
        recvbuff = [
            np.zeros((self.source_count[source]), dtype=dtype) for source in sources
        ]

        # =========================
        # Send and recieve the data
        # =========================

        ## set up recieve request
        recvreq = [
            self.comm.Irecv(recvbuff[source_ind], source=source, tag=tag)
            for source_ind, source in enumerate(sources)
        ]

        ## Set up and complete the send request
        if isinstance(data, list):
            # If it is a list, send matching position to destination
            for dest_ind, dest in enumerate(destination):
                sendreq = self.comm.Isend(data[dest_ind].flatten(), dest=dest, tag=tag)
                sendreq.wait()
        else:
            # If it is not a list, send the same data to all destinations
            for dest_ind, dest in enumerate(destination):
                sendreq = self.comm.Isend(data.flatten(), dest=dest, tag=tag)
                sendreq.wait()

        ## complete the recieve request
        for req in recvreq:
            req.wait()

        return sources, recvbuff

    def all_to_all(self, destination=None, data=None, dtype=None, **kwargs):
        """
        Sends data to specified destinations and recieves data from whoever sent.

        In this instance we use all to all collective.

        Parameters
        ----------
        destination : list
            A list with the rank ids that the data should be sent to.
        data : list or ndarray
            The data that will be sent. If it is a list,
            the data will be sent to the corresponding index in the destination list.
            if the data is an ndarray, the same data will be sent to all destinations.
        dtype : dtype
            The data type of the data that is sent.

        Notes
        -----
        Extra keyword arguments are ignored. This is to keep the same interface as the send_recv method.

        Returns
        -------
        sources : list
            A list with the rank ids that the data was recieved from.
        recvbuff : list
            A list with the recieved data. The data is stored in the same order as the sources.

        Examples
        --------
        To send and recieve data between ranks, do the following:

        local_data = np.zeros(((rank+1)*10, 3), dtype=np.double)

        >>> rt = Router(comm)
        >>> destination = [rank + 1, rank + 2]
        >>> for i, dest in enumerate(destination):
        >>>     if dest >= size:
        >>>         destination[i] = dest - size
        >>> sources, recvbf = rt.all_to_all(destination = destination,
        >>>                   data = local_data, dtype=np.double)
        >>> for i in range(0, len(recvbf)):
        >>>     recvbf[i] = recvbf[i].reshape((-1, 3))
        """

        # ===========================
        # Fill the destination count
        # ===========================

        self.destination_count[:] = 0
        # Check if the data to send is a list
        if isinstance(data, list):
            # If it is a list, match each destination with its data
            for dest_ind, dest in enumerate(destination):
                self.destination_count[dest] = data[dest_ind].size
        else:
            # If it is not a list, send the same data to all destinations
            self.destination_count[destination] = data.size

        # =====================
        # Fill the source count
        # =====================
        self.source_count[:] = 0
        self.comm.Alltoall(sendbuf=self.destination_count, recvbuf=self.source_count)
        sources = np.where(self.source_count != 0)[0]

        # Check if any message is too large
        check_sendrecv_counts(self.comm, self.source_count)

        # ==============================
        # Allocate send and recv buffers
        # as flattened arrays
        # ==============================
        if isinstance(data, list):
            # Create a send buffer that globally holds all
            # the send counts for each destination
            sendbuff = np.zeros((np.sum(self.destination_count)), dtype=dtype)
        else:
            # If we are sending the same data everywhere, then no need to
            # duplicate the data. We will just set the dispalcements to 0
            # later.
            sendbuff = np.zeros((data.size), dtype=dtype)
        recvbuff = np.zeros((np.sum(self.source_count)), dtype=dtype)

        # ===========================
        # Calculate the dispaclements
        # ===========================
        if isinstance(data, list):
            # Check where the data for each rank starts in the global buffer
            self.destination_displacement[:] = 0
            for i in range(1, len(self.destination_displacement)):
                self.destination_displacement[i] = (
                    self.destination_displacement[i - 1] + self.destination_count[i - 1]
                )
        else:
            # The same data goes everywhere, so the displacement for all
            # destinations is 0
            self.destination_displacement[:] = 0

        self.source_displacement[:] = 0
        for i in range(1, len(self.source_displacement)):
            self.source_displacement[i] = (
                self.source_displacement[i - 1] + self.source_count[i - 1]
            )

        # ========================
        # Populate the send buffer
        # ========================

        if isinstance(data, list):
            # If it is a list, send matching position to destination
            for dest_ind, dest in enumerate(destination):
                sendbuff[
                    self.destination_displacement[dest] : self.destination_displacement[
                        dest
                    ]
                    + data[dest_ind].size
                ] = data[dest_ind].flatten()
        else:
            # If it is not a list, send the same data to all destinations.
            # This one works like this since we set the displacement to 0.
            sendbuff[:] = data.flatten()

        # =========================
        # Send and recieve the data
        # =========================

        self.comm.Alltoallv(
            sendbuf=(sendbuff, (self.destination_count, self.destination_displacement)),
            recvbuf=(recvbuff, (self.source_count, self.source_displacement)),
        )

        # =========================
        # Reshape the data
        # =========================

        recvbuff = [
            recvbuff[
                self.source_displacement[source] : self.source_displacement[source]
                + self.source_count[source]
            ]
            for source in sources
        ]

        return sources, recvbuff

    def gather_in_root(self, data=None, root=0, dtype=None):
        """
        Gathers data from all processes to the root process.

        This is a wrapper to the MPI Gatherv function.

        Parameters
        ----------
        data : ndarray
            Data that is gathered in the root process.
        root : int
            The rank that will gather the data.
        dtype : dtype
            The data type of the data that is gathered.

        Returns
        -------
        recvbuf : ndarray
            The gathered data in the root process.
            The data is always recieved flattened. User must reshape it.
        sendcounts : ndarray
            The number of data that was sent from each rank.

        Examples
        --------
        To gather data from all ranks to the root rank, do the following:

        >>> rt = Router(comm)
        >>> local_data = np.ones(((rank+1)*10, 3), dtype=np.double)*rank
        >>> recvbf, sendcounts = rt.gather_in_root(data = local_data,
        >>>                      root = 0, dtype = np.double)
        """

        rank = self.comm.Get_rank()

        # Populate the send buffer
        sendbuff = np.zeros((data.size), dtype=dtype)
        sendbuff[:] = data.flatten()

        # Collect local array sizes using the high-level mpi4py gather
        sendcounts = np.array(self.comm.allgather(data.size), dtype=np.int64)
        if rank == root:
            # print("sendcounts: {}, total: {}".format(sendcounts, np.sum(sendcounts)))
            recvbuf = np.empty(np.sum(sendcounts), dtype=dtype)
        else:
            recvbuf = None

        # Chunk the data if it is too large
        if np.any(sendcounts >= int32_limit) or np.sum(sendcounts) >= int32_limit:

            # Initialize the recieve displacement
            recv_pos = 0

            # Initialize buffers to keep track of moved data 
            chunk_sent = np.zeros_like(sendcounts)
            chunk_sendcounts = np.zeros_like(sendcounts)
            chunk_msg = True
            
            if self.comm.Get_rank() == root:
                print("Data size is too large for a single send, using chunks")

            while chunk_msg:

                # Identify the number of data that is left to send
                chunk_sendcounts = sendcounts - chunk_sent
                # Get the cumulative sum of the sendcounts to be sent
                sum_chunk_sendcounts = np.cumsum(chunk_sendcounts)

                # As soon as one rank has too much data, only sent data up to that rank
                # This is done this way to avoid the int32 limit, while keeping it simple
                # to put in the full recieve buffer that is returned.          
                already_found_limit = False
                for i in range(len(chunk_sendcounts)):
                    if not already_found_limit:
                        # If The sum is not too large for the rank, do nothing
                        ## If the sum of the sendcounts is too large, set the sendcount of that rank to the limit
                        if sum_chunk_sendcounts[i] >= int32_limit:
                            if i == 0:
                                chunk_sendcounts[i] = int32_limit
                            else:
                                chunk_sendcounts[i] = int32_limit - sum_chunk_sendcounts[i-1]
                            already_found_limit = True

                    else:
                        # If we have found one rank that has too large sendcount,
                        # set all the subsequent rank sendcount to 0
                        chunk_sendcounts[i] = 0
                
                # Reset for next iteration
                already_found_limit = False

                # Set a temporary recieve buffer with the size of the current chunk
                if rank == root:
                    # print("sendcounts: {}, total: {}".format(sendcounts, np.sum(sendcounts)))
                    temp_recvbuf = np.empty(np.sum(chunk_sendcounts), dtype=dtype)
                else:
                    temp_recvbuf = None

                # Each rank starts from the index that was left off in the previous iteration                
                send_start_id = chunk_sent[rank]
                # and end at the sendcount of the current iteration
                send_end_id = send_start_id + chunk_sendcounts[rank]
                self.comm.Gatherv(sendbuf=sendbuff[send_start_id:send_end_id], recvbuf=(temp_recvbuf, chunk_sendcounts), root=root)

                # In the root rank, update the recieve buffer with the data recieved this iteration
                if rank == root:
                    recvbuf[recv_pos:recv_pos+np.sum(chunk_sendcounts)] = temp_recvbuf
                    recv_pos = recv_pos + np.sum(chunk_sendcounts)

                # Update the sendcounts that have been sent 
                chunk_sent = chunk_sent + chunk_sendcounts

                # Once the sent sendcounts are the same that the original sendcounts,
                # the process is over.
                if np.sum(chunk_sent) == np.sum(sendcounts):
                    chunk_msg = False
                    break 
        
        # Or simply send the data
        else:

            self.comm.Gatherv(sendbuf=sendbuff, recvbuf=(recvbuf, sendcounts), root=root)

        return recvbuf, sendcounts

    def scatter_from_root(self, data=None, sendcounts=None, root=0, dtype=None):
        """
        Scatters data from the root process to all other processes.

        This is a wrapper to the MPI Scatterv function.

        Parameters
        ----------
        data : ndarray
            The data that is scattered to all processes.
        sendcounts : ndarray, optional
            The number of data that is sent to each process.
            If not specified, the data is divided equally among all processes.
        root : int
            The rank that will scatter
        dtype : dtype
            The data type of the data that is scattered.

        Returns
        -------
        recvbuf : ndarray
            The scattered data in the current process.
            The data is always recieved flattened. User must reshape it.

        Examples
        --------
        To scatter data from the root rank, do the following:

        >>> rt = Router(comm)
        >>> recvbf = rt.scatter_from_root(data = recvbf,
                        sendcounts=sendcounts, root = 0, dtype = np.double)
        >>> recvbf = recvbf.reshape((-1, 3))

        Note tha the sendcounts are just a ndarray of size comm.Get_size()
        with the number of data that is sent to each rank.
        """

        if self.comm.Get_rank() == root:
            if isinstance(sendcounts, NoneType):
                # Divide the data equally among all processes
                sendcounts = np.zeros((self.comm.Get_size()), dtype=np.int64)
                sendcounts[:] = data.size // self.comm.Get_size()

            sendbuf = data.flatten()
        else:
            sendbuf = None

        # Check if any message is too large
        check_sendrecv_counts(self.comm, sendcounts)

        rank = self.comm.Get_rank()
        recvbuf = np.ones(sendcounts[rank], dtype=dtype) * -100

        self.comm.Scatterv(sendbuf=(sendbuf, sendcounts), recvbuf=recvbuf, root=root)

        return recvbuf

    def all_gather(self, data=None, dtype=None):
        """
        Gathers data from all processes to all processes.

        This is a wrapper to the MPI Allgatherv function.

        Parameters
        ----------
        data : ndarray
            Data that is gathered in all processes.
        dtype : dtype
            The data type of the data that is gathered.

        Returns
        -------
        recvbuf : ndarray
            The gathered data in the root process.
            The data is always recieved flattened. User must reshape it.
        sendcounts : ndarray
            The number of data that was sent from each rank.

        Examples
        --------
        To gather data from all ranks to the root rank, do the following:

        >>> rt = Router(comm)
        >>> local_data = np.ones(((rank+1)*10, 3), dtype=np.double)*rank
        >>> recvbf, sendcounts = rt.all_gather(data = local_data, dtype = np.double)
        """

        rank = self.comm.Get_rank()

        if isinstance(data, np.ndarray):
            data = data.flatten()
            count = data.size
        else:
            data = np.ones((1), dtype=dtype) * data
            count = 1

        # Collect local array sizes using the high-level mpi4py gather
        sendcounts = np.array(self.comm.allgather(count), dtype=np.int64)

        # Check if any message is too large
        check_sendrecv_counts(self.comm, sendcounts)

        recvbuf = np.empty(np.sum(sendcounts), dtype=dtype)

        self.comm.Allgatherv(sendbuf=data, recvbuf=(recvbuf, sendcounts))

        return recvbuf, sendcounts

def check_sendrecv_counts(comm, sendrecv_count: np.ndarray):

    if np.any(sendrecv_count >= int32_limit):
        raise ValueError("Send/Recv sendcount is too large for a single send according to MPI standard (max int32 = 2**31 -1 counts), use chunks or more ranks")
    elif np.any(sendrecv_count < 0):
        raise ValueError("Send/Recv sendcount cannot be negative, you might have overflowed the int32 limit")
    