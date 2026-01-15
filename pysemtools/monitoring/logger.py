""" Module that contains classes for the logging of messages during the POD calculation"""

import logging
import sys
import os
from mpi4py.MPI import Wtime as time

USE_COLORS = os.getenv("PYSEMTOOLS_USE_COLORS", "False").lower() in ("true", "1", "t")
DEBUG = os.getenv("PYSEMTOOLS_DEBUG", "False").lower() in ("true", "1", "t")
HIDE = os.getenv("PYSEMTOOLS_HIDE_LOG", "False").lower() in ("true", "1", "t")


# Modified from https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
class CustomFormatter(logging.Formatter):
    """Custom formatter for the log messages"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # formatt = (
    #    "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    # )
    #formatt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatt = "%(asctime)s  %(name)-20s  %(levelname)-8s  %(message)s"


    FORMATS_colored = {
        logging.DEBUG: grey + formatt + reset,
        logging.INFO: grey + formatt + reset,
        logging.WARNING: yellow + formatt + reset,
        logging.ERROR: red + formatt + reset,
        logging.CRITICAL: bold_red + formatt + reset,
    }

    FORMATS_no_color = {
        logging.DEBUG: formatt,
        logging.INFO: formatt,
        logging.WARNING: formatt,
        logging.ERROR: formatt,
        logging.CRITICAL: formatt,
    }

    def format(self, record):

        # Check if output is redirected to a file
        if USE_COLORS:
            log_fmt = self.FORMATS_colored.get(record.levelno)
        else:
            log_fmt = self.FORMATS_no_color.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    """Class that takes charge of logging messages
    
    This class takes care of logging only in one rank and setting differnent logging levels.

    Generally, the levels are set using the environment variables PYSEMTOOLS_DEBUG and PYSEMTOOLS_HIDE_LOG.

    Parameters
    ----------
    level : int, optional
        Logging level. The default is None, which sets it to logging.INFO.
    comm : MPI.Comm
        MPI communicator.
    module_name : str, optional
        Name of the module that is using the logger. The default is None.
    
    Attributes
    ----------
    log : logging.Logger
        Logger object that handles the logging
    comm : MPI.Comm
        MPI communicator
    sync_time : dict
        Dictionary to store the times for sync_tic/sync_toc methods
    time : float
        Variable to store the time for tic/toc methods
    
    Methods
    -------
    write(level, message)
        Method that writes messages in the log
    tic()
        Store the current time.
    toc()
        Write elapsed time since the last call to tic.
    sync_tic(id=0)
        Store the current time.
    sync_toc(id=0, message=None, time_message="Elapsed time: ")
        Write elapsed time since the last call to tic.
    """

    def __init__(self, level=None, comm=None, module_name=None):

        if isinstance(level, type(None)):
            level = logging.INFO
        elif level == "debug":
            level = logging.DEBUG
        elif level == "info":
            level = logging.INFO
        elif level == "warning":
            level = logging.WARNING
        elif level == "error":
            level = logging.ERROR
        elif level == "critical":
            level = logging.CRITICAL
        elif level == "hide":
            level = logging.CRITICAL
        if DEBUG:
            level = logging.DEBUG
        if HIDE:
            level = logging.CRITICAL

        self.comm = comm

        # Instanciate
        if module_name:
            logger = logging.getLogger(module_name)
        else:
            logger = logging.getLogger(__name__)

        logger.setLevel(level)

        # create console handler with a higher log level
        if not logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(level)
            ch.setFormatter(CustomFormatter())
            logger.addHandler(ch)

        logger.propagate = False

        self.log = logger

        self.sync_time = {}

        # if DEBUG:
        #    self.write("warning", "Debug mode activated - This will produce a lot of output.")
        #    self.write("warning", "Options where all ranks write are followed by a comm barrier.")
        #    self.write("warning", "bad for performance. Do not use debug mode in production.")

    def tic(self):
        """
        Store the current time.

        Returns
        -------
        None.

        """

        self.time = time()
    
    def sync_tic(self, id = 0):
        """
        Store the current time in the attibute sync_time with key id.

        Returns
        -------
        None.

        """
        
        self.comm.Barrier()
        self.sync_time[id] = time()

    def toc(self, message=None, time_message="Elapsed time: ", level="info"):
        """
        Write elapsed time since the last call to tic.

        """

        if message is None:
            self.write(level, f"{time_message}{time() - self.time}s")
        else:
            self.write(level, f"{message} - {time_message}{time() - self.time}s")
    
    def sync_toc(self, id = 0, message=None, time_message="Elapsed time: ", level="info"):
        """
        Write elapsed time since the last call to tic for the given id in the sync_time attribute.

        """

        self.comm.Barrier()
        if message is None:
            self.write(level, f"{time_message}{time() - self.sync_time[id]}s")
        else:
            self.write(level, f"{message} - {time_message}{time() - self.sync_time[id]}s")

    def write(self, level, message):
        """Writes messages in the log
        
        Parameters
        ----------
        level : str
            Level of the message. Possible values are: "debug_all", "debug", "info
            "info_all", "warning", "error", "critical".
        message : str
            Message to be logged.

        """
        comm = self.comm
        rank = comm.Get_rank()

        if level == "debug_all":
            # self.log.debug(message)
            # comm.Barrier()
            if rank == 0:
                self.log.debug("This debug message type is not implemented yet.")

        if level == "debug":
            if rank == 0:
                self.log.debug(message)

        if level == "info":
            if rank == 0:
                self.log.info(message)

        if level == "info_all":
            comm.Barrier()
            self.log.info(f"Rank {comm.Get_rank()}: {message}")
            
        if level == "warning":
            if rank == 0:
                self.log.warning(message)

        if level == "error":
                self.log.error(message)

        if level == "critical":
                self.log.critical(message)
