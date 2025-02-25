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
    formatt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

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
    """Class that takes charge of logging messages during POD execution"""

    def __init__(self, level=None, comm=None, module_name=None):

        if isinstance(level, type(None)):
            level = logging.INFO
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

    def toc(self):
        """
        Write elapsed time since the last call to tic.

        """

        self.write("info", f"Elapsed time: {time() - self.time}s")

    def write(self, level, message):
        """Method that writes messages in the log"""
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

        if level == "warning":
            if rank == 0:
                self.log.warning(message)

        if level == "error":
                self.log.error(message)

        if level == "critical":
                self.log.critical(message)
