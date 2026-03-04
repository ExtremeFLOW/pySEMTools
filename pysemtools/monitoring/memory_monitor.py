""" Contains the MemoryMonitor class. """

from pympler import asizeof
from memory_profiler import memory_usage


class MemoryMonitor:
    """
    Class to monitor the memory usage of the system and objects.

    This class is used to monitor the memory usage of the system and objects.

    Attributes
    ----------
    base_mem : float
        Initial memory usage of the system.

    mem_usage : float
        Current memory usage of the system.

    system_report : dict
        History of the memory usage of the system.

    object_report : dict
        History of the memory usage of the objects.

    Methods
    -------
    system_memory_usage(msg, print_msg=True)
        Report the memory usage of the system.

    object_memory_usage(comm, obj, obj_name, print_msg=True)
        Analize the memory usage of an object.

    object_memory_usage_per_attribute(comm, obj, obj_name, print_msg=True)
        Store and print the memory usage of each attribute of the object.

    """

    def __init__(self):
        self.base_mem = memory_usage()[0]
        self.mem_usage = self.base_mem
        self.system_report = dict()
        self.object_report = dict()

    def system_memory_usage(self, comm, msg, print_msg=False):
        """
        Report the memory usage of the system.

        This function is used to report the memory usage of the system.

        Parameters
        ----------
        msg : str
            Message to be printed. It is also used as keyword in the history.

        print_msg : bool, optional
            Print the message. The default is True.

        Returns
        -------
        None

        """

        self.delta = memory_usage()[0] - self.mem_usage
        self.mem_usage = memory_usage()[0]

        if print_msg:
            print(
                f"Rank: {comm.Get_rank()} - "
                + msg
                + f" - System memory usage: {self.mem_usage} MB, delta since last check: {self.delta} MB"
            )

        report = dict()
        report["memory"] = self.mem_usage
        report["delta"] = self.delta
        report["msg"] = msg
        self.system_report[msg] = report

    def object_memory_usage(self, comm, obj, obj_name, print_msg=True):
        """
        Analize the memory usage of an object.

        This function is used to check the memory usage of the object.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.

        obj : object
            Object to be analyzed.

        obj_name : str
            Name of the object.

        print_msg : bool, optional
            Print the message. The default is True.

        Returns
        -------
        None

        """
        #memory_usage = asizeof.asizeof(obj) / (1024**2)  # Convert bytes to MB

        attributes = dir(obj)
        non_callable_attributes = [
            attr
            for attr in attributes
            if not callable(getattr(obj, attr)) and not attr.startswith("__") and not attr == 'log'
        ]
        size_per_attribute = [
            asizeof.asizeof(getattr(obj, attr)) / (1024**2)
            for attr in non_callable_attributes
        ]  # Convert bytes to MB

        memory_usage = sum(size_per_attribute)

        if print_msg:
            print(
                f"Rank: {comm.Get_rank()} - Memory usage of {obj_name}: {memory_usage} MB"
            )

        if obj_name not in self.object_report:
            self.object_report[obj_name] = dict()

        report = self.object_report[obj_name]
        report["memory_usage"] = memory_usage

    def object_memory_usage_per_attribute(self, comm, obj, obj_name, print_msg=False):
        """
        Store and print the memory usage of each attribute of the object.

        This function is used to print the memory usage of each attribute of the object.
        The results are stored in the mem_per_attribute attribute.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.

        obj : object
            Object to be analyzed.

        obj_name : str
            Name of the object.

        print_msh : bool, optional
            If True, the memory usage of each attribute will be printed.

        Returns
        -------
        None

        """
        attributes = dir(obj)
        non_callable_attributes = [
            attr
            for attr in attributes
            if not callable(getattr(obj, attr)) and not attr.startswith("__") and not attr == 'log'
        ]
        size_per_attribute = [
            asizeof.asizeof(getattr(obj, attr)) / (1024**2)
            for attr in non_callable_attributes
        ]  # Convert bytes to MB

        if obj_name not in self.object_report:
            self.object_report[obj_name] = dict()
        self.object_report[obj_name]["memory_usage_per_attribute"] = dict()

        attribute_report = self.object_report[obj_name]["memory_usage_per_attribute"]
        for i, attr in enumerate(non_callable_attributes):
            attribute_report[attr] = size_per_attribute[i]

            if print_msg:
                print(
                    f"Rank: {comm.Get_rank()} - Memory usage of "
                    + obj_name
                    + f" attr - {attr}: {size_per_attribute[i]} MB"
                )

    def report_system_information(self, comm):
        """
        Report the memory usage of the system.

        This function is used to report the memory usage of the system.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.

        Returns
        -------
        None

        """

        for msg in self.system_report:
            print(
                f"Rank: {comm.Get_rank()} - {msg} - System memory usage: {self.system_report[msg]['memory']} MB, delta since last check: {self.system_report[msg]['delta']} MB"
            )

    def report_object_information(self, comm, obj_name):
        """
        Report the memory usage of the object and its attributes.

        This function is used to report the memory usage of the object and its attributes.

        Parameters
        ----------
        comm : Comm
            MPI communicator object.

        obj : object
            Object to be analyzed.

        obj_name : str
            Name of the object.

        Returns
        -------
        None

        """

        if obj_name in self.object_report:

            if "memory_usage" in self.object_report[obj_name]:
                print(
                    f"Rank: {comm.Get_rank()} - Memory usage of {obj_name}: {self.object_report[obj_name]['memory_usage']} MB"
                )

            if "memory_usage_per_attribute" in self.object_report[obj_name]:
                print(
                    f"Rank: {comm.Get_rank()} - Memory usage of {obj_name} attributes:"
                )
                for attr in self.object_report[obj_name]["memory_usage_per_attribute"]:
                    print(
                        f"Rank: {comm.Get_rank()} - Memory usage of {obj_name} attr - {attr}: {self.object_report[obj_name]['memory_usage_per_attribute'][attr]} MB"
                    )
        else:
            print(f"Rank: {comm.Get_rank()} - No information about {obj_name} object.")
