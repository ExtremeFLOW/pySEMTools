"""
Registry of file conversions.

A conversion turns a file of one format into a file of another. Each one is
a function registered under a ``(source, target)`` pair of format names,
together with the options it accepts and whether it runs in parallel.
:func:`convert` detects the formats of two paths from their names, looks up
the conversion, checks the rank count and the options and runs it. New
conversions are added by registering a :class:`Conversion`, which also makes
them available to the ``pysemtools_convert`` command line tool.
"""

import os
import re
from dataclasses import dataclass

from mpi4py import MPI

from ..monitoring.logger import Logger
from ..comm.utils import require_single_rank

__all__ = [
    "Format",
    "FORMATS",
    "detect_format",
    "Option",
    "Conversion",
    "register",
    "conversions",
    "get_conversion",
    "convert",
]


@dataclass(frozen=True)
class Format:
    """
    A file format known to the converter.

    Parameters
    ----------
    name : str
        Short name used on the command line, for example ``"nmsh"``.
    description : str
        One line description.
    pattern : str
        Regular expression matched against the file name to detect the
        format.
    """

    name: str
    description: str
    pattern: str

    def matches(self, path):
        """True if the name of ``path`` matches the format pattern."""
        return re.search(self.pattern, os.path.basename(path)) is not None


#: Known formats by name.
FORMATS = {
    "re2": Format("re2", "NEKTON/Nek5000 .re2 mesh", r"\.re2$"),
    "nmsh": Format("nmsh", "Neko .nmsh mesh", r"\.nmsh$"),
    "fld": Format("fld", "Nek5000 field file (.f##### or .fld)", r"\.f\d{5}$|\.fld$"),
}


def detect_format(path):
    """
    Detect the format of a file from its name.

    Parameters
    ----------
    path : str
        Path of the file.

    Returns
    -------
    str
        Name of the format, a key of :data:`FORMATS`.

    Raises
    ------
    ValueError
        If no known format matches the name.
    """
    for fmt in FORMATS.values():
        if fmt.matches(path):
            return fmt.name
    known = ", ".join(f"{f.name} ({f.description})" for f in FORMATS.values())
    raise ValueError(
        f"Cannot tell the format of {path} from its name; known formats are {known}"
    )


@dataclass(frozen=True)
class Option:
    """
    An option of a conversion, passed as a keyword argument to its function.

    Parameters
    ----------
    name : str
        Keyword argument name. On the command line it becomes
        ``--name`` with underscores replaced by dashes.
    type : callable
        Type used to parse the command line value.
    help : str
        Help text.
    default : str, optional
        Description of the default shown in the help. None marks the option
        as required.
    choices : tuple, optional
        Allowed values.
    """

    name: str
    type: callable
    help: str
    default: str = None
    choices: tuple = None

    @property
    def required(self):
        """True if the option has no default."""
        return self.default is None

    @property
    def flag(self):
        """The command line flag of the option."""
        return "--" + self.name.replace("_", "-")

    def add_to_parser(self, parser):
        """Add the option to an ``argparse`` parser, with None as the default."""
        text = self.help if self.required else f"{self.help} (default {self.default})"
        kwargs = {"dest": self.name, "type": self.type, "default": None, "help": text}
        if self.choices is not None:
            kwargs["choices"] = self.choices
        parser.add_argument(self.flag, **kwargs)


@dataclass(frozen=True)
class Conversion:
    """
    A registered conversion between two formats.

    Parameters
    ----------
    source : str
        Name of the input format.
    target : str
        Name of the output format.
    func : callable
        Called as ``func(input_path, output_path, comm=comm, **options)``.
    description : str
        One line description.
    parallel : bool, optional
        True if the function runs on any number of ranks. Default is False,
        in which case a single rank is required.
    options : tuple of Option, optional
        Options accepted by the function.
    """

    source: str
    target: str
    func: callable
    description: str
    parallel: bool = False
    options: tuple = ()

    @property
    def key(self):
        """The ``(source, target)`` pair."""
        return (self.source, self.target)

    @property
    def option_names(self):
        """Names of the accepted options."""
        return tuple(opt.name for opt in self.options)

    def __str__(self):
        return f"{self.source} -> {self.target}"


_CONVERSIONS = {}


def register(conversion):
    """
    Register a conversion.

    Parameters
    ----------
    conversion : Conversion
        The conversion to add. Its formats must be in :data:`FORMATS` and
        the pair must not be registered yet.

    Returns
    -------
    Conversion
        The registered conversion.
    """
    for name in conversion.key:
        if name not in FORMATS:
            raise ValueError(f"Unknown format {name}; add it to FORMATS first")
    if conversion.key in _CONVERSIONS:
        raise ValueError(f"Conversion {conversion} is already registered")
    _CONVERSIONS[conversion.key] = conversion
    return conversion


def conversions():
    """
    List the registered conversions.

    Returns
    -------
    list of Conversion
        The conversions, in registration order.
    """
    return list(_CONVERSIONS.values())


def get_conversion(source, target):
    """
    Look up the conversion between two formats.

    Parameters
    ----------
    source : str
        Name of the input format.
    target : str
        Name of the output format.

    Returns
    -------
    Conversion
        The registered conversion.

    Raises
    ------
    ValueError
        If no conversion is registered for the pair.
    """
    try:
        return _CONVERSIONS[(source, target)]
    except KeyError:
        available = ", ".join(str(c) for c in _CONVERSIONS.values())
        raise ValueError(
            f"No conversion from {source} to {target}; available conversions are {available}"
        ) from None


def convert(input_path, output_path, comm=None, source=None, target=None, **options):
    """
    Convert a file from one format to another.

    The formats are detected from the file names unless given explicitly.
    A conversion that does not run in parallel requires ``comm`` to have a
    single rank.

    Parameters
    ----------
    input_path : str
        Input file.
    output_path : str
        Output file. Must differ from the input file.
    comm : MPI.Comm, optional
        MPI communicator. Default is ``MPI.COMM_WORLD``.
    source : str, optional
        Name of the input format. Default is detected from ``input_path``.
    target : str, optional
        Name of the output format. Default is detected from ``output_path``.
    **options
        Options of the conversion, see :func:`conversions`.

    Returns
    -------
    object
        Whatever the conversion function returns, typically the converted
        mesh.

    Raises
    ------
    ValueError
        If a format cannot be detected, the pair has no conversion, or the
        output is the input.
    TypeError
        If an option is not accepted by the conversion, or a required one
        is missing.
    RuntimeError
        If a serial conversion is run on several ranks.

    Examples
    --------
    >>> from pysemtools.convert import convert
    >>> convert("hemi.re2", "hemi.nmsh")
    >>> convert("hemi.nmsh", "hemi0.f00000", order=5)
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    if source is None:
        source = detect_format(input_path)
    if target is None:
        target = detect_format(output_path)
    conv = get_conversion(source, target)

    unknown = sorted(set(options) - set(conv.option_names))
    if unknown:
        raise TypeError(f"Conversion {conv} does not accept the option(s) {', '.join(unknown)}")
    missing = [opt.name for opt in conv.options if opt.required and opt.name not in options]
    if missing:
        raise TypeError(f"Conversion {conv} requires the option(s) {', '.join(missing)}")
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file {input_path} does not exist")
    if os.path.exists(output_path) and os.path.samefile(input_path, output_path):
        raise ValueError(f"Output {output_path} is the input file")
    if not conv.parallel:
        require_single_rank(comm, f"The {conv} conversion")

    log = Logger(comm=comm, module_name="convert")
    log.write("info", f"Converting {input_path} ({source}) to {output_path} ({target})")
    return conv.func(input_path, output_path, comm=comm, **options)
