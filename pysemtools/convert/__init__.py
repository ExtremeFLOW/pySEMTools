"""
Conversion between the file formats known to pySEMTools.

The :func:`convert` function turns a file into another format, choosing the
conversion from the file names. The conversions themselves are functions
registered in :mod:`pysemtools.convert.registry`; importing this package
registers the ones shipped with pySEMTools. The same registry drives the
``pysemtools_convert`` command line tool.
"""

from .registry import (
    Format,
    FORMATS,
    detect_format,
    Option,
    Conversion,
    register,
    conversions,
    get_conversion,
    convert,
)
from .re2_to_nmsh import re2_to_nmsh
from .mesh_to_fld import mesh_to_fld

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
    "re2_to_nmsh",
    "mesh_to_fld",
]
