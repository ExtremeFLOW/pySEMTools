#!/usr/bin/env python3
""" Command line tool to convert files between the formats known to pySEMTools. """

import argparse
import sys

from mpi4py import MPI

from pysemtools.convert import FORMATS, conversions, get_conversion, detect_format, convert


def conversion_table():
    """
    Describe the registered conversions, one line each.

    Returns
    -------
    str
        The table.
    """
    lines = []
    for conv in conversions():
        flags = ", ".join(opt.flag for opt in conv.options)
        req = ", ".join(opt.flag for opt in conv.options if opt.required)
        mode = "parallel" if conv.parallel else "serial"
        lines.append(f"  {conv.source:5s} -> {conv.target:5s} {mode:9s} {conv.description}")
        if flags:
            extra = f" (required: {req})" if req else ""
            lines.append(f"{'':27s} options: {flags}{extra}")
    return "\n".join(lines)


def build_parser():
    """
    Build the argument parser.

    The conversion options are collected from the registry, so a newly
    registered conversion shows up here without changes.

    Returns
    -------
    argparse.ArgumentParser
        The parser.
    """
    formats = "\n".join(f"  {f.name:5s} {f.description}" for f in FORMATS.values())
    parser = argparse.ArgumentParser(
        prog="pysemtools_convert",
        description="Convert a file to another format. The formats are detected from the file "
        "names unless given with --from and --to.",
        epilog=f"formats:\n{formats}\n\nconversions:\n{conversion_table()}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", nargs="?", help="input file")
    parser.add_argument("output", nargs="?", help="output file")
    parser.add_argument("--from", dest="source", choices=tuple(FORMATS), help="input format")
    parser.add_argument("--to", dest="target", choices=tuple(FORMATS), help="output format")
    parser.add_argument("--list", action="store_true", help="list the conversions and exit")

    group = parser.add_argument_group("conversion options")
    seen = set()
    for conv in conversions():
        for opt in conv.options:
            if opt.name not in seen:
                seen.add(opt.name)
                opt.add_to_parser(group)
    return parser


def given_options(args, conv):
    """
    Collect the conversion options given on the command line.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments.
    conv : Conversion
        The conversion about to run.

    Returns
    -------
    dict
        The options as keyword arguments for the conversion.

    Raises
    ------
    ValueError
        If an option of another conversion was given, or a required one is
        missing.
    """
    options = {}
    for other in conversions():
        for opt in other.options:
            value = getattr(args, opt.name)
            if value is None:
                continue
            if opt.name not in conv.option_names:
                raise ValueError(f"Option {opt.flag} does not apply to the {conv} conversion")
            options[opt.name] = value
    missing = [opt.flag for opt in conv.options if opt.required and opt.name not in options]
    if missing:
        raise ValueError(f"The {conv} conversion requires {', '.join(missing)}")
    return options


def main(argv=None):
    """
    Convert a file to another format.

    The input and output formats are detected from the file names, or given
    with ``--from`` and ``--to``. Conversions that run in parallel can be
    launched with ``mpirun``; the others require a single rank. Run
    ``pysemtools_convert --list`` to see the conversions and their options.

    Parameters
    ----------
    argv : list of str, optional
        Arguments to parse instead of ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code, 0 on success.

    Examples
    --------
    >>> pysemtools_convert hemi.re2 hemi.nmsh
    >>> mpirun -n 4 pysemtools_convert hemi.nmsh hemi0.f00000 --order 5
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    comm = MPI.COMM_WORLD
    root = comm.Get_rank() == 0

    if args.list:
        if root:
            print(conversion_table())
        return 0
    if args.input is None or args.output is None:
        parser.error("an input and an output file are required")

    try:
        source = args.source or detect_format(args.input)
        target = args.target or detect_format(args.output)
        conv = get_conversion(source, target)
        options = given_options(args, conv)
        convert(args.input, args.output, comm=comm, source=source, target=target, **options)
    except (ValueError, TypeError, RuntimeError, OSError) as ex:
        if root:
            print(f"Error: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
