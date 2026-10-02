Format conversion
-----------------

The :mod:`pysemtools.convert` module converts files between the formats known to pySEMTools.
A conversion is chosen from the names of the input and the output file, so converting is a single call:

.. code-block:: python

    from pysemtools.convert import convert

    convert("hemi.re2", "hemi.nmsh")                 # NEKTON .re2 to Neko .nmsh
    convert("hemi.nmsh", "hemi0.f00000", order=5)    # GLL points of the mesh to a Nek5000 field file

The same conversions are available from the command line through ``pysemtools_convert``:

.. code-block:: bash

    pysemtools_convert hemi.re2 hemi.nmsh
    mpirun -n 4 pysemtools_convert hemi.nmsh hemi0.f00000 --order 5
    pysemtools_convert --list

The conversions shipped with the package are:

==========  ==========  ==========  ==================================================================
Input       Output      Runs        Description
==========  ==========  ==========  ==================================================================
``re2``     ``nmsh``    serial      Convert a NEKTON ``.re2`` mesh to Neko ``.nmsh``.
                                    Option ``periodic_tol``.
``re2``     ``fld``     parallel    Write the GLL points of the mesh at a chosen polynomial order to
                                    a Nek5000 field file without data. Options ``order`` and ``wdsz``.
``nmsh``    ``fld``     parallel    As above, from a Neko mesh.
==========  ==========  ==========  ==================================================================

Parallel conversions read the input distributed over the ranks and write the output collectively,
so they can be launched with ``mpirun``. Serial conversions require a single rank.
The field files hold only the mesh, that is, the coordinates of ``order + 1`` GLL points per
direction of every element. Curved edges are not applied.

Adding a conversion
~~~~~~~~~~~~~~~~~~~

A conversion is a function ``func(input_path, output_path, comm=comm, **options)`` registered with
:func:`pysemtools.convert.register`. Registering it makes it available to :func:`convert` and to the
command line tool, including its options:

.. code-block:: python

    from pysemtools.convert import Conversion, Option, register

    register(
        Conversion(
            "nmsh",
            "vtk",
            nmsh_to_vtk,
            "write a Neko mesh as VTK",
            parallel=False,
            options=(Option("binary", bool, "write binary VTK", default="False"),),
        )
    )

New formats are added to :data:`pysemtools.convert.FORMATS` with a name and the file name pattern
used to detect them.

------------------

.. automodule :: pysemtools.convert.registry
    :members:

.. automodule :: pysemtools.convert.re2_to_nmsh
    :members:

.. automodule :: pysemtools.convert.mesh_to_fld
    :members:
