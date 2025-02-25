---
title: 'PySEMTools: A library for post-processing hexahedral spectral element data.'
tags:
  - Python
  - Computational fluid dynamics
  - Message passing interdace (MPI)
  - High performance computing
authors:
  - name: Adalberto Perez
    orcid: 0000-0001-5204-8549
    affiliation: 1 # (Multiple affiliations must be quoted)
    corresponding: true # (This is how to denote the corresponding author)
  - name: Siavash Toosi
    orcid: 0000-0001-6733-9744
    affiliation: 2
  - name: Tim Felle Olsen
    orcid: 0000-0002-4531-7226
    affiliation: 4
  - name: Stefano Markidis
    orcid: 0000-0003-0639-0639
    affiliation: 3
  - name: Philipp Schlatter
    orcid: 0000-0001-9627-5903
    affiliation: "1, 2"

affiliations:
 - name: FLOW Dept.  Engineering Mechanics, KTH Royal Institute of Technology
   index: 1
 - name: Institute of Fluid Mechanics (LSTM), Friedrich--Alexander--Universität (FAU)
   index: 2
 - name: CST, KTH Royal Institute of Technology
   index: 3
 - name: Technical University of Denmark
   index: 4
date: 18 February 2025
bibliography: paper.bib

---

# Summary

`PySEMTools` is a Python-based library for post-processing simulation data produced with high-order hexahedral elements in the context of the spectral element method in computational fluid dynamics. It aims to minimize intermediate steps typically needed when analyzing large files. Specifically, the need to use separate codebases (like the solvers themselves) at post-processing. For this effect, we leverage the use of message passing interface (MPI) for distributed computing to perform typical data processing tasks such as spectrally accurate differentiation, integration, interpolation, and reduced order modeling, among others, on a spectral element mesh. All the functionalities are provided in self-contained Python code and do not depend on the use of a particular solver. We believe that `PySEMTools` provides tools to researchers to accelerate scientific discovery and reduce the entry requirements for the use of advanced methods in computational fluid dynamics.

# Statement of need

The motion of fluids around objects is fundamental for many industrial and natural systems, from aerodynamics and cooling to the behavior of weather systems. Particularly relevant applications generally exist in the turbulent flow regime, where the fluid is subject to unsteady motions that are characterized by the interactions of eddies of multiple sizes and where increased levels of fluctuations and mixing exist.

A popular method to study these phenomena is to use computers to simulate their governing physics. The multi-scale characteristic of turbulence and the high Reynolds numbers (ratio between inertial and viscous forces) that are typically of interest require that the numerical grids are fine enough to capture the motion of the smallest eddies. While this has implied that the computational cost of simulations is high, the advent of graphics processing units (GPUs) has opened the doors to perform simulations that would not have been possible in the past. This increase in capability has made managing the data produced on a typical simulation campaign more challenging. Our work in `PySEMTools` aims to streamline the data management and post-processing of the results obtained from a particular numerical method often used to study turbulent flows while keeping high-order accuracy.

`PySEMTools` aims to help post-processing data from solvers that use the spectral element method (SEM) originally proposed by @PATERA1984, which is a high-order variant of the finite element method (FEM). In SEM, the computational domain is divided into a finite set of elements in which a Gauss-Lobatto-Legendre (GLL) grid of a given degree $N$ is embedded. Inside each element, the solution is expanded using polynomials of order $P = N - 1$, resulting in low dissipation and dispersive errors. 

Nek5000 [@nek5000-web-page], written in Fortran 77, is a successful implementation of SEM that has been used for several studies on the field, such as simulations of vascular flows by @fischer2006, turbulent pipe flow by @elkhoury2013, flow around wings by @mallor2024 and even nuclear applications as shown in the overview by @merzari2020. In general, the post-processing pipeline has been somewhat complicated, as when the data is needed in the SEM format, for example, to calculate derivatives of velocity fields, the solver itself has been used in a "post-processing" mode. This mode uses the solver and additional Fortran code that needs to be compiled to produce smaller files that can be used in Matlab or Python with e.g. PyMech by @pymech, to perform signal processing, create plots, etc. NekRS by @fischer2021nekrs,  a GPU version of Nek5000, and Neko [@jansson2024; @jansson2023], a modern Fortran implementation of SEM have followed the same approach, motivating the necessity of our PySEMTools for the future.

The motivation behind using the solvers themselves with the data in its raw format is understandable, as these large files need to be processed in parallel due to their sheer size. Still, we believe that the process has become very cumbersome as multiple code bases need to be maintained for post-processing the data. With `PySEMTools` we have brought a solution to this, as we have included all the functionalities that are typically needed from the solvers while ensuring that the codes perform efficiently in parallel while also taking advantage of the rich library ecosystem present in Python.

# Features 

`PySEMTools` relies heavily on MPI for Python by @mpi4py, given that it has been designed from the beginning to work on distributed settings. For computations, we rely on NumPy [@numpy]. It has been extensively tested on data produced by Nek5000 and Neko but, as mentioned before, the implemented methods and routines are consistent with any SEM-like data structure with hexahedral elements. Among its most relevant features are the following:

* **Parallel IO**: A set of routines to perform distributed IO on Nek5000/Neko field files and directly keep the data in memory on NumPy arrays or PyMech data objects.
* **Parallel data interfaces**: A set of objects that aim to facilitate the transfer of messages among processors. Done to ease the use of MPI functions for more inexperienced users.
* **Calculus**:  Objects to calculate the derivation and integration matrices based on the geometry, which allows to perform calculus operations on the spectral element mesh.
* **Mesh connectivity and partitioning**: Objects to determine the connectivity based on the geometry and mesh repartitioning tools for tasks such as global summation, among others.
* **Interpolation**: Routines to perform high-order interpolation from an SEM mesh into any arbitrary query point. A crucial functionality when performing post-processing.
* **Reduced-order modeling**: Objects to perform parallel and streaming proper orthogonal decomposition (POD).
* **Data compression/streaming**: Through the use of ADIOS2 [@adios2], a set of interfaces is available to perform data compression or to connect Python scripts to running simulations to perform in-situ data processing. 
* **Visualization**: Given that the data is available in Python, visualizations can be performed from readily available packages. 


We note that all of these functionalities are supported by examples in the software repository.


# Acknowledgements

This work is partially funded by the “Adaptive multi-tier intelligent data manager for Exascale (ADMIRE)” project, which is funded by the European Union's Horizon 2020 JTI-EuroHPC research and innovation program under grant Agreement number: 956748. Computations for testing were enabled by resources provided by the National Academic Infrastructure for Super­computing in Sweden (NAISS), partially funded by the Swedish Research Council through grant agreement no. 2018-05973.

# References