cmake_minimum_required(VERSION 3.20)
project(pysemtools_third_party NONE)

include(ExternalProject)

# ===============================================================
# Adios2
# ===============================================================
option(INSTALL_ADIOS2 "Install ADIOS2" OFF)
option(ADIOS2_USE_PYTHON "Build ADIOS2 with Python support" ON)
option(ADIOS2_USE_FORTRAN "Build ADIOS2 with Fortran support" OFF)
option(ADIOS2_USE_BZIP2 "Build ADIOS2 with BZip2 support" ON)

# ===============================================================
# HDF5
# ===============================================================
option(INSTALL_HDF5 "Install HDF5" OFF)

# ===============================================================
# h5py
# ===============================================================
option(INSTALL_H5PY "Install h5py against the built HDF5" OFF)

# ===============================================================
# Catalyst
# ===============================================================
option(INSTALL_CATALYST "Install Catalyst" OFF)
option(CATALYST_WRAP_PYTHON "Build Catalyst Python wrappers" ON)

# ===============================================================
# Paraview
# ===============================================================
option(INSTALL_PARAVIEW "Install ParaView" OFF)

set(PARAVIEW_VARIANT "SERVER" CACHE STRING "ParaView build variant: SERVER or DESKTOP")
set_property(CACHE PARAVIEW_VARIANT PROPERTY STRINGS SERVER DESKTOP)

option(PARAVIEW_USE_MPI "Build ParaView with MPI support" ON)
option(PARAVIEW_USE_PYTHON "Build ParaView with Python support" ON)
option(PARAVIEW_ENABLE_CATALYST "Build ParaView Catalyst implementation" ON)

set(PARAVIEW_SMP_IMPLEMENTATION_TYPE "TBB" CACHE STRING "VTK SMP backend for ParaView")
set_property(CACHE PARAVIEW_SMP_IMPLEMENTATION_TYPE PROPERTY STRINGS
  Sequential STDThread TBB OpenMP)

#if(PARAVIEW_ENABLE_CATALYST)
#   set(INSTALL_CATALYST ON)
#endif()

# ===============================================================
# General folder structure
# ===============================================================

set(THIRD_PARTY_INSTALL_PREFIX
    "${CMAKE_SOURCE_DIR}/install"
    CACHE PATH "Install prefix for third-party libraries")

set(THIRD_PARTY_DOWNLOAD_DIR
    "${CMAKE_SOURCE_DIR}/downloads"
    CACHE PATH "Directory for downloaded source archives and clones")

set(THIRD_PARTY_BUILD_DIR
    "${CMAKE_SOURCE_DIR}/build"
    CACHE PATH "Directory for third-party build trees")

set(THIRD_PARTY_BINDIR "bin" CACHE STRING "Binary install dir")
set(THIRD_PARTY_INCLUDEDIR "include" CACHE STRING "Include install dir")
set(THIRD_PARTY_LIBDIR "lib" CACHE STRING "Library install dir")

set(PYTHON_EXECUTABLE ""
    CACHE FILEPATH "Exact Python interpreter to use for third-party builds")

# Optional: also allow passing the root of that Python install
set(Python3_ROOT_DIR ""
    CACHE PATH "Optional root dir for the selected Python installation")

file(MAKE_DIRECTORY "${THIRD_PARTY_INSTALL_PREFIX}")
file(MAKE_DIRECTORY "${THIRD_PARTY_DOWNLOAD_DIR}")
file(MAKE_DIRECTORY "${THIRD_PARTY_BUILD_DIR}")
file(MAKE_DIRECTORY "${THIRD_PARTY_INSTALL_PREFIX}/${THIRD_PARTY_BINDIR}")
file(MAKE_DIRECTORY "${THIRD_PARTY_INSTALL_PREFIX}/${THIRD_PARTY_INCLUDEDIR}")
file(MAKE_DIRECTORY "${THIRD_PARTY_INSTALL_PREFIX}/${THIRD_PARTY_LIBDIR}")

message(STATUS "Third-party install prefix: ${THIRD_PARTY_INSTALL_PREFIX}")
message(STATUS "Third-party download dir: ${THIRD_PARTY_DOWNLOAD_DIR}")
message(STATUS "Third-party build dir: ${THIRD_PARTY_BUILD_DIR}")
message(STATUS "Third-party bin dir: ${THIRD_PARTY_BINDIR}")
message(STATUS "Third-party include dir: ${THIRD_PARTY_INCLUDEDIR}")
message(STATUS "Third-party lib dir: ${THIRD_PARTY_LIBDIR}")

if(PYTHON_EXECUTABLE)
  message(STATUS "Requested Python executable: ${PYTHON_EXECUTABLE}")
endif()

if(Python3_ROOT_DIR)
  message(STATUS "Requested Python root dir: ${Python3_ROOT_DIR}")
endif()

# ===============================================================
# Add the libraries installation instructions to the build
# ===============================================================

if(INSTALL_ADIOS2)
  include(${CMAKE_SOURCE_DIR}/cmake/adios2.cmake)
endif()

if(INSTALL_HDF5 OR INSTALL_H5PY)
  include(${CMAKE_SOURCE_DIR}/cmake/hdf5.cmake)
endif()
  
if(INSTALL_CATALYST)
  include(${CMAKE_SOURCE_DIR}/cmake/catalyst.cmake)
endif()

if(INSTALL_PARAVIEW)
  include(${CMAKE_SOURCE_DIR}/cmake/paraview.cmake)
endif()