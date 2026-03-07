cmake_minimum_required(VERSION 3.20)
project(pysemtools_third_party NONE)

include(ExternalProject)

# ===============================================================
# Adios2
# ===============================================================
option(INSTALL_ADIOS2 "Install ADIOS2" ON)
option(ADIOS2_USE_PYTHON "Build ADIOS2 with Python support" ON)
option(ADIOS2_USE_FORTRAN "Build ADIOS2 with Fortran support" OFF)
option(ADIOS2_USE_BZIP2 "Build ADIOS2 with BZip2 support" ON)


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

if(INSTALL_ADIOS2)
  include(${CMAKE_SOURCE_DIR}/cmake/adios2.cmake)
endif()