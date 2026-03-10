include(ExternalProject)

set(ADIOS2_VERSION "2.10.1" CACHE STRING "ADIOS2 version to install")
set(ADIOS2_GIT_TAG "" CACHE STRING "Explicit ADIOS2 git tag override")

set(BZIP2_VERSION "1.0.8" CACHE STRING "BZip2 version to install")
set(BZIP2_URL
    "https://www.sourceware.org/pub/bzip2/bzip2-${BZIP2_VERSION}.tar.gz"
    CACHE STRING "BZip2 source archive URL")

if(ADIOS2_GIT_TAG STREQUAL "")
  set(_adios2_git_tag "v${ADIOS2_VERSION}")
else()
  set(_adios2_git_tag "${ADIOS2_GIT_TAG}")
endif()

message(STATUS "ADIOS2 version: ${ADIOS2_VERSION}")
message(STATUS "ADIOS2 git tag: ${_adios2_git_tag}")
message(STATUS "BZip2 version: ${BZIP2_VERSION}")
message(STATUS "BZip2 URL: ${BZIP2_URL}")
set(BZIP2_INSTALL_PREFIX "${THIRD_PARTY_INSTALL_PREFIX}")
set(BZIP2_INCLUDE_DIR "${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_INCLUDEDIR}")
set(BZIP2_LIBRARY_RELEASE "${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_LIBDIR}/libbz2.so")
set(BZIP2_LIBRARY_DEBUG   "${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_LIBDIR}/libbz2.so")

# Python handling
set(_python_args -DADIOS2_USE_Python=OFF)

if(ADIOS2_USE_PYTHON)
  if(PYTHON_EXECUTABLE)
    if(NOT EXISTS "${PYTHON_EXECUTABLE}")
      message(FATAL_ERROR "PYTHON_EXECUTABLE does not exist: ${PYTHON_EXECUTABLE}")
    endif()

    get_filename_component(_python_root_from_exe "${PYTHON_EXECUTABLE}" DIRECTORY)
    get_filename_component(_python_prefix_from_exe "${_python_root_from_exe}/.." ABSOLUTE)

    set(_python_args
      -DADIOS2_USE_Python=ON
      -DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPython_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPython3_FIND_STRATEGY=LOCATION
      -DPython_FIND_STRATEGY=LOCATION
      -DPython3_ROOT_DIR=${_python_prefix_from_exe}
    )

    if(Python3_ROOT_DIR)
      list(APPEND _python_args -DPython3_ROOT_DIR=${Python3_ROOT_DIR})
    endif()
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)

    set(_python_args
      -DADIOS2_USE_Python=ON
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
      -DPython_EXECUTABLE=${Python3_EXECUTABLE}
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -DPython3_FIND_STRATEGY=LOCATION
      -DPython_FIND_STRATEGY=LOCATION
    )

    if(Python3_ROOT_DIR)
      list(APPEND _python_args -DPython3_ROOT_DIR=${Python3_ROOT_DIR})
    endif()
  endif()
endif()

if(ADIOS2_USE_FORTRAN)
  set(_fortran_arg -DADIOS2_USE_Fortran=ON)
else()
  set(_fortran_arg -DADIOS2_USE_Fortran=OFF)
endif()

if(ADIOS2_USE_BZIP2)
  set(_bzip2_args
    -DADIOS2_USE_BZip2=ON
    -DBZIP2_INCLUDE_DIR=${BZIP2_INCLUDE_DIR}
    -DBZIP2_LIBRARY_RELEASE=${BZIP2_LIBRARY_RELEASE}
    -DBZIP2_LIBRARY_DEBUG=${BZIP2_LIBRARY_DEBUG}
  )

  ExternalProject_Add(bzip2_ext
    URL ${BZIP2_URL}
    DOWNLOAD_DIR ${THIRD_PARTY_DOWNLOAD_DIR}
    SOURCE_DIR ${THIRD_PARTY_BUILD_DIR}/src/bzip2-${BZIP2_VERSION}
    BUILD_IN_SOURCE 1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CMAKE_MAKE_PROGRAM} CC=${CMAKE_C_COMPILER} -f Makefile-libbz2_so
    INSTALL_COMMAND
      ${CMAKE_COMMAND} -E make_directory ${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_INCLUDEDIR}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_LIBDIR}
      COMMAND ${CMAKE_COMMAND} -E copy libbz2.so.${BZIP2_VERSION} ${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_LIBDIR}/libbz2.so.${BZIP2_VERSION}
      COMMAND ${CMAKE_COMMAND} -E copy libbz2.so.${BZIP2_VERSION} ${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_LIBDIR}/libbz2.so
      COMMAND ${CMAKE_COMMAND} -E copy bzlib.h ${BZIP2_INSTALL_PREFIX}/${THIRD_PARTY_INCLUDEDIR}/bzlib.h
  )

  set(_adios2_depends bzip2_ext)
else()
  set(_bzip2_args -DADIOS2_USE_BZip2=OFF)
  set(_adios2_depends "")
endif()

if(INSTALL_MPI4PY AND ADIOS2_USE_PYTHON)
  if(NOT TARGET mpi4py_ext)
    message(FATAL_ERROR "INSTALL_MPI4PY is ON, but mpi4py_ext target was not created")
  endif()
  list(APPEND _adios2_depends mpi4py_ext)
endif()

ExternalProject_Add(adios2_ext
  GIT_REPOSITORY https://github.com/ornladios/ADIOS2.git
  GIT_TAG ${_adios2_git_tag}
  GIT_SHALLOW 1
  DOWNLOAD_DIR ${THIRD_PARTY_DOWNLOAD_DIR}
  SOURCE_DIR ${THIRD_PARTY_BUILD_DIR}/src/ADIOS2
  BINARY_DIR ${THIRD_PARTY_BUILD_DIR}/adios2-build
  INSTALL_DIR ${THIRD_PARTY_INSTALL_PREFIX}

  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PREFIX}
    -DCMAKE_INSTALL_BINDIR=${THIRD_PARTY_BINDIR}
    -DCMAKE_INSTALL_INCLUDEDIR=${THIRD_PARTY_INCLUDEDIR}
    -DCMAKE_INSTALL_LIBDIR=${THIRD_PARTY_LIBDIR}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DADIOS2_USE_ZeroMQ=OFF
    -DCMAKE_INSTALL_PYTHONDIR=${THIRD_PARTY_PYTHONDIR}
    ${_fortran_arg}
    ${_python_args}
    ${_bzip2_args}

  DEPENDS ${_adios2_depends}
)