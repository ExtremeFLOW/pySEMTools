include(ExternalProject)

set(HDF5_VERSION "1.14.6" CACHE STRING "HDF5 version to install")
set(HDF5_GIT_TAG "" CACHE STRING "Explicit HDF5 git tag override")

if(HDF5_GIT_TAG STREQUAL "")
  set(_hdf5_git_tag "hdf5_${HDF5_VERSION}")
else()
  set(_hdf5_git_tag "${HDF5_GIT_TAG}")
endif()

message(STATUS "HDF5 version: ${HDF5_VERSION}")
message(STATUS "Resolved HDF5 git tag: ${_hdf5_git_tag}")

# Python handling for h5py
set(_python_exe "")
if(PYTHON_EXECUTABLE)
  if(NOT EXISTS "${PYTHON_EXECUTABLE}")
    message(FATAL_ERROR "PYTHON_EXECUTABLE does not exist: ${PYTHON_EXECUTABLE}")
  endif()
  set(_python_exe "${PYTHON_EXECUTABLE}")
else()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_python_exe "${Python3_EXECUTABLE}")
endif()

set(_hdf5_depends "")
if(INSTALL_H5PY)
  set(_h5py_depends h5py_ext)
endif()

if(INSTALL_MPI4PY AND INSTALL_H5PY)
  if(NOT TARGET mpi4py_ext)
    message(FATAL_ERROR "INSTALL_MPI4PY is ON, but mpi4py_ext target was not created")
  endif()
  list(APPEND _hdf5_depends mpi4py_ext)
  list(APPEND _h5py_depends mpi4py_ext)

endif()

if(INSTALL_HDF5 OR INSTALL_H5PY)
  ExternalProject_Add(hdf5_ext
  GIT_REPOSITORY https://github.com/HDFGroup/hdf5.git
  GIT_TAG ${_hdf5_git_tag}
  GIT_SHALLOW 1
  DOWNLOAD_DIR ${THIRD_PARTY_DOWNLOAD_DIR}
  SOURCE_DIR ${THIRD_PARTY_BUILD_DIR}/src/hdf5
  BINARY_DIR ${THIRD_PARTY_BUILD_DIR}/hdf5-build
  INSTALL_DIR ${THIRD_PARTY_INSTALL_PREFIX}

  CMAKE_ARGS
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PREFIX}
    -DHDF5_ENABLE_PARALLEL=ON

  DEPENDS ${_hdf5_depends} 
  )
endif()

if(INSTALL_H5PY)
  set(H5PY_USE_MPI ON)
  message(STATUS "H5PY use MPI: ${H5PY_USE_MPI}")
  set(_h5py_env
    HDF5_MPI=ON
    CC=${CMAKE_C_COMPILER}
    HDF5_DIR=${THIRD_PARTY_INSTALL_PREFIX}
  )

    ExternalProject_Add(h5py_ext
      DOWNLOAD_COMMAND ""
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND
        ${CMAKE_COMMAND} -E env
          ${_h5py_env}
          ${_python_exe} -m pip install --no-cache-dir git+https://github.com/h5py/h5py.git
      DEPENDS ${_h5py_depends}
    )
endif()