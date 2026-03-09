include(ExternalProject)

set(MPI4PY_VERSION "4.1.1" CACHE STRING "MPI4PY version to install")

message(STATUS "MPI4PY version: ${MPI4PY_VERSION}")

# Python handling for mpi4py
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


set(_mpi4py_depends "")

set(_mpi4py_env
MPICC=${CMAKE_C_COMPILER}
)

ExternalProject_Add(mpi4py_ext
    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND
    ${CMAKE_COMMAND} -E env
        ${_mpi4py_env}
        ${_python_exe} -m pip install --no-cache-dir --no-binary=mpi4py mpi4py==${MPI4PY_VERSION}
    DEPENDS ${_mpi4py_depends}
)