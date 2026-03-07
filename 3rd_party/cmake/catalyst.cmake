include(ExternalProject)

set(CATALYST_VERSION "2.0.0" CACHE STRING "Catalyst version to install")
set(CATALYST_GIT_TAG "" CACHE STRING "Explicit Catalyst git tag override")

if(CATALYST_GIT_TAG STREQUAL "")
  set(_catalyst_git_tag "v${CATALYST_VERSION}")
else()
  set(_catalyst_git_tag "${CATALYST_GIT_TAG}")
endif()

message(STATUS "Catalyst version: ${CATALYST_VERSION}")
message(STATUS "Resolved Catalyst git tag: ${_catalyst_git_tag}")

set(_catalyst_args
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
  -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PREFIX}
  -DCMAKE_INSTALL_BINDIR=${THIRD_PARTY_BINDIR}
  -DCMAKE_INSTALL_INCLUDEDIR=${THIRD_PARTY_INCLUDEDIR}
  -DCMAKE_INSTALL_LIBDIR=${THIRD_PARTY_LIBDIR}
  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
  -DCATALYST_WRAP_PYTHON=${CATALYST_WRAP_PYTHON}
)

if(CATALYST_WRAP_PYTHON)
  if(PYTHON_EXECUTABLE)
    if(NOT EXISTS "${PYTHON_EXECUTABLE}")
      message(FATAL_ERROR "PYTHON_EXECUTABLE does not exist: ${PYTHON_EXECUTABLE}")
    endif()
    list(APPEND _catalyst_args
      -DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPython_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPython3_FIND_STRATEGY=LOCATION
      -DPython_FIND_STRATEGY=LOCATION
    )
    if(Python3_ROOT_DIR)
      list(APPEND _catalyst_args
        -DPython3_ROOT_DIR=${Python3_ROOT_DIR}
      )
    endif()
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    list(APPEND _catalyst_args
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
      -DPython_EXECUTABLE=${Python3_EXECUTABLE}
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -DPython3_FIND_STRATEGY=LOCATION
      -DPython_FIND_STRATEGY=LOCATION
    )
  endif()
endif()

if(CATALYST_WRAP_FORTRAN)
  if(NOT CMAKE_Fortran_COMPILER)
    message(FATAL_ERROR "CATALYST_WRAP_FORTRAN=ON requires CMAKE_Fortran_COMPILER to be set")
  endif()
  list(APPEND _catalyst_args
    -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
  )
endif()

set(_catalyst_depends "")

ExternalProject_Add(catalyst_ext
  GIT_REPOSITORY https://gitlab.kitware.com/paraview/catalyst.git
  GIT_TAG ${_catalyst_git_tag}
  GIT_SHALLOW 1
  DOWNLOAD_DIR ${THIRD_PARTY_DOWNLOAD_DIR}
  SOURCE_DIR ${THIRD_PARTY_BUILD_DIR}/src/catalyst
  BINARY_DIR ${THIRD_PARTY_BUILD_DIR}/catalyst-build
  INSTALL_DIR ${THIRD_PARTY_INSTALL_PREFIX}
  CMAKE_ARGS ${_catalyst_args}
  DEPENDS ${_catalyst_depends}
)