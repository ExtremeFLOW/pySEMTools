include(ExternalProject)

set(PARAVIEW_VERSION "6.0.1" CACHE STRING "ParaView version to install")
set(PARAVIEW_GIT_TAG "" CACHE STRING "Explicit ParaView git tag override")

if(PARAVIEW_GIT_TAG STREQUAL "")
  set(_paraview_git_tag "v${PARAVIEW_VERSION}")
else()
  set(_paraview_git_tag "${PARAVIEW_GIT_TAG}")
endif()

if(PARAVIEW_VARIANT STREQUAL "DESKTOP")
  set(_paraview_use_qt ON)
elseif(PARAVIEW_VARIANT STREQUAL "SERVER")
  set(_paraview_use_qt OFF)
else()
  message(FATAL_ERROR
    "Invalid PARAVIEW_VARIANT='${PARAVIEW_VARIANT}'. Use SERVER or DESKTOP.")
endif()

message(STATUS "ParaView version: ${PARAVIEW_VERSION}")
message(STATUS "Resolved ParaView git tag: ${_paraview_git_tag}")
message(STATUS "ParaView variant: ${PARAVIEW_VARIANT}")
message(STATUS "ParaView Qt GUI enabled: ${_paraview_use_qt}")

set(_paraview_args
  -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY_INSTALL_PREFIX}
  -DCMAKE_INSTALL_BINDIR=${THIRD_PARTY_BINDIR}
  -DCMAKE_INSTALL_INCLUDEDIR=${THIRD_PARTY_INCLUDEDIR}
  -DCMAKE_INSTALL_LIBDIR=${THIRD_PARTY_LIBDIR}

  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
  -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}

  -DPARAVIEW_USE_PYTHON=${PARAVIEW_USE_PYTHON}
  -DPARAVIEW_USE_MPI=${PARAVIEW_USE_MPI}  
  -DVTK_SMP_IMPLEMENTATION_TYPE=${PARAVIEW_SMP_IMPLEMENTATION_TYPE}
  -DCMAKE_BUILD_TYPE=Release
  -DPARAVIEW_ENABLE_CATALYST=${PARAVIEW_ENABLE_CATALYST} 
  -DPARAVIEW_USE_QT=${_paraview_use_qt}

  # Help ParaView find dependencies installed in the same prefix.
  -DCMAKE_PREFIX_PATH=${THIRD_PARTY_INSTALL_PREFIX}

)

# Python forwarding
if(PARAVIEW_USE_PYTHON)
  if(PYTHON_EXECUTABLE)
    if(NOT EXISTS "${PYTHON_EXECUTABLE}")
      message(FATAL_ERROR "PYTHON_EXECUTABLE does not exist: ${PYTHON_EXECUTABLE}")
    endif()

    list(APPEND _paraview_args
      -DPython3_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPython_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE}
      -DPython3_FIND_STRATEGY=LOCATION
      -DPython_FIND_STRATEGY=LOCATION
    )

    if(Python3_ROOT_DIR)
      list(APPEND _paraview_args
        -DPython3_ROOT_DIR=${Python3_ROOT_DIR}
      )
    endif()
  else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    list(APPEND _paraview_args
      -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
      -DPython_EXECUTABLE=${Python3_EXECUTABLE}
      -DPYTHON_EXECUTABLE=${Python3_EXECUTABLE}
      -DPython3_FIND_STRATEGY=LOCATION
      -DPython_FIND_STRATEGY=LOCATION
    )
  endif()
endif()

set(_paraview_depends "")

# Auto-wire Catalyst from the same superbuild install tree
if(PARAVIEW_ENABLE_CATALYST)
  set(_auto_catalyst_dir
    "${THIRD_PARTY_INSTALL_PREFIX}/${THIRD_PARTY_LIBDIR}/cmake/catalyst-2.0")
  list(APPEND _paraview_args
    -Dcatalyst_DIR=${_auto_catalyst_dir}
  )

  message(STATUS "ParaView will use catalyst_DIR=${_auto_catalyst_dir}")

  if(TARGET catalyst_ext)
    list(APPEND _paraview_depends catalyst_ext)
  endif()
endif()

if(INSTALL_MPI4PY AND PARAVIEW_USE_PYTHON)
  if(NOT TARGET mpi4py_ext)
    message(FATAL_ERROR "INSTALL_MPI4PY is ON, but mpi4py_ext target was not created")
  endif()
  list(APPEND _paraview_depends mpi4py_ext)
endif()

ExternalProject_Add(paraview_ext
  GIT_REPOSITORY https://gitlab.kitware.com/paraview/paraview.git
  GIT_TAG ${_paraview_git_tag}
  GIT_SHALLOW 1
  GIT_SUBMODULES_RECURSE 1

  DOWNLOAD_DIR ${THIRD_PARTY_DOWNLOAD_DIR}
  SOURCE_DIR ${THIRD_PARTY_BUILD_DIR}/src/paraview
  BINARY_DIR ${THIRD_PARTY_BUILD_DIR}/paraview-build
  INSTALL_DIR ${THIRD_PARTY_INSTALL_PREFIX}

  CMAKE_ARGS ${_paraview_args}

  DEPENDS ${_paraview_depends}
)