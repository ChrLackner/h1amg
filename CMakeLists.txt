cmake_minimum_required(VERSION 3.1)
project(h1amg CXX)

set(NETGEN_BASE_DIR $ENV{NETGENDIR}/.. CACHE PATH "Netgen install dir")

find_package(NGSolve CONFIG REQUIRED
  HINTS ${NETGEN_BASE_DIR} /opt/netgen/ /Applications/Netgen.app/Contents/Resources/CMake C:/netgen)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif(NOT CMAKE_BUILD_TYPE)

if(USE_CCACHE)
  find_program(CCACHE_FOUND ccache)
  if(CCACHE_FOUND)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif(CCACHE_FOUND)
endif(USE_CCACHE)


# check if CMAKE_INSTALL_PREFIX is set by user, if not install in NGSolve python dir
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX ${NGSOLVE_INSTALL_DIR}/${NGSOLVE_INSTALL_DIR_PYTHON} CACHE PATH "Install dir" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)


add_subdirectory(src)

option(BUILD_TESTING "" OFF)
include(CTest)
if(BUILD_TESTING)
  include(cmake_modules/PythonLibTest.cmake)
  add_subdirectory(tests)
endif(BUILD_TESTING)
