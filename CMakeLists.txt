# ========================================================================
#     This file is part of ParMooN.
# 
#     ParMooN (cmg.cds.iisc.ac.in/parmoon) is free software:
#     you can redistribute it and/or modify it under the terms of the 
#     GNU General Public License as published by the Free Software 
#     Foundation, either version 3 of the License, or (at your option)
#     any later version. 
#
#     ParMooN is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
# 
#     You should have received a copy of the GNU General Public License
#     along with ParMooN.  If not, see <http://www.gnu.org/licenses/>.
# =======================================================================
# PURPOSE :  configuration file for the ParMooN Version 1.1
# AUTHOR  : Sashikumaar Ganesan, CDS, IISc Bangalore, India
# DATE    : 05 June 2015
# ========================================================================

# normally no need to alter any thing in this file
# please DO NOT modify unless if it is necessary
# beginners are advised NOT to alter anything the below contents

cmake_minimum_required (VERSION 2.8)

project (ParMooN LANGUAGES C CXX)

include(${CMAKE_SOURCE_DIR}/UserConfig.cmake)

set (ParMooN_VERSION_MAJOR 1)
set (ParMooN_VERSION_MINOR 0)

SET(BUILD_SHARED_LIBS ON)

# set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/lib)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/intel/oneapi/mkl/latest/lib/intel64)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} $ENV{HOME}/intel/oneapi/mkl/latest/lib/intel64)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} $ENV{HOME}/lib)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} $ENV{HOME}/software/lib)
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} $ENV{HOME}/Software/lib)

set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} /opt/local/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} /opt/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} /opt/intel/oneapi/mkl/latest/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} $ENV{HOME}/intel/oneapi/mkl/latest/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} $ENV{HOME}/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} $ENV{HOME}/software/include)
set(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH} $ENV{HOME}/Software/include)

# set(ENV{PKG_CONFIG_PATH} $ENV{PKG_CONFIG_PATH}:/opt/local/lib/pkgconfig )
set(PARMOON_EXTLIB_PATH ${PROJECT_SOURCE_DIR}/EXT_LIB)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

if("${AParMooN_PARALLEL_TYPE}" STREQUAL "SCUDA")
  project (ParMooN LANGUAGES C CXX CUDA)
  set (CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
  include_directories(/usr/local/cuda-11.7/include)
  set(PARMOON_CXX_DEF "${PARMOON_CXX_DEF} -DSCUDA")
endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "SCUDA")


if(NOT "${AParMooN_ARCH}" STREQUAL "CRAY64")
# search the blas in the system, if available it will be default
message("==================Searching BLAS in the System - begin===================================================")
if(AParMooN_USE_SYSTEM_MKLBLAS)
    message("Searching MKL BLAS ...")
    find_package(MKLBLAS REQUIRED)
    ## Thivin -- Added Preprocessor Definitions if MKL blas is found. 
    ## Else, the INtel routines such as Pardiso and mkl based routines will not be used in the code.
    set(PARMOON_CXX_DEF "${PARMOON_CXX_DEF} -DINTELMKLBLAS")
    include_directories(${BLAS_INCLUDE_DIRS})
    if(CMAKE_VERBOSE_MAKEFILE)
      if(BLAS_FOUND)
        message("Available system BLAS_LIB =${BLAS_LIBRARIES}")
        ## Set the definition that the INTEL MKL BLAS has been found
      elseif(NOT BLAS_FOUND)
        message("MKL BLAS_NOT_FOUND in your system, searching ...")
      endif(BLAS_FOUND)
    endif(CMAKE_VERBOSE_MAKEFILE)  
   

endif(AParMooN_USE_SYSTEM_MKLBLAS)

if(NOT BLAS_FOUND)
# if(AParMooN_USE_SYSTEM_BLAS)
  find_package(BLAS)
  include_directories(${BLAS_INCLUDE_DIRS})
  if(CMAKE_VERBOSE_MAKEFILE)
    if(BLAS_FOUND)
      message("Available system BLAS_LIB =${BLAS_LIBRARIES}")
    elseif(NOT BLAS_FOUND)
       message("BLAS_NOT_FOUND in your system, BLAS must be linked before compiling")
    endif()
  endif()
# endif(AParMooN_USE_SYSTEM_BLAS)
endif(NOT BLAS_FOUND)
message("==================Searching BLAS in the System - end=====================================================")



if("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")


#   message("==================Searching MPI in the System - begin===================================================")
find_package(MPIBLACS)

   if("${AParMooN_MPI_IMPLEMENTATION}" STREQUAL "MPICH")
       set(MPI_LIBS -lmpich -lmpichf90 -lmpichcxx -lgfortran )
   elseif("${AParMooN_MPI_IMPLEMENTATION}" STREQUAL "OPENMPI")
      set(MPI_LIBS -lmpi -lmpifort -lmpicxx)
   elseif("${AParMooN_MPI_IMPLEMENTATION}" STREQUAL "INTELMPI")
      set(MPI_LIBS -lifcore)    
       message("Available MPI_LIBS =${MPI_LIBS}")
   elseif("${AParMooN_MPI_IMPLEMENTATION}" STREQUAL "MACMPI")
     set(MPI_LIBS -lmpich -lmpichf90 -lmpichcxx)             
   endif("${AParMooN_MPI_IMPLEMENTATION}" STREQUAL "MPICH")   

message(" MPI_LIBS " = ${MPI_LIBS})

 # search the MUMPS in the system, if available it wil be default
 message("==================Searching MUMPS in the System - begin=====================================================")
 find_package(MUMPS)
 include_directories(${MUMPS_INCLUDE_DIRS})
 if(CMAKE_VERBOSE_MAKEFILE)
  if(MUMPS_FOUND)
    message("Available system MUMPS_LIB =${MUMPS_LIBRARIES}")
  elseif(NOT MUMPS_FOUND)
    message(FATAL_ERROR "MUMPS_NOT_FOUND in your system or in the ParMooN")
  endif()
 endif()
 message("==================Searching MUMPS in the System - end=======================================================")

endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")

endif(NOT "${AParMooN_ARCH}" STREQUAL "CRAY64")

# search the UMFPACK in the system, if available it wil be default
message("==================Searching UMFPACK in the System - begin===================================================")
find_package(UMFPACK)
include_directories(${UMFPACK_INCLUDE_DIRS})
if(CMAKE_VERBOSE_MAKEFILE)
  if(UMFPACK_FOUND)
    message("Available system UMFPACK_LIB =${UMFPACK_LIBRARIES}")
  elseif(NOT UMFPACK_FOUND)
    message(FATAL_ERROR "UMFPACK_NOT_FOUND in your system or in the ParMooN")
  endif(UMFPACK_FOUND)
endif(CMAKE_VERBOSE_MAKEFILE)
message("==================Searching UMFPACK in the System - end=====================================================")

# search the GRIDGEN in the system, if available it wil be default
message("==================Searching GRIDGEN in the System - begin==================================================")
find_package(GRIDGEN)
include_directories(${GRIDGEN_INCLUDE_DIRS})
if(CMAKE_VERBOSE_MAKEFILE)
  if(GRIDGEN_FOUND)
    message("Available system GRIDGEN_LIB =${GRIDGEN_LIBRARIES}")
  elseif(NOT GRIDGEN_FOUND)
    message(FATAL_ERROR "GRIDGEN_NOT_FOUND in your system or in the ParMooN")
  endif()
endif()
message("==================Searching GRIDGEN in the System - end====================================================")

# search the TETGEN in the system, if available it wil be default
if("${AParMooN_GEO}" STREQUAL "3D")
  message("==================Searching TETGEN in the System - begin==================================================")
  find_package(TETGEN)
  include_directories(${TETGEN_INCLUDE_DIRS})
  if(CMAKE_VERBOSE_MAKEFILE)
    if(TETGEN_FOUND)
      message("Available system TETGEN_LIB =${TETGEN_LIBRARIES}")
    elseif(NOT TETGEN_FOUND)
      message(FATAL_ERROR "TETGEN_NOT_FOUND in your system or in the ParMooN")
    endif()
  endif()
message("==================Searching TETGEN in the System - end======================================================")
endif("${AParMooN_GEO}" STREQUAL "3D")

if (NOT CONFIGURED_ONCE)
    set(CMAKE_CXX_FLAGS "${warnings}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
    set(CMAKE_C_FLAGS   "${warnings}"
        CACHE STRING "Flags used by the compiler during all build types." FORCE)
endif()

# search the TECPLOT in the system, if available it wil be default
message("==================Searching TECPLOT in the System - begin==================================================")
find_package(TECPLOT)
include_directories(${TECPLOT_INCLUDE_DIRS})
if(CMAKE_VERBOSE_MAKEFILE)
  if(TECPLOT_FOUND)
    message("Available system TECPLOT_LIB =${TECPLOT_LIBRARIES}")
  elseif(NOT TECPLOT_FOUND)
    message(FATAL_ERROR "TECPLOT_NOT_FOUND in your system or in the ParMooN")
  endif()
endif()
message("==================Searching TECPLOT in the System - end====================================================")

message("==================Add mlpack from EXT_LIB - begin====================================================")

if(AParMooN_USE_ANN)

  list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR/EXT_LIB/Armadillo}")
  find_package(ARMADILLO REQUIRED)
  include_directories(${ARMADILLO_INCLUDE_DIRS})
    if(ARMADILLO_FOUND)
        message("ARMADILLO_LIB (read from EXT_LIB/Armadillo) = ${ARMADILLO_LIBRARIES}")
        set(CMAKE_INSTALL_RPATH "${PARMOON_EXTLIB_PATH}/Armadillo/lib/")
        set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    elseif(NOT ARMADILLO_FOUND)
      message(FATAL_ERROR "ARMADILLO_NOT_FOUND in your system or in the ParMooN")
    endif()
  find_package(MLPACK REQUIRED)
  include_directories(${MLPACK_INCLUDE_DIRS})
    if(MLPACK_FOUND)
        message("MLPACK_LIB (read from EXT_LIB/mlpack) = ${MLPACK_LIBRARIES}")
    elseif(NOT MLPACK_FOUND)
      message(FATAL_ERROR "MLPACK_NOT_FOUND in your system or in the ParMooN")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR/EXT_LIB/Ensmallen}")
    find_package(ENSMALLEN REQUIRED)
    include_directories(${ENSMALLEN_INCLUDE_DIRS})
    # if(ENSMALLEN_FOUND)
    #    message("ENSMALLEN_LIB (read from EXT_LIB/Ensmallen) = ${ENSMALLEN_LIBRARIES}")
    #  elseif(NOT ENSMALLEN_FOUND)
    #    message(FATAL_ERROR "ENSMALLEN_NOT_FOUND in your system or in the ParMooN")
    #  endif() 

endif(AParMooN_USE_ANN)
message("==================Add mlpack from EXT_LIB - end====================================================")


# select the compiler type,
# CMAKE_BUILD_TYPE [DEBUG|RELEASE|RELWITHDEBINFO|MINSIZEREL]
set(CMAKE_BUILD_TYPE DEBUG CACHE STRING "options")
set(PARMOON_INCLUDE )

include (CMakeForceCompiler)
  set(CMAKE_SYSTEM_NAME ${AParMooN_ARCH})
  
if("${AParMooN_ARCH}" STREQUAL "MAC64")

  if("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")
    find_path(MPI_CPATH   mpicc-mpich-mp)
    set(CMAKE_C_COMPILER ${MPI_CPATH}/mpicc-mpich-mp)
    find_path(MPI_CXXPATH   mpicxx-mpich-mp)  
    set(CMAKE_CXX_COMPILER ${MPI_CXXPATH}/mpicxx-mpich-mp)      
  elseif("${AParMooN_PARALLEL_TYPE}" STREQUAL "SEQUENTIAL"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "OMPONLY")
    find_path(CPATH  clang)
    set(CMAKE_C_COMPILER ${CPATH}/clang  )
    find_path(CXXPATH clang++)  
    set(CMAKE_CXX_COMPILER ${CXXPATH}/clang++)  
  endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")
 
elseif("${AParMooN_ARCH}" STREQUAL "LINUX64")

  if("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")
    find_path(MPI_CPATH   mpicc)
    set(CMAKE_C_COMPILER ${MPI_CPATH}/mpicc)
    find_path(MPI_CXXPATH   mpicxx)  
    set(CMAKE_CXX_COMPILER ${MPI_CXXPATH}/mpicxx)      
  elseif("${AParMooN_PARALLEL_TYPE}" STREQUAL "SEQUENTIAL"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "OMPONLY")
    find_path(CPATH  gcc)
    set(CMAKE_C_COMPILER ${CPATH}/gcc)
    find_path(CXXPATH g++)  
    set(CMAKE_CXX_COMPILER ${CXXPATH}/g++)  
  endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")
  
elseif("${AParMooN_ARCH}" STREQUAL "INTEL64")

  if("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")
    find_path(MPI_CPATH   mpiicc)
    set(CMAKE_C_COMPILER ${MPI_CPATH}/mpiicc)
    find_path(MPI_CXXPATH   mpiicpc)  
    set(CMAKE_CXX_COMPILER ${MPI_CXXPATH}/mpiicpc)      
  elseif("${AParMooN_PARALLEL_TYPE}" STREQUAL "SEQUENTIAL"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "OMPONLY" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SCUDA")
  
    include_directories("/opt/intel/oneapi/include")
  
    find_path(CPATH  icc)
    set(CMAKE_C_COMPILER ${CPATH}/icc)
    find_path(CXXPATH icpc)  
    set(CMAKE_CXX_COMPILER ${CXXPATH}/icpc)  
  endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")  
  
# message("MPI_CXXPATH  = ${CXXPATH}")

elseif("${AParMooN_ARCH}" STREQUAL "CRAY64")

    find_path(CPATH cc)
    set(CMAKE_C_COMPILER ${CPATH}/cc)
    find_path(CXXPATH CC)  
    set(CMAKE_CXX_COMPILER ${CXXPATH}/CC)  

endif()


# set the define for compiler  options
  set(CMAKE_C_FLAGS "${PARMOON_C_DEF}")
  set(CMAKE_C_FLAGS_DEBUG "-g ${PARMOON_C_DEF}")
  set(CMAKE_C_FLAGS_RELEASE " -s  ${PARMOON_C_DEF}")
  set(CMAKE_CXX_FLAGS "${PARMOON_CXX_DEF} " )
  set(CMAKE_CXX_FLAGS_DEBUG "${PARMOON_CXX_DEF} ")
  set(CMAKE_CXX_FLAGS_RELEASE " -s ${PARMOON_CXX_DEF}") 
  set(CMAKE_CUDA_FLAGS "${PARMOON_CUDA_DEF} " )
  
 
include_directories("${CMAKE_SOURCE_DIR}/include/General")
include_directories("${CMAKE_SOURCE_DIR}/include/FE")
include_directories("${CMAKE_SOURCE_DIR}/include/Geometry")
include_directories("${CMAKE_SOURCE_DIR}/include/QuadFormulas")
include_directories("${CMAKE_SOURCE_DIR}/include/FE1D")
include_directories("${CMAKE_SOURCE_DIR}/include/FE2D")
include_directories("${CMAKE_SOURCE_DIR}/include/FE3D")
include_directories("${CMAKE_SOURCE_DIR}/include/AMG")
include_directories("${CMAKE_SOURCE_DIR}/include/PBE")
include_directories("${CMAKE_SOURCE_DIR}/include/Refinement")
include_directories("${CMAKE_SOURCE_DIR}/include/System")
include_directories("${CMAKE_SOURCE_DIR}/include/Parallel")
include_directories("${CMAKE_SOURCE_DIR}/Examples")
include_directories("${CMAKE_SOURCE_DIR}/include/Solvers")


if("${AParMooN_USE_ANN}" STREQUAL "TRUE")
  include_directories("${CMAKE_SOURCE_DIR}/include/ANN")
endif("${AParMooN_USE_ANN}" STREQUAL "TRUE")

if("${AParMooN_PARALLEL_TYPE}" STREQUAL "SCUDA" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "MCUDA" )
  include_directories("${CMAKE_SOURCE_DIR}/include/Solvers/cuda_general")
endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "SCUDA"  OR  "${AParMooN_PARALLEL_TYPE}" STREQUAL "MCUDA")


add_subdirectory(src/AMG) 
add_subdirectory(src/Refinement) 
add_subdirectory(src/QuadFormulas)  

add_subdirectory(src/Geometry)   
add_subdirectory(src/General) 
add_subdirectory(src/FE) 
add_subdirectory(src/System)
add_subdirectory(src/Solvers)

if("${AParMooN_USE_ANN}" STREQUAL "TRUE")
  add_subdirectory(src/ANN)
  set(ANN_LIBRARIES "ann")
else("${AParMooN_USE_ANN}" STREQUAL "FALSE")
  set(MLPACK_LIBRARIES "")
  set(ARMADILLO_LIBRARIES "")
  set(ANN_LIBRARIES "")
endif("${AParMooN_USE_ANN}" STREQUAL "TRUE")

 
# add_subdirectory(src/PBE) 


message("CMAKE_SOURCE_DIR  = ${CMAKE_SOURCE_DIR}")


if("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")
   add_subdirectory(src/Parallel) 
endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")

add_executable(parmoon_${AParMooN_GEO}_${AParMooN_PARALLEL_TYPE}.exe  ${AParMooN_MODEL})

if( "${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI" )
   target_link_libraries(parmoon_${AParMooN_GEO}_${AParMooN_PARALLEL_TYPE}.exe "solver" "sys" "par" "geo"  "fe" "gen" "geo" "fe" "gen"  
                      "amg"  "quad" "ref" "geo" "fe" "quad" "geo"  ${UMFPACK_LIBRARIES} ${TECPLOT_LIBRARIES}
                      ${MPIBLACS_LIBRARIES}  ${TETGEN_LIBRARIES} ${GRIDGEN_LIBRARIES}  ${MUMPS_LIBRARIES}   ${BLAS_LIBRARIES} ${MPI_LIBS} )  
elseif( "${AParMooN_PARALLEL_TYPE}" STREQUAL "SEQUENTIAL"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "OMPONLY")
   target_link_libraries(parmoon_${AParMooN_GEO}_${AParMooN_PARALLEL_TYPE}.exe "solver" "sys" "geo"  "fe" "gen" "geo" "fe" "gen" "fe"
                      "amg"  "quad" "ref" "geo" "fe" "quad" "geo" "solver" "sys" "fe" "geo" ${ANN_LIBRARIES} ${UMFPACK_LIBRARIES} ${TECPLOT_LIBRARIES}
                       ${TETGEN_LIBRARIES} ${GRIDGEN_LIBRARIES} ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES})

elseif("${AParMooN_PARALLEL_TYPE}" STREQUAL "SCUDA")
target_link_libraries(parmoon_${AParMooN_GEO}_${AParMooN_PARALLEL_TYPE}.exe "solver" "sys" "geo"  "fe" "gen" "geo" "fe" "gen" "fe"
                  "amg"  "quad" "ref" "geo" "fe" "quad" "geo" "solver" "sys" "fe" "geo" "cudaSpLinSolver" ${UMFPACK_LIBRARIES} ${TECPLOT_LIBRARIES}
                  ${BLAS_LIBRARIES} ${TETGEN_LIBRARIES} ${GRIDGEN_LIBRARIES}  -lcusparse -lculibos -lcublas -lcusolver -lnvToolsExt )
                  # ${BLAS_LIBRARIES} ${TETGEN_LIBRARIES} ${GRIDGEN_LIBRARIES}   -lculibos -lcublas -lnvToolsExt )
endif("${AParMooN_PARALLEL_TYPE}" STREQUAL "MPI"  OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "HYBRID" OR "${AParMooN_PARALLEL_TYPE}" STREQUAL "SMPI")


string(CONCAT runScriptString [=[#!/bin/bash
## Color Codes
# Black        0;30     Dark Gray     1;30
# Red          0;31     Light Red     1;31S
# Green        0;32     Light Green   1;32
# Brown/Orange 0;33     Yellow        1;33
# Blue         0;34     Light Blue    1;34
# Purple       0;35     Light Purple  1;35
# Cyan         0;36     Light Cyan    1;36
# Light Gray   0;37     White         1;37
RED='\033[0;31m'
NC='\033[0m' # No Color
GREEN='\033[0;32m'
if (( $# <= 1 )); then
    printf "${RED} Please pass the dat file as parameter${NC}\n"
    exit;
fi
cd ]=] "${CMAKE_CURRENT_BINARY_DIR}" 
[=[

make -j8 | tee make.out
OUTPUT=`make -j8 &> make.out`
OUTPUT=`cat make.out`
case "${OUTPUT}" in 
  *Error*)
  printf "${RED}******** ERROR ******** \n"
  printf "${RED}*Error in the make Process*\n"
  exit;
  ;;
esac
printf "${GREEN}******** SUCCESS ******** \n"
printf "${GREEN}*Make Process completed without any errors*\n${NC}"
cd ]=] "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}" 
[=[

./]=]
"parmoon_${AParMooN_GEO}_${AParMooN_PARALLEL_TYPE}.exe" [=[ $1]=]
 )
write_file("${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/runscript.sh" "${runScriptString}")

message("============================================================================================================"  )  
message("AParMooN_GEO=${AParMooN_GEO}")
message("AParMooN_MODEL=${AParMooN_MODEL}")
message("AParMooN_ARCH=${AParMooN_ARCH}")
message("AParMooN_PARALLEL_TYPE=${AParMooN_PARALLEL_TYPE}")
message("C_COMPILER=${CMAKE_C_COMPILER}")
message("CXX_COMPILER=${CMAKE_CXX_COMPILER}")
message("PARMOON_CXX_DEF=${PARMOON_CXX_DEF}")
message("EXE_FILE=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/parmoon_${AParMooN_GEO}_${AParMooN_PARALLEL_TYPE}.exe")
message("============================================================================================================"  )        
message("Configuration for the ParMooN Version 1.1")  
message("         Sashikumaar Ganesan")
message("         Dept. of Computational & Data Sciences")
message("         Indian Institute of Science, Bangalore, India")  
message("mailt0 : sashi-AT-iisc.ac.in") 
message("============================================================================================================"  ) 


