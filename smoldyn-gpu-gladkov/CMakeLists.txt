
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:
cmake_minimum_required(VERSION 2.8)

project(SmolGPU)
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
FIND_PACKAGE(CUDA 3.0 REQUIRED)
FIND_PACKAGE(OpenGL REQUIRED)
FIND_PACKAGE(GLEW REQUIRED)
FIND_PACKAGE(GLUT REQUIRED)
FIND_PACKAGE(ZLIB REQUIRED)
FIND_PACKAGE(PNG REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
    FIND_PACKAGE(Boost REQUIRED)
    INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})
    ADD_DEFINITIONS("-DHAS_BOOST")
endif()


#set(CUDPP_INCLUDE_PATH "default value" CACHE PATH "path to CUDPP include directory")
#FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUDPP DEFAULT_MSG ${CUDPP_INCLUDE_PATH})
if(NOT DEFINED CUDPP_INCLUDE_PATH)
    MESSAGE(FATAL_ERROR "Please define the path for CUDPP inlcude directory in CUDPP_INCLUDE_PATH variable")
endif(NOT DEFINED CUDPP_INCLUDE_PATH)

IF(NOT EXISTS "${CUDPP_INCLUDE_PATH}" OR NOT IS_DIRECTORY "${CUDPP_INCLUDE_PATH}")
    MESSAGE(FATAL_ERROR "Wrong path used for CUDPP_INCLUDE_PATH variable")
ENDIF(NOT EXISTS "${CUDPP_INCLUDE_PATH}" OR NOT IS_DIRECTORY "${CUDPP_INCLUDE_PATH}")


if(NOT DEFINED CUDPP_LIBRARY)
    MESSAGE(FATAL_ERROR "Please define the path for CUDPP inlcude directory in CUDPP_LIBRARY variable")
endif(NOT DEFINED CUDPP_LIBRARY)

IF(NOT EXISTS "${CUDPP_LIBRARY}" OR IS_DIRECTORY "${CUDPP_LIBRARY}")
    MESSAGE(FATAL_ERROR "Wrong path used for CUDPP_LIBRARY variable")
ENDIF(NOT EXISTS "${CUDPP_LIBRARY}" OR IS_DIRECTORY "${CUDPP_LIBRARY}")


MESSAGE("${GLUT_LIBRARY}")
MESSAGE("${ZLIB_LIBRARIES}")
MESSAGE("${PNG_LIBRARIES}")
#link_directories(${GLUT_LIBRARY_DIRS})

INCLUDE_DIRECTORIES(${GLEW_INCLUDE_PATH}
    ${GLUT_INCLUDE_PATH}
    ${PNG_PNG_INCLUDE_DIR}
    ${CUDPP_INCLUDE_PATH})
#FIND_PACKAGE(CUDA QUIET REQUIRED)
# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
)



# The include_directories adds paths to both C/C++ compilation in the native
# compiler and cuda compilation in NVCC.  Note that CUDA_INCLUDE_DIRS is added
# automatically by CUDA_ADD_EXECUTABLE and CUDA_ADD_LIBRARY.

# INCLUDE_DIRECTORIES(
#  /home/denis/Projects/GasDSMC/include
# )

##############################################################################
# There are four ways to compile source files with NVCC. 

# Set CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE when you want to add the same .cu
# file to multiple targets.
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=sm_20)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -g -G)

#set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++0x)

#enable this for realase
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -O3)


#set(CUDA_BUILD_EMULATION off)

#LINK_DIRECTORIES( ${LINK_DIRECTORIES} ./cudpp/ )

# Use one executable only.
CUDA_ADD_EXECUTABLE(SmoldynGPU
    smoldyn.cpp
    sim_settings.cpp
    draw_helper.cpp
    data_providers.cu
    molecules.cu
    radixsort.cu
    smol_kernels.cu
    mtwister.cpp
    model.cpp
    matrix.cpp
    frame3d.cpp
    image_helper.cpp
    genmtrand.c
    aabb.cpp
    vector.cpp
    reggrid.cpp
    radixsort.cpp
    shader.cpp
    smolparser.cpp
    smol_solver.cpp
    smolparameters.cpp
    texture.cpp
    log_file.cpp
    cmd_handler.cpp
    call_policy.cpp
    file_path.cpp
    cudahelp/common.cpp
    cudahelp/random/seedgen.cpp
)

# add_library(${ZLIB_LIBRARIES} STATIC IMPORTED)
# add_library(${PNG_LIBRARIES} STATIC IMPORTED)

TARGET_LINK_LIBRARIES(SmoldynGPU
    ${OPENGL_LIBRARIES}
    ${GLU_LIBRARY}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARY}
    ${ZLIB_LIBRARIES}
    ${PNG_LIBRARIES}
    ${CUDPP_LIBRARY}
    -lGLU
)

# Or compile the cuda code into a shared library.

# Anything other than -D or /D is not passed along to nvcc.
add_definitions(-DMULTIPLIER=2)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/sprite2.png DESTINATION ${CMAKE_BINARY_DIR})

CUDA_BUILD_CLEAN_TARGET()

