
##############################################################################
# There are two ways to add include directories to the NVCC command
# line:
cmake_minimum_required(VERSION 2.6)

project(experimental)
#set(CUDA_PROPAGATE_HOST_FLAGS OFF)

FIND_PACKAGE(CUDA 3.0)

# The cuda_include_directories adds paths to only cuda compilation.
CUDA_INCLUDE_DIRECTORIES(
    ${CMAKE_CURRENT_SOURCE_DIR}
    /home/denis/Projects/include
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
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -std=c++0x)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)

set(CUDA_BUILD_EMULATION off)

LINK_DIRECTORIES(${LINK_DIRECTORIES} ./cudpp/)

# Use one executable only.
CUDA_ADD_EXECUTABLE(exp
    main.cpp
    devmanager.cpp
    benchmark.cpp
)
TARGET_LINK_LIBRARIES(exp
)

# Or compile the cuda code into a shared library.

# Anything other than -D or /D is not passed along to nvcc.
add_definitions(-DMULTIPLIER=2)

CUDA_BUILD_CLEAN_TARGET()

