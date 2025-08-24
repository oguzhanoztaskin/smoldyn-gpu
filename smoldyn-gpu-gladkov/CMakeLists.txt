cmake_minimum_required(VERSION 3.18) # need >= 3.18 for CUDA_ARCHITECTURES
project(SmolGPU LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PNG REQUIRED)

if(CMAKE_COMPILER_IS_GNUCXX)
    find_package(Boost REQUIRED)
    include_directories(${Boost_INCLUDE_DIR})
    add_definitions("-DHAS_BOOST")
endif()

include_directories(
    ${GLEW_INCLUDE_DIRS}
    ${GLUT_INCLUDE_PATH}
    ${PNG_PNG_INCLUDE_DIR}
)

# Auto-select CUDA architecture (sm_89 for RTX 4060, but extendable)
# You can hardcode "89" if you only care about your machine
set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures")

add_executable(SmoldynGPU
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

# Set build type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# Compiler flags for CUDA and C++
target_compile_options(SmoldynGPU PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:$<IF:$<CONFIG:Debug>,-g -G -O0,-O3>>
    $<$<COMPILE_LANGUAGE:CXX>:$<IF:$<CONFIG:Debug>,-g -O0,-O3>>
)

# Link libraries
target_link_libraries(SmoldynGPU
    CUDA::cudart
    ${OPENGL_LIBRARIES}
    ${GLU_LIBRARY}
    ${GLUT_LIBRARY}
    ${GLEW_LIBRARIES}
    ${ZLIB_LIBRARIES}
    ${PNG_LIBRARIES}
)

# Copy resource files (example: sprite2.png)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/sprite2.png DESTINATION ${CMAKE_BINARY_DIR})
