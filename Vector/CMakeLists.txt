cmake_minimum_required(VERSION 3.18)
project(cuda_samples LANGUAGES CXX CUDA)

# RTX 2080 = Ada Lovelace → sm_75
set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

set_property(DIRECTORY APPEND PROPERTY
    CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

foreach(src IN LISTS CUDA_SOURCES)
    get_filename_component(target "${src}" NAME_WE)
    add_executable("${target}" "${src}")
    set_target_properties("${target}" PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION OFF
    )
    target_link_libraries("${target}" PRIVATE
        CUDA::cudart
        CUDA::cublas
    )
endforeach()