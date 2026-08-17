cmake_minimum_required(VERSION 3.18)
project(cuda_samples LANGUAGES CXX CUDA)

# RTX 4090 = Ada Lovelace → sm_89
set(CMAKE_CUDA_ARCHITECTURES "89")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

# ============================================================
# 1. Static library: attention (all kernels + launch functions)
# ============================================================
add_library(attention_lib STATIC
    attention.cu
    attention.cuh
)
set_target_properties(attention_lib PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)
target_link_libraries(attention_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
)
target_include_directories(attention_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ============================================================
# 2. Executables: flash_attention_v1/v2/v3
# ============================================================
set(TEST_SOURCES
    flash_attention2_v1.cu
    flash_attention2_v2.cu
    flash_attention2_v3.cu
    attention_baseline.cu
)

foreach(src IN LISTS TEST_SOURCES)
    get_filename_component(target "${src}" NAME_WE)
    add_executable("${target}" "${src}")
    set_target_properties("${target}" PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries("${target}" PRIVATE
        attention_lib
    )
endforeach()