cmake_minimum_required(VERSION 3.18)
project(cuda_samples LANGUAGES CXX CUDA)

# RTX 4090 = Ada Lovelace → sm_89
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "75")
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 找到 CUDA 运行时 + cuBLAS
find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

# 自动收集所有 .cu 文件
file(GLOB CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# 每次 build 前检查目录是否有新增/删除的 .cu 文件，有则自动触发 cmake 重新配置
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