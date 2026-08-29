cmake_minimum_required(VERSION 3.18)
project(flash_attention_v2_plus LANGUAGES CXX CUDA)


set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(CUDAToolkit REQUIRED COMPONENTS cublas)

set(CMAKE_CUDA_ARCHITECTURES "121")

add_executable(flash_attention_v2_plus
  flash_attn_mma.cu
  main.cu
)

target_include_directories(flash_attention_v2_plus PRIVATE .)

target_link_libraries(flash_attention_v2_plus PRIVATE cudart cuda cublas)