cmake_minimum_required(VERSION 3.10)
project(Zeus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 60)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(util
    src/utils.cu
    include/utils.cuh
)

add_executable(zeus main.cu)

target_link_libraries(zeus PUBLIC CUDA::cudart)
# Table‐generator executable: dumps a LaTeX table to stdout
#add_executable(cuda_table
#  table_generator.cu
#  parallel_pso.cu
#)

target_link_libraries(zeus PUBLIC util CUDA::cudart) # build util into zeus

add_executable(unit_test
  testing/test_main.cu
  main.cu
)
target_compile_definitions(unit_test PRIVATE UNIT_TEST)
target_link_libraries(unit_test
  PUBLIC
    util          # bring in vector_scale, vector_add
    CUDA::cudart
)
enable_testing()
#add_test(NAME RunCudaApp  COMMAND cuda_app -30 30 1000 100 0 10 1e-6)
add_test(NAME RunCudaTest COMMAND unit_test)

#add_executable(zeus_test test.cpp)
#target_include_directories(zeus_test PRIVATE ${CMAKE_SOURCE_DIR}/include)
#target_link_libraries(zeus_test PRIVATE util CUDA::cudart)

