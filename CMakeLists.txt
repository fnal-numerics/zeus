cmake_minimum_required(VERSION 3.10)
project(Zeus LANGUAGES CXX CUDA)

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

add_executable(zeus zeus.cu)

target_link_libraries(zeus PUBLIC CUDA::cudart)
# Table‐generator executable: dumps a LaTeX table to stdout
#add_executable(cuda_table
#  table_generator.cu
#  parallel_pso.cu
#)

target_link_libraries(zeus PUBLIC util CUDA::cudart) # build util into zeus
target_link_libraries(zeus PUBLIC util CUDA::cudart)
# target_compile_definitions(cuda_table PRIVATE …)

#target_link_libraries(cuda_table PUBLIC CUDA::cudart)

#add_executable(test testing/test_main.cu)
add_executable(unit_test
  testing/test_main.cu
  zeus.cu
)
target_compile_definitions(unit_test PRIVATE UNIT_TEST)
#target_link_libraries(unit_test PUBLIC CUDA::cudart)
target_link_libraries(unit_test
  PUBLIC
    util          # bring in vector_scale, vector_add
    CUDA::cudart
)
enable_testing()
#add_test(NAME RunCudaApp  COMMAND cuda_app -30 30 1000 100 0 10 1e-6)
add_test(NAME RunCudaTest COMMAND unit_test)

