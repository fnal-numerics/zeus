cmake_minimum_required(VERSION 3.10)
project(Zeus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 80)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(util
    src/utils.cu
    include/utils.cuh
)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
add_executable(zeus main.cpp)
target_link_libraries(zeus PUBLIC CUDA::cudart)
target_link_libraries(zeus PUBLIC util CUDA::cudart)

target_compile_options(zeus PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)

set_source_files_properties(metaprogramming_t.cpp PROPERTIES LANGUAGE CUDA)
add_executable(zeus_test metaprogramming_t.cpp)
target_compile_options(zeus_test
    PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)

target_compile_options(zeus_test PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)

target_include_directories(zeus_test PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(zeus_test PRIVATE
  util
  CUDA::cudart
)

target_link_libraries(zeus_test PUBLIC util CUDA::cudart) # build util into zeus

add_executable(unit_test
  testing/test_main.cu
  main.cpp
)
target_compile_options(unit_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>
)
target_compile_definitions(unit_test PRIVATE UNIT_TEST)
target_link_libraries(unit_test
  PUBLIC
    util          # bring in vector_scale, vector_add
    CUDA::cudart
)
enable_testing()
add_test(NAME RunCudaTest COMMAND unit_test)

