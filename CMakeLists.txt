cmake_minimum_required(VERSION 3.24)
project(Zeus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

#set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_ARCHITECTURES native)
#set_property(TARGET stateless PROPERTY CUDA_ARCHITECTURES OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


find_package(CUDAToolkit REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)
add_library(util
    src/bfgs.cu
    src/utils.cu
    src/util.cpp
    include/util.hpp
    include/utils.cuh
    src/cuda_buffer.cu
)

set_source_files_properties(benchmarks/stateless_funs.cpp PROPERTIES LANGUAGE CUDA)
add_executable(stateless benchmarks/stateless_funs.cpp)
target_link_libraries(stateless PUBLIC CUDA::cudart)
target_link_libraries(stateless PUBLIC util CUDA::cudart)
target_compile_options(stateless PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)

set_source_files_properties(benchmarks/examples/metaprogramming_t.cpp PROPERTIES LANGUAGE CUDA)
add_executable(zeus_test benchmarks/examples/metaprogramming_t.cpp)
target_compile_options(zeus_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(zeus_test PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
target_include_directories(zeus_test PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(zeus_test PRIVATE  util  CUDA::cudart)
target_link_libraries(zeus_test PUBLIC util CUDA::cudart) # build util into zeus

target_compile_options(zeus_test PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
)

## TESTING SETUP
include(FetchContent)
include(CTest)
FetchContent_Declare(
  catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(catch2)

add_executable(unit_test
  testing/catch_main.cpp
  testing/util_tests.cu
  testing/fun_tests.cu
  testing/pso_tests.cu
  testing/bfgs_tests.cu
  testing/dual_tests.cu
  testing/matrix_tests.cu
  testing/cuda_buffer_tests.cu
)

set_target_properties(unit_test PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
target_compile_options(unit_test PRIVATE  $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
target_compile_definitions(unit_test PRIVATE UNIT_TEST)
target_include_directories(unit_test  PRIVATE $<TARGET_PROPERTY:Catch2::Catch2WithMain,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(unit_test
  PUBLIC
    util          # bring in vector_scale, vector_add
    CUDA::cudart
    Catch2::Catch2WithMain
)
enable_testing()
include(CTest)
include(Catch)
catch_discover_tests(unit_test
  #TEST_PREFIX unit_
  REPORTER console
)


# set up bad objectives
set(BAD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/testing/bad_objectives.cu")
try_compile(BAD_OK
  "${CMAKE_CURRENT_BINARY_DIR}"
  "${BAD_SRC}"
  CMAKE_FLAGS
    "-DCMAKE_CUDA_STANDARD=14"
    "-DCMAKE_CXX_STANDARD=17"
  OUTPUT_VARIABLE BAD_OUT
)

if(BAD_OK)
  message(FATAL_ERROR
    "❌ bad_objectives.cu unexpectedly compiled!\n${BAD_OUT}"
  )
else()
  message(STATUS
    "✅ bad_objectives.cu failed to compile as expected."
  )
endif()

