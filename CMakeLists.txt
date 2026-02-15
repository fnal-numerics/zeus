cmake_minimum_required(VERSION 3.24)
project(Zeus LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES native)

# Default to shared libraries if not specified
if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_SHARED_LIBS ON)
endif()

# Options
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(ZEUS_MASTER_PROJECT ON)
else()
    set(ZEUS_MASTER_PROJECT OFF)
endif()

option(ZEUS_BUILD_TESTS "Build Zeus tests" ${ZEUS_MASTER_PROJECT})
option(ZEUS_BUILD_EXAMPLES "Build Zeus examples" ${ZEUS_MASTER_PROJECT})

# Dependencies
find_package(CUDAToolkit REQUIRED)

# Core Library
add_library(zeus
    src/utils.cu
    src/util.cpp
    src/pso.cu
    include/util.hpp
    include/utils.cuh
    include/cuda_buffer.cuh
    include/zeus.cuh
    include/bfgs_common.cuh
    include/bfgs_parallel.cuh
    include/bfgs_sequential.cuh
    include/context.cuh
    include/device_matrix.cuh
    include/duals.cuh
    include/exception.hpp
    include/fun.h
    include/matrix.hpp
    include/pso.cuh
    include/traits.hpp
)

add_library(Zeus::zeus ALIAS zeus)

set_target_properties(zeus PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_include_directories(zeus
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(zeus
    PUBLIC
        CUDA::cudart
)

target_compile_options(zeus
    PUBLIC
        $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr -O3 --use_fast_math -Xptxas=-O3 -rdc=true>
)

# Installation support
include(GNUInstallDirs)

install(TARGETS zeus
    EXPORT ZeusTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.cuh"
)

# Export configuration
install(EXPORT ZeusTargets
    FILE ZeusTargets.cmake
    NAMESPACE Zeus::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Zeus
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/ZeusConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/ZeusConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Zeus
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ZeusConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/Zeus
)

# Examples
if(ZEUS_BUILD_EXAMPLES)
    set_source_files_properties(examples/stateless_funs.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(stateless examples/stateless_funs.cpp)
    target_link_libraries(stateless PRIVATE zeus)
    target_compile_options(stateless PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr -O3 --use_fast_math -Xptxas=-O3 -rdc=true>)

    set_source_files_properties(examples/metaprogramming_t.cpp PROPERTIES LANGUAGE CUDA)
    add_executable(zeus_test examples/metaprogramming_t.cpp)
    target_link_libraries(zeus_test PRIVATE zeus)
    target_compile_options(zeus_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -expt-relaxed-constexpr -lineinfo>)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(zeus_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
    endif()
endif()

# Tests
if(ZEUS_BUILD_TESTS)
    include(FetchContent)
    include(CTest)
    FetchContent_Declare(
      catch2
      GIT_REPOSITORY https://github.com/catchorg/Catch2.git
      GIT_TAG        v3.4.0
    )
    FetchContent_MakeAvailable(catch2)

    add_executable(non_null_tests testing/non_null_tests.cu)
    target_link_libraries(non_null_tests PRIVATE zeus Catch2::Catch2WithMain)
    add_test(NAME non_null_tests COMMAND non_null_tests)

    add_executable(unit_test
      testing/catch_main.cpp
      testing/util_tests.cu
      testing/fun_tests.cu
      testing/pso_tests.cu
      testing/bfgs_tests.cu
      testing/dual_tests.cu
      testing/matrix_tests.cu
      testing/cuda_buffer_tests.cu
      testing/device_matrix_tests.cu
    )
    
    set_target_properties(unit_test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(unit_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
    target_compile_definitions(unit_test PRIVATE UNIT_TEST)
    target_link_libraries(unit_test PRIVATE zeus Catch2::Catch2WithMain)
    
    enable_testing()
    include(Catch)
    catch_discover_tests(unit_test REPORTER console)

    # Bad objectives test (compile-time failure verification)
    set(BAD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/testing/bad_objectives.cu")
    # Note: try_compile is tricky with FetchContent and dependencies. 
    # For now, keeping it as is but it might need refinement if it fails to see zeus headers.
    try_compile(BAD_OK
      "${CMAKE_CURRENT_BINARY_DIR}"
      "${BAD_SRC}"
      CMAKE_FLAGS
        "-DCMAKE_CUDA_STANDARD=14"
        "-DCMAKE_CXX_STANDARD=17"
      LINK_LIBRARIES zeus
      OUTPUT_VARIABLE BAD_OUT
    )

    if(BAD_OK)
      message(FATAL_ERROR "❌ bad_objectives.cu unexpectedly compiled!\n${BAD_OUT}")
    else()
      message(STATUS "✅ bad_objectives.cu failed to compile as expected.")
    endif()
endif()
