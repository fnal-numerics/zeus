cmake_minimum_required(VERSION 3.25)

# Must be set before project() so that it takes effect when CMake initializes
# the CUDA language. If the user passes -DCMAKE_CUDA_ARCHITECTURES=... on the
# command line the cache variable will already be defined and this is skipped.
if(NOT DEFINED CACHE{CMAKE_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "CUDA architectures to compile for")
endif()

project(Zeus LANGUAGES CXX CUDA)

# Use modern CMake CUDA runtime linking (silences pgacclnk -lcudart_static warning)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# (Default is now set before project(); nothing needed here.)

# Zeus requires static libraries: nvlink cannot extract device objects from a
# shared library (.so), so BUILD_SHARED_LIBS=ON would silently break every
# executable that uses -rdc=true (relocatable device code).
if(BUILD_SHARED_LIBS)
    message(FATAL_ERROR
        "BUILD_SHARED_LIBS=ON is not supported by Zeus.\n"
        "CUDA separable compilation (-rdc=true) requires that all device code "
        "be available to nvlink as static object files at build time. "
        "A shared library cannot satisfy this requirement.\n"
        "Please configure with -DBUILD_SHARED_LIBS=OFF (the default).")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Zeus requires static libraries" FORCE)

# Options
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set(ZEUS_MASTER_PROJECT ON)
else()
    set(ZEUS_MASTER_PROJECT OFF)
endif()

option(ZEUS_BUILD_TESTS "Build Zeus tests" ${ZEUS_MASTER_PROJECT})
option(ZEUS_BUILD_EXAMPLES "Build Zeus examples" ${ZEUS_MASTER_PROJECT})
option(ZEUS_USE_NETCDF4 "Enable NetCDF4 trajectory output support" ON)

# Dependencies
find_package(CUDAToolkit REQUIRED)

if(ZEUS_USE_NETCDF4)
    # Try PkgConfig first as it's often more reliable on HPC systems for NetCDF4 C++
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(NetCDF_PKG netcdf-cxx4)
        if(NetCDF_PKG_FOUND)
            set(NetCDF_FOUND ON)
            set(NetCDF_VERSION ${NetCDF_PKG_VERSION})
            # Create aliases for compatibility with the rest of the script
            if(NOT TARGET NetCDF::NetCDF_CXX)
                add_library(NetCDF::NetCDF_CXX INTERFACE IMPORTED)
                target_link_libraries(NetCDF::NetCDF_CXX INTERFACE ${NetCDF_PKG_LIBRARIES})
                target_include_directories(NetCDF::NetCDF_CXX INTERFACE ${NetCDF_PKG_INCLUDE_DIRS})
                target_link_directories(NetCDF::NetCDF_CXX INTERFACE ${NetCDF_PKG_LIBRARY_DIRS})
            endif()
            # NetCDF C is usually a dependency of CXX, but we'll add it if needed
            if(NOT TARGET NetCDF::NetCDF_C)
                add_library(NetCDF::NetCDF_C INTERFACE IMPORTED)
            endif()
        endif()
    endif()

    if(NOT NetCDF_FOUND)
        # Fallback to standard FindNetCDF or config
        find_package(NetCDF COMPONENTS C CXX)
    endif()

    if(NetCDF_FOUND)
        message(STATUS "✅ Found NetCDF: ${NetCDF_VERSION}")
    else()
        message(WARNING "⚠️ NetCDF not found. Disabling NetCDF4 support.")
        set(ZEUS_USE_NETCDF4 OFF)
    endif()
endif()

# Core Library
add_library(zeus
    src/utils.cu
    src/util.cpp
    src/pso.cu
    src/exception.cpp
    src/duals.cu
    include/util.hpp
    include/utils.cuh
    include/cuda_buffer.hpp
    include/zeus.cuh
    include/bfgs_common.cuh
    include/bfgs_parallel.cuh
    include/bfgs_sequential.cuh
    include/context.hpp
    include/device_matrix.cuh
    include/duals.cuh
    include/exception.hpp
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
        CUDA::curand
)

if(ZEUS_USE_NETCDF4)
    target_compile_definitions(zeus PUBLIC ZEUS_HAS_NETCDF4)
    target_link_libraries(zeus PUBLIC NetCDF::NetCDF_C NetCDF::NetCDF_CXX)
endif()

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
    # Individual optimizer programs
    set(OPTIMIZERS rosenbrock rastrigin ackley goldstein_price himmelblau)
    set(OPTIMIZER_FLAGS $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr -O3 --use_fast_math -Xptxas=-O3 -rdc=true>)
    
    foreach(optimizer IN LISTS OPTIMIZERS)
        add_executable(optimize_${optimizer} examples/optimize_${optimizer}.cu)
        target_include_directories(optimize_${optimizer} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
        target_link_libraries(optimize_${optimizer} PRIVATE zeus)
        target_compile_options(optimize_${optimizer} PRIVATE ${OPTIMIZER_FLAGS})
    endforeach()

    # Dijet spectrum fit (HEP real-world example)
    add_executable(fit_dijet_spectrum examples/fit_dijet_spectrum.cu)
    target_include_directories(fit_dijet_spectrum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(fit_dijet_spectrum PRIVATE zeus)
    target_compile_options(fit_dijet_spectrum PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -expt-relaxed-constexpr -O3 --use_fast_math -Xptxas=-O3 -rdc=true>)

    # Gaussian + neural-network weight fitting examples
    add_executable(optimize_gaussian_nn examples/optimize_gaussian_nn.cu)
    target_include_directories(optimize_gaussian_nn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    target_link_libraries(optimize_gaussian_nn PRIVATE zeus)
    target_compile_options(optimize_gaussian_nn PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda -expt-relaxed-constexpr -O3 --use_fast_math -Xptxas=-O3 -rdc=true>)
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

    # Compile the Catch2 session runner once as a plain C++ static library.
    # Every test executable links against this instead of Catch2::Catch2WithMain,
    # so main() is never recompiled per target.
    add_library(test_main STATIC test/catch_main.cpp)
    target_link_libraries(test_main PUBLIC Catch2::Catch2)

    enable_testing()
    include(Catch)

    add_executable(non_null_tests test/non_null_tests.cu)
    set_target_properties(non_null_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(non_null_tests PRIVATE zeus test_main)
    catch_discover_tests(non_null_tests REPORTER console)

    # One executable per test file so all CUDA device-link (nvlink) steps run
    # in parallel rather than waiting on a single serial nvlink over all TUs.
    set(ZEUS_UNIT_TESTS
        util_tests
        fun_tests
        pso_tests
        bfgs_tests
        dual_tests
        matrix_tests
        cuda_buffer_tests
        device_matrix_tests
    )

    foreach(test_name IN LISTS ZEUS_UNIT_TESTS)
        add_executable(${test_name} test/${test_name}.cu)
        set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/examples)
        target_compile_options(${test_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
        target_compile_definitions(${test_name} PRIVATE UNIT_TEST)
        target_link_libraries(${test_name} PRIVATE zeus test_main)
        catch_discover_tests(${test_name} REPORTER console)
    endforeach()

    # Bad objectives test (compile-time failure verification)
    set(BAD_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test/bad_objectives.cu")
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

    # Bad objectives with C++20 concepts test (compile-time failure verification)
    set(BAD_CONCEPTS_SRC "${CMAKE_CURRENT_SOURCE_DIR}/test/bad_objectives_concepts.cu")
    try_compile(BAD_CONCEPTS_OK
      "${CMAKE_CURRENT_BINARY_DIR}"
      "${BAD_CONCEPTS_SRC}"
      CMAKE_FLAGS
        "-DCMAKE_CUDA_STANDARD=20"
        "-DCMAKE_CXX_STANDARD=20"
      LINK_LIBRARIES zeus
      OUTPUT_VARIABLE BAD_CONCEPTS_OUT
    )

    if(BAD_CONCEPTS_OK)
      message(FATAL_ERROR "❌ bad_objectives_concepts.cu unexpectedly compiled!\n${BAD_CONCEPTS_OUT}")
    else()
      message(STATUS "✅ bad_objectives_concepts.cu failed to compile as expected (concept violations detected).")
    endif()
endif()
