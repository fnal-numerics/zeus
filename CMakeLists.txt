cmake_minimum_required(VERSION 3.10)
project(Zeus VERSION 1.0 LANGUAGES CXX CUDA)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-11.8")
find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda-11.8/bin)
if(NOT CMAKE_CUDA_COMPILER)
  message(FATAL_ERROR "nvcc not found")
endif()

set(CMAKE_CUDA_ARCHITECTURES 70)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_ARCH "60")

# Production executable
add_executable(cuda_app parallel_pso.cu)
set_target_properties(cuda_app PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH} CUDA_SEPARABLE_COMPILATION ON)

# Test executable: compile both main.cu and test_main.cu with UNIT_TEST defined
add_executable(cuda_test testing/test_main.cu parallel_pso.cu)
target_compile_definitions(cuda_test PRIVATE UNIT_TEST)
set_target_properties(cuda_test PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH}
	CUDA_SEPARABLE_COMPILATION ON)

enable_testing()
#add_test(NAME RunCudaApp COMMAND cuda_app)
add_test(NAME RunCudaApp
         COMMAND cuda_app
           -30 30          # bounds
           1000            # max_iter
           100             # pso_iters
           0               # converged (flag)
           10              # number_of_optimizations
           1e-6            # tolerance
)
add_test(NAME RunCudaTest COMMAND cuda_test)

