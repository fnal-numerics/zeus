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
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CUDA_ARCH "60")

# Build the normal executable (uses main() from main.cu)
add_executable(cuda_app main.cu)
set_target_properties(cuda_app PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

# Build an OBJECT library from main.cu (it contains a main() that we want to remove)
add_library(main_obj OBJECT main.cu)
set_target_properties(main_obj PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

find_program(OBJCOPY_EXECUTABLE objcopy REQUIRED)
get_target_property(MAIN_OBJ_FILES main_obj OBJECT_FILES)
list(GET MAIN_OBJ_FILES 0 MAIN_OBJ_FILE)
set(STRIPPED_MAIN ${CMAKE_BINARY_DIR}/main_stripped.o)
add_custom_command(
  OUTPUT ${STRIPPED_MAIN}
  COMMAND ${OBJCOPY_EXECUTABLE} --strip-symbol=main ${MAIN_OBJ_FILE} ${STRIPPED_MAIN}
  DEPENDS ${MAIN_OBJ_FILE}
  COMMENT "Stripping main() from main.o"
)
add_custom_target(main_stripped_obj ALL DEPENDS ${STRIPPED_MAIN})

# Build the test executable from test_main.cu (contains separate test functions) and the stripped main.o 
add_executable(cuda_test test_main.cu ${STRIPPED_MAIN})
add_dependencies(cuda_test main_stripped_obj)
set_target_properties(cuda_test PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCH})

enable_testing()
add_test(NAME RunCudaApp COMMAND cuda_app)
add_test(NAME RunCudaTest COMMAND cuda_test)

