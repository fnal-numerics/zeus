#include "utils.cuh"
#include <string>

namespace bfgs {}

namespace util {

  void
  set_stack_size()
  {
    // logic to set the stact size limit to 65 kB per thread
    size_t currentStackSize = 0;
    cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
    // printf("Current stack size: %zu bytes\n", currentStackSize);
    size_t newStackSize = 64 * 1024; // 65 kB
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (err != cudaSuccess) {
      printf("cudaDeviceSetLimit Stack error: %s\n", cudaGetErrorString(err));
    }

    // get current device ID
    int device;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      printf("cudaGetDevice error: %s\n", cudaGetErrorString(err));
      return;
    }

    // Get device properties (including total global memory)
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, device);
    if (err != cudaSuccess) {
      printf("cudaGetDeviceProperties error: %s\n", cudaGetErrorString(err));
      return;
    }

    // Use total free memory as heap size
    size_t freeBytes = 0, total = 0;
    cudaMemGetInfo(&freeBytes, &total);
    printf("GPU reporting %.2f GB free of %.2f GB total\n",
           freeBytes / 1e9,
           total / 1e9);
    size_t newHeap = freeBytes;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeap);
    if (err != cudaSuccess) {
      printf("Failed to set heap to %zu bytes: %s\n",
             newHeap,
             cudaGetErrorString(err));
    } else {
      printf("Successfully set heap size to %zu bytes (%.2f GB)\n",
             newHeap,
             newHeap / (1024.0 * 1024.0 * 1024.0));
    }
  }

  cudaError_t
  writeTrajectoryData(double* hostTrajectory,
                      int N,
                      int MAX_ITER,
                      int DIM,
                      const std::string& fun_name,
                      const std::string& basePath)
  {
    // construct the directory path and create it.
    std::string dirPath = basePath + "/" + fun_name + "/" +
                          std::to_string(DIM) + "d/" +
                          std::to_string(MAX_ITER * N) + "/trajectories";
    std::filesystem::create_directories(dirPath);
    // createOutputDirs(dirPath);

    // the final filename.
    std::string filename = dirPath + "/" + std::to_string(MAX_ITER) + "it_" +
                           std::to_string(N) + ".tsv";

    std::ofstream stepOut(filename);
    stepOut << "OptIndex\tStep";
    for (int d = 0; d < DIM; d++)
      stepOut << "\tX_" << d;
    stepOut << "\n";
    stepOut << std::scientific << std::setprecision(17);
    for (int i = 0; i < N; i++) {
      for (int it = 0; it < MAX_ITER; it++) {
        stepOut << i << "\t" << it;
        for (int d = 0; d < DIM; d++) {
          stepOut << "\t"
                  << hostTrajectory[i * (MAX_ITER * DIM) + it * DIM + d];
        }
        stepOut << "\n";
      }
    }
    stepOut.close();
    return cudaSuccess;
  }

  __device__ double
  dot_product_device(const double* a, const double* b, int size)
  {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  __device__ void
  outer_product_device(const double* v1,
                       const double* v2,
                       double* result,
                       int size)
  {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        int idx = i * size + j;
        if (idx < size * size) {
          result[idx] = v1[i] * v2[j];
        }
      }
    }
  }

  extern "C" {
  __device__ void
  vector_add(const double* a, const double* b, double* result, int size)
  {
    for (int i = 0; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
  }

  __device__ void
  vector_scale(const double* a, double scalar, double* result, int dim)
  {
    for (int i = 0; i < dim; ++i) {
      result[i] = a[i] * scalar;
    }
  }

  } // end extern C

  __device__ double
  pow2(double x)
  {
    return x * x;
  }

  template <int DIM>
  __device__ void
  matrix_multiply_device(const double* A, const double* B, double* C)
  {
    for (int i = 0; i < DIM; ++i) {
      for (int j = 0; j < DIM; ++j) {
        double sum = 0.0;
        for (int k = 0; k < DIM; ++k) {
          sum += A[i * DIM + k] * B[k * DIM + j];
        }
        C[i * DIM + j] = sum;
      }
    }
  }

  __device__ double
  generate_random_double(curandState* state, double lower, double upper)
  {
    return lower + (upper + (-lower)) * curand_uniform_double(state);
  }

  __global__ void
  setup_curand_states(util::non_null<curandState*> states, uint64_t seed, int N)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      curand_init(seed, idx, 0, &states[idx]);
    }
  }

  __device__ double
  atomicMinDouble(double* addr, double val)
  {
    // reinterpret the address as 64‑bit unsigned
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old_bits = *ptr, assumed_bits;

    do {
      assumed_bits = old_bits;
      double old_val = __longlong_as_double(assumed_bits);
      // if the current value is already <= our candidate, nothing to do
      if (old_val <= val)
        break;
      // else try to swap in the new min value’s bit‐pattern
      unsigned long long new_bits = __double_as_longlong(val);
      old_bits = atomicCAS(ptr, assumed_bits, new_bits);
    } while (assumed_bits != old_bits);

    // return the previous minimum
    return __longlong_as_double(old_bits);
  }

} // end namespace util
