#include "utils.cuh"
#include <string>
#include <fstream>
#include <string_view>
#include <iomanip>
#include <cmath>

namespace bfgs {}

namespace util {

  __global__ void
  setupXorwowStates(util::NonNull<curandStateXORWOW_t*> states,
                    uint64_t seed,
                    int N)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
      curand_init(
        seed, (unsigned long long)thread_id, 0ULL, &states[thread_id]);
    }
  }

  __global__ void
  setupPhiloxStates(util::NonNull<curandStatePhilox4_32_10_t*> states,
                    uint64_t seed,
                    int N)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
      curand_init(
        seed, (unsigned long long)thread_id, 0ULL, &states[thread_id]);
    }
  }

  __global__ void
  setupSobolStates(util::NonNull<curandStateSobol32_t*> states,
                   unsigned int* vectors,
                   int N)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
      curand_init(vectors, (unsigned int)thread_id, &states[thread_id]);
    }
  }

  void
  setStackSize()
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
           freeBytes / (1024.0 * 1024.0 * 1024.0),
           total / (1024.0 * 1024.0 * 1024.0));
    // Use a reasonable heap size (up to 1GB or 50% of free memory, whichever is
    // smaller)
    size_t newHeap = std::min(freeBytes / 2, (size_t)1024 * 1024 * 1024);
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeap);
    if (err != cudaSuccess) {
      printf("Failed to set heap to %zu bytes: %s\n",
             newHeap,
             cudaGetErrorString(err));
    } else {
      printf("Successfully set heap size to %.2f GB\n",
             newHeap / (1024.0 * 1024.0 * 1024.0));
    }
  }

  __global__ void
  fillWithNaN_kernel(double* d_ptr, size_t n)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      d_ptr[idx] = NAN;
    }
  }

  cudaError_t
  fillWithNaN(double* d_ptr, size_t n)
  {
    if (n == 0)
      return cudaSuccess;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    fillWithNaN_kernel<<<gridSize, blockSize>>>(d_ptr, n);
    return cudaGetLastError();
  }

  cudaError_t
  writeTrajectoryData(double* hostTrajectory,
                      int8_t* hostStatus,
                      int N,
                      int MAX_ITER,
                      int DIM,
                      std::string_view filename)
  {
    std::ofstream stepOut(std::string{filename});
    stepOut << "traj\tstep\tfval\tgrad\tstatus";
    for (int d = 0; d < DIM; d++)
      stepOut << "\tx" << d;
    stepOut << "\n";
    stepOut << std::scientific << std::setprecision(17);
    for (int i = 0; i < N; i++) {
      for (int it = 0; it < MAX_ITER; it++) {
        stepOut << i << "\t" << it;

        // Write fval and grad first
        double fval = hostTrajectory[trajectoryIndex(it, DIM, i, DIM + 2, N)];
        double gnorm =
          hostTrajectory[trajectoryIndex(it, DIM + 1, i, DIM + 2, N)];
        if (std::isnan(fval)) {
          stepOut << "\tNaN";
        } else {
          stepOut << "\t" << fval;
        }
        if (std::isnan(gnorm)) {
          stepOut << "\tNaN";
        } else {
          stepOut << "\t" << gnorm;
        }
        stepOut << "\t" << (int)hostStatus[it * N + i];

        // Then write coordinates
        for (int d = 0; d < DIM; d++) {
          double v = hostTrajectory[trajectoryIndex(it, d, i, DIM + 2, N)];
          if (std::isnan(v)) {
            stepOut << "\tNaN";
          } else {
            stepOut << "\t" << v;
          }
        }
        stepOut << "\n";
      }
    }
    stepOut.close();
    return cudaSuccess;
  }

} // end namespace util
