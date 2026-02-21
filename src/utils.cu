#include "utils.cuh"
#include <string>

namespace bfgs {}

namespace util {

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

  __global__ void
  setupCurandStates(util::NonNull<curandState*> states, uint64_t seed, int N)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
      curand_init(seed, idx, 0, &states[idx]);
    }
  }

} // end namespace util
