#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "context.cuh"

namespace bfgs {

  /// Initialize CURAND states for N parallel optimizations.
  /// Allocates device memory and launches kernel to set up PRNG states.
  /// Returns pointer to device memory containing initialized states.
  inline curandState*
  initializeStates(int N, int seed, float& ms_rand)
  {
    // PRNG setup
    curandState* d_states;
    cudaMalloc(&d_states, N * sizeof(curandState));

    // Launch setup
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    util::setupCurandStates<<<blocks, threads>>>(
      util::NonNull{d_states}, seed, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_rand, t0, t1);
    cudaDeviceSynchronize();
    return d_states;
  }

  /// Device helper to populate a Result structure.
  /// Used by BFGS kernels to record optimization outcomes.
  template <int DIM>
  __device__ void
  writeResult(Result<DIM>& r,
              int status,
              double fval,
              const double* coordinates,
              double gradientNorm,
              int iter,
              int idx)
  {
    r.status = status;
    r.iter = iter;
    r.fval = fval;
    r.idx = idx;
    r.gradientNorm = gradientNorm;
    for (int d = 0; d < DIM; ++d) {
      r.coordinates[d] = coordinates[d];
    }
  }

  /// Find and return the best result from N parallel optimizations.
  /// Uses CUB's DeviceReduce::ArgMin to find the minimum function value,
  /// then prints and returns the corresponding Result.
  template <int DIM>
  Result<DIM>
  launchReduction(int N, double* deviceResults, Result<DIM> const* h_results)
  {
    // ArgMin & final print
    int* d_argmin_idx;
    double* d_argmin_val;
    cudaMalloc(&d_argmin_idx, sizeof(int));
    cudaMalloc(&d_argmin_val, sizeof(double));
    void* d_temp_storage = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::ArgMin(
      d_temp_storage, temp_bytes, deviceResults, d_argmin_val, d_argmin_idx, N);
    cudaMalloc(&d_temp_storage, temp_bytes);
    cub::DeviceReduce::ArgMin(
      d_temp_storage, temp_bytes, deviceResults, d_argmin_val, d_argmin_idx, N);

    int globalMinIndex;
    cudaMemcpy(
      &globalMinIndex, d_argmin_idx, sizeof(int), cudaMemcpyDeviceToHost);

    // print the "best" thread's full record
    Result best = h_results[globalMinIndex];
    printf("Global best summary:\n");
    printf("   idx          = %d\n", best.idx);
    printf("   status       = %d\n", best.status);
    printf("   fval         = %.7e\n", best.fval);
    printf("   gradientNorm = %.7e\n", best.gradientNorm);
    printf("   iter         = %d\n", best.iter);
    printf("   coords       = [");
    for (int d = 0; d < DIM; ++d) {
      printf(" %.7e", best.coordinates[d]);
    }
    printf(" ]\n");

    cudaFree(d_argmin_idx);
    cudaFree(d_argmin_val);
    cudaFree(d_temp_storage);
    return best;
  }

} // namespace bfgs
