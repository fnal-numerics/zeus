#pragma once

#include <array>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "utils.cuh"
#include "context.hpp"

namespace bfgs {

  /// Initialize CURAND states for N parallel optimizations.
  /// Allocates device memory and launches kernel to set up PRNG states.
  /// Returns pointer to device memory containing initialized states.
  template <typename StateType = curandState>
  inline StateType*
  initializeStates(int N, int seed, float& ms_rand)
  {
    // PRNG setup
    StateType* d_states;
    cudaMalloc(&d_states, N * sizeof(StateType));

    // Launch setup
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    if constexpr (std::is_same_v<StateType, curandStateXORWOW_t>) {
      util::setupXorwowStates<<<blocks, threads>>>(
        util::NonNull{d_states}, seed, N);
    } else if constexpr (std::is_same_v<StateType,
                                        curandStatePhilox4_32_10_t>) {
      util::setupPhiloxStates<<<blocks, threads>>>(
        util::NonNull{d_states}, seed, N);
    }
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_rand, t0, t1);
    cudaDeviceSynchronize();
    return d_states;
  }

  /// Specialized initialization for Sobol states.
  inline curandStateSobol32_t*
  initializeSobolStates(int N, float& ms_rand)
  {
    curandStateSobol32_t* d_states;
    cudaMalloc(&d_states, N * sizeof(curandStateSobol32_t));

    // Get direction vectors on host
    curandDirectionVectors32_t* h_vectors;
    curandGetDirectionVectors32(&h_vectors,
                                CURAND_DIRECTION_VECTORS_32_JOEKUO6);

    // Copy to device (Sobol requires these on device for init)
    unsigned int* d_vectors;
    cudaMalloc(&d_vectors, 32 * sizeof(unsigned int));
    cudaMemcpy(
      d_vectors, h_vectors, 32 * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Launch setup
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    util::setupSobolStates<<<blocks, threads>>>(
      util::NonNull{d_states}, d_vectors, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_rand, t0, t1);
    cudaDeviceSynchronize();

    // Clean up temporary device vectors
    cudaFree(d_vectors);

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

  /// Device helpers for trajectory termination.
  /// All functions are __forceinline__ so they compile to the same PTX as the
  /// equivalent inlined code — zero call overhead.
  namespace termination {

    /// Checks the global stop flag set by a peer that already converged.
    /// Returns true if the trajectory should terminate (status 2).
    ///
    /// atomicAdd(&ctx->stopFlag, 0) is a strong acquire-barrier: once any
    /// thread writes 1 via atomicExch, subsequent reads observe 1.
    template <int DIM, bool SaveTrajectories, typename Function>
    __device__ __forceinline__ bool
    checkStopFlag(util::BFGSContext* ctx,
                  zeus::Result<DIM>& r,
                  const Function& f,
                  const std::array<double, DIM>& x_arr,
                  const std::array<double, DIM>& g_arr,
                  int iter,
                  int id,
                  int N,
                  int8_t* deviceStatus)
    {
      if (atomicAdd(&ctx->stopFlag, 0) != 0) {
        writeResult<DIM>(r,
                         2,
                         f(x_arr),
                         x_arr.data(),
                         util::calculateGradientNorm<DIM>(g_arr),
                         iter,
                         id);
        if constexpr (SaveTrajectories) {
          deviceStatus[iter * N + id] = 2;
        }
        return true;
      }
      return false;
    }

    /// Checks for a non-finite gradient norm or function value.
    /// Returns true if the trajectory should terminate (status 5).
    template <int DIM, bool SaveTrajectories>
    __device__ __forceinline__ bool
    checkNonFinite(zeus::Result<DIM>& r,
                   double grad_norm,
                   double fnew,
                   const double* x_arr_data,
                   int iter,
                   int id,
                   int N,
                   int8_t* deviceStatus)
    {
      if (!isfinite(grad_norm) || !isfinite(fnew)) {
        writeResult<DIM>(r, 5, fnew, x_arr_data, grad_norm, iter, id);
        if constexpr (SaveTrajectories) {
          deviceStatus[iter * N + id] = 5;
        }
        return true;
      }
      return false;
    }

    /// Checks convergence (gradient norm below tolerance).
    /// Atomically increments ctx->convergedCount; sets ctx->stopFlag and
    /// fences when requiredConverged is reached so peers exit promptly.
    /// Returns true if this trajectory has converged (status 1).
    template <int DIM, bool SaveTrajectories, typename Function>
    __device__ __forceinline__ bool
    checkConvergence(util::BFGSContext* ctx,
                     zeus::Result<DIM>& r,
                     const Function& f,
                     const std::array<double, DIM>& x_arr,
                     double grad_norm,
                     double tolerance,
                     int requiredConverged,
                     int iter,
                     int id,
                     int N,
                     int8_t* deviceStatus)
    {
      if (grad_norm < tolerance) {
        const int oldCount = atomicAdd(&ctx->convergedCount, 1);
        const double fcurr = f(x_arr);
        writeResult<DIM>(r, 1, fcurr, x_arr.data(), grad_norm, iter, id);
        if constexpr (SaveTrajectories) {
          deviceStatus[iter * N + id] = 1;
        }
        if (oldCount + 1 == requiredConverged) {
          atomicExch(&ctx->stopFlag, 1);
          __threadfence();
          printf("\nThread %d is the %d%s converged thread (iter=%d); fn = "
                 "%.6f.\n",
                 id,
                 oldCount + 1,
                 (oldCount + 1 == 1 ? "st" :
                  oldCount + 1 == 2 ? "nd" :
                  oldCount + 1 == 3 ? "rd" :
                                      "th"),
                 iter,
                 fcurr);
        }
        return true;
      }
      return false;
    }

    /// Writes a max-iterations result when the loop ran to completion.
    /// done should be true if the loop exited early via break.
    /// Status 0: maximum iterations reached without convergence.
    template <int DIM, bool SaveTrajectories, typename Function>
    __device__ __forceinline__ void
    checkMaxIter(zeus::Result<DIM>& r,
                 const Function& f,
                 const std::array<double, DIM>& x_arr,
                 const std::array<double, DIM>& g_arr,
                 bool done,
                 int iter,
                 int MAX_ITER,
                 int id,
                 int N,
                 int8_t* deviceStatus)
    {
      if (!done && iter == MAX_ITER) {
        writeResult<DIM>(r,
                         0,
                         f(x_arr),
                         x_arr.data(),
                         util::calculateGradientNorm<DIM>(g_arr),
                         iter,
                         id);
        if constexpr (SaveTrajectories) {
          deviceStatus[(MAX_ITER - 1) * N + id] = 0;
        }
      }
    }

  } // namespace termination

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
