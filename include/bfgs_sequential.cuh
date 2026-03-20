#pragma once

#include <cuda_runtime.h>

#include "duals.cuh"
#include "utils.cuh"
#include "pso.cuh"
#include "traits.hpp"
#include "bfgs_common.cuh"

namespace bfgs {
  namespace sequential {
    /// Sequential BFGS optimization kernel.
    /// Each thread performs one independent BFGS optimization using
    /// automatic differentiation for gradient computation.
    template <typename Function,
              std::size_t ZEUS_DIM = zeus::FnTraits<Function>::arity,
              unsigned int blockSize = 128,
              bool SaveTrajectories = false,
              typename StateType = curandState>
      requires zeus::ZeusObjective<Function, ZEUS_DIM>
    __global__ void
    optimize(Function f,
             const double lower,
             const double upper,
             const double* pso_array, // pso initialized positions (optional)
             util::NonNull<double*> deviceResults,
             double* deviceTrajectoryCoords,
             double* deviceTrajectoryFval,
             double* deviceTrajectoryGrad,
             int8_t* deviceStatus,
             int N,
             const int MAX_ITER,
             const int requiredConverged,
             const double tolerance,
             const int nzerosteps,
             util::NonNull<zeus::Result<ZEUS_DIM>*> result,
             util::NonNull<StateType*> states,
             util::NonNull<util::BFGSContext*> ctx,
             unsigned long long* ad_cycles_out = nullptr,
             int* ad_calls_out = nullptr,
             unsigned long long* bfgs_cycles_out = nullptr,
             int* bfgs_calls_out = nullptr,
             unsigned long long* total_cycles_out = nullptr)
    {
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= N)
        return;

      unsigned long long k_begin = clock64();

      StateType localState = states[idx];
      std::array<double, ZEUS_DIM> x_arr, x_new, g_arr, g_new, p_arr;
      DeviceMatrix<double> H(ZEUS_DIM, ZEUS_DIM);
      DeviceMatrix<double> Htmp(ZEUS_DIM, ZEUS_DIM);
      double delta_x[ZEUS_DIM], delta_g[ZEUS_DIM];
      zeus::Result<ZEUS_DIM> r;
      for (int i = 0; i < ZEUS_DIM; i++)
        x_arr[i] = 7.0;
      writeResult<ZEUS_DIM>(r,
                            /*status=*/-1,
                            /*fval=*/333777.0,
                            /*coordinates=*/x_arr.data(),
                            /*norm=*/69.0,
                            /*iter*/ 0,
                            idx);
      util::initializeIdentityMatrix(&H, ZEUS_DIM);
      int num_steps = 0, iter;
      int consecutive_zero_steps = 0;
      double x_raw[ZEUS_DIM];
      // initialize x either from PSO array or fallback by RNG
#pragma unroll
      for (int d = 0; d < ZEUS_DIM; ++d) {
        x_raw[d] = pso_array ?
                     pso_array[idx * ZEUS_DIM + d] :
                     util::generateRandomDouble(&localState, lower, upper);
        x_arr[d] = x_raw[d];
        g_arr[d] = 0.0;
      }
      states[idx] = localState;

      double f0 = f(x_arr); // rosenbrock_device(x, ZEUS_DIM);
      deviceResults[idx] = f0;
      double bestVal = f0;

      unsigned long long ad_cycles = 0ULL;
      int ad_calls = 0;

      unsigned long long bfgs_cycles = 0ULL;
      int bfgs_calls = 0;

      // first gradient
      unsigned long long t0 = clock64();
      dual::calculateGradientUsingAD(f, x_arr, g_arr);
      unsigned long long t1 = clock64();
      ad_cycles += (t1 - t0);
      ad_calls += 1;

      for (iter = 0; iter < MAX_ITER; ++iter) {
        if constexpr (SaveTrajectories) {
          for (int d = 0; d < ZEUS_DIM; ++d) {
            deviceTrajectoryCoords[iter * ZEUS_DIM * N + d * N + idx] =
              x_arr[d];
          }
          deviceTrajectoryFval[iter * N + idx] = bestVal;
          deviceTrajectoryGrad[iter * N + idx] =
            util::calculateGradientNorm<ZEUS_DIM>(g_arr);
          deviceStatus[iter * N + idx] = -1;
        }
        // printf("inside BeeG File System");
        //  check if somebody already asked to stop
        // check if somebody already asked to stop
        if (termination::checkStopFlag<ZEUS_DIM, SaveTrajectories>(
              ctx, r, f, x_arr, g_arr, iter, idx, N, deviceStatus))
          break;
        num_steps++;
        util::computeSearchDirection<ZEUS_DIM>(p_arr, &H, g_arr); // p = -H * g

        // use the alpha obtained from the line search
        double alpha =
          util::lineSearch<Function, ZEUS_DIM>(bestVal, x_arr, p_arr, g_arr, f);
        if (alpha == 0.0) {
          if (termination::checkZeroSteps<ZEUS_DIM, SaveTrajectories>(
                r, consecutive_zero_steps, nzerosteps, f, x_arr, g_arr,
                iter, idx, N, deviceStatus))
            break;
          alpha = 1e-3;
        } else {
          consecutive_zero_steps = 0;
        }

        // update current point by taking a step size of alpha in the direction
        // p
        for (int i = 0; i < ZEUS_DIM; ++i) {
          x_new[i] = x_arr[i] + alpha * p_arr[i];
          delta_x[i] = x_new[i] - x_arr[i];
        }

        double fnew = f(x_new);
        //  get the new gradient g_new at x_new
        t0 = clock64();
        dual::calculateGradientUsingAD(f, x_new, g_new);
        t1 = clock64();
        ad_cycles += (t1 - t0);
        ad_calls += 1;

        // calculate new delta_x and delta_g
        for (int i = 0; i < ZEUS_DIM; ++i) {
          delta_g[i] =
            g_new[i] -
            g_arr[i]; // difference in gradient at the new point vs old point
        }

        // calculate the the dot product between the change in x and change in
        // gradient using new point
        double delta_dot = util::dotProductDevice(delta_x, delta_g, ZEUS_DIM);

        unsigned long long b0 = clock64();
        // bfgs update on H
        util::bfgsUpdate<ZEUS_DIM>(&H, delta_x, delta_g, delta_dot, &Htmp);
        // only update x and g for next iteration if the new minima is smaller
        // than previous double min =
        unsigned long long b1 = clock64();
        bfgs_cycles += (b1 - b0);
        bfgs_calls += 1;

        if (fnew < bestVal) {
          bestVal = fnew;
          for (int i = 0; i < ZEUS_DIM; ++i) {
            x_arr[i] = x_new[i];
            g_arr[i] = g_new[i];
          }
        }
        double grad_norm = util::calculateGradientNorm<ZEUS_DIM>(g_arr);
        if (termination::checkNonFinite<ZEUS_DIM, SaveTrajectories>(
              r, grad_norm, fnew, x_arr.data(), iter, idx, N, deviceStatus))
          break;
        if (termination::checkConvergence<ZEUS_DIM, SaveTrajectories>(
              ctx, r, f, x_arr, grad_norm, tolerance, requiredConverged,
              iter, idx, N, deviceStatus))
          break;

        /*  deviceTrajectory layout: idx * (MAX_ITER * ZEUS_DIM) + iter *
        ZEUS_DIM + i if (save_trajectories) { for (int i = 0; i < ZEUS_DIM; i++)
        { deviceTrajectory[idx * (MAX_ITER * ZEUS_DIM) + iter * ZEUS_DIM + i] =
        x_raw[i];
          }
        }*/

      } // end bfgs loop
      // if we exited by exhausting iterations, record the surrender result
      termination::checkMaxIter<ZEUS_DIM, SaveTrajectories>(
        r, f, x_arr, g_arr, /*done=*/(iter < MAX_ITER), iter, MAX_ITER,
        idx, N, deviceStatus);
      deviceResults[idx] = r.fval;
      result[idx] = r;
      if (ad_cycles_out)
        ad_cycles_out[idx] = ad_cycles;
      if (ad_calls_out)
        ad_calls_out[idx] = ad_calls;
      if (bfgs_cycles_out)
        bfgs_cycles_out[idx] = bfgs_cycles;
      if (bfgs_calls_out)
        bfgs_calls_out[idx] = bfgs_calls;
      unsigned long long k_end = clock64();
      if (total_cycles_out)
        total_cycles_out[idx] = (k_end - k_begin);
      // if (ad_calls_out) ad_calls_out[idx] = 123;
    } // end optimizerKernel

    inline Metrics
    /// Report performance metrics and cleanup device memory.
    /// Computes timing statistics for AD or BFGS operations and frees
    /// the provided device buffers.
    report_metrics_and_cleanup(
      const char* label,
      int N,
      unsigned long long* d_cycles, // pass AD or BFGS cycles
      int* d_calls,                 // pass AD or BFGS calls
      float ms_opt,
      int device_id,
      dim3 gridDimUsed,
      dim3 blockDimUsed,
      const unsigned long long* d_total_cycles = nullptr)
    {
      Metrics out;

      // Copy back the arrays the caller passed in (AD or BFGS).
      std::vector<unsigned long long> h_cycles(N);
      std::vector<int> h_calls(N);
      cudaMemcpy(h_cycles.data(),
                 d_cycles,
                 N * sizeof(unsigned long long),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(
        h_calls.data(), d_calls, N * sizeof(int), cudaMemcpyDeviceToHost);

      // Compute stats (identical math for either AD or BFGS).
      cudaDeviceProp prop{};
      cudaGetDeviceProperties(&prop, device_id);
      const double sm_clock_hz = static_cast<double>(prop.clockRate) * 1000.0;

      unsigned long long sum_cycles = 0ULL;
      long long sum_calls = 0LL;
      for (int i = 0; i < N; ++i) {
        sum_cycles += h_cycles[i];
        sum_calls += h_calls[i];
      }

      const double avg_cycles_per_thread = double(sum_cycles) / double(N);
      const double avg_calls_per_thread =
        (sum_calls > 0 ? double(sum_calls) / double(N) : 0.0);
      const double avg_cycles_per_call =
        (avg_calls_per_thread > 0 ?
           avg_cycles_per_thread / avg_calls_per_thread :
           0.0);

      const double avg_ms_per_thread =
        (avg_cycles_per_thread / sm_clock_hz) * 1e3;
      const double avg_ms_per_call = (avg_cycles_per_call / sm_clock_hz) * 1e3;

      const double frac_of_total = (avg_ms_per_thread > 0.0 && ms_opt > 0.0) ?
                                     (avg_ms_per_thread / ms_opt) :
                                     0.0;

      const double est_total_ms_serialized =
        (double)sum_cycles / sm_clock_hz * 1e3;
      const double frac_serialized =
        (ms_opt > 0.0) ? (est_total_ms_serialized / ms_opt) : 0.0;

      // averaged per-thread fraction cycles_i / total_cycles_i
      double fraction_of_thread_mean = 0.0;
      if (d_total_cycles) {
        std::vector<unsigned long long> h_total(N);
        cudaMemcpy(h_total.data(),
                   d_total_cycles,
                   N * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost);

        long double acc = 0.0L;
        long long cnt = 0;
        for (int i = 0; i < N; ++i) {
          const double tot = (double)h_total[i];
          if (tot > 0.0) {
            acc += (long double)((double)h_cycles[i] / tot);
            ++cnt;
          }
        }
        fraction_of_thread_mean =
          (cnt > 0) ? (double)(acc / (long double)cnt) : 0.0;
      }

      // per-block aggregation (active blocks only) for p95
      double block95_fraction_local = 0.0;
      const long long threads_per_block =
        1LL * blockDimUsed.x * blockDimUsed.y * blockDimUsed.z;
      const long long num_blocks_active =
        (threads_per_block > 0) ?
          ((static_cast<long long>(N) + threads_per_block - 1) /
           threads_per_block) :
          0;

      if (threads_per_block > 0 && num_blocks_active > 0) {
        std::vector<long double> block_cycles((size_t)num_blocks_active, 0.0L);
        for (int tid = 0; tid < N; ++tid) {
          const long long b = tid / threads_per_block;
          block_cycles[(size_t)b] += (long double)h_cycles[tid];
        }
        std::vector<double> f_block;
        f_block.reserve((size_t)num_blocks_active);
        for (long long b = 0; b < num_blocks_active; ++b) {
          const long double t_ms =
            (block_cycles[(size_t)b] / sm_clock_hz) * 1e3L;
          f_block.push_back((double)t_ms / (double)ms_opt);
        }
        if (!f_block.empty()) {
          std::sort(f_block.begin(), f_block.end());
          const size_t idx95 = std::min<size_t>(
            f_block.size() - 1, (size_t)std::ceil(0.95 * f_block.size()) - 1);
          block95_fraction_local = f_block[idx95];
        }
      }

      // print with label
      std::printf("[%s] Kernel wall time: %.3f ms\n", label, ms_opt);
      std::printf("[%s] Avg cycles/thread: %.3f ; avg ms/thread: %.6f\n",
                  label,
                  avg_cycles_per_thread,
                  avg_ms_per_thread);
      std::printf("[%s] Calls/thread ≈ %.2f; ms/call ≈ %.6f\n",
                  label,
                  avg_calls_per_thread,
                  avg_ms_per_call);
      std::printf(
        "[%s] Heuristic fraction: %.3f%%\n", label, 100.0 * frac_of_total);
      std::printf("[%s] Serialized sum: %.3f ms; fraction vs kernel: %.3f%%\n",
                  label,
                  est_total_ms_serialized,
                  100.0 * frac_serialized);
      std::printf("[%s] Block-level fraction p95: %.3f%%\n",
                  label,
                  100.0 * block95_fraction_local);

      // fill outputs
      out.ms_per_call = avg_ms_per_call;
      out.calls_per_thread_mean = avg_calls_per_thread;
      out.fraction_of_kernel = frac_of_total;
      out.block95 = block95_fraction_local;
      out.serialized = frac_serialized;
      out.fraction_of_thread = fraction_of_thread_mean;

      // cleanup the device buffers we were given
      cudaFree(d_cycles);
      cudaFree(d_calls);

      return out;
    }

    template <typename Function,
              std::size_t ZEUS_DIM = zeus::FnTraits<Function>::arity,
              typename StateType = curandState>
    zeus::Result<ZEUS_DIM>
    launch(size_t N,
           const int pso_iter,
           const int MAX_ITER,
           const double lower,
           const double upper,
           double* pso_results_device,
           double* deviceTrajectoryCoords,
           double* deviceTrajectoryFval,
           double* deviceTrajectoryGrad,
           int8_t* deviceStatus,
           const int requiredConverged,
           const double tolerance,
           const int nzerosteps,
           bool save_trajectories,
           float& ms_opt,
           std::string fun_name,
           StateType* states,
           const int run,
           Function f)
    {
      int blockSize, minGridSize;

      if (save_trajectories) {
        cudaOccupancyMaxPotentialBlockSize(
          &minGridSize,
          &blockSize,
          optimize<Function, ZEUS_DIM, 128, true, StateType>,
          0,
          N);
      } else {
        cudaOccupancyMaxPotentialBlockSize(
          &minGridSize,
          &blockSize,
          optimize<Function, ZEUS_DIM, 128, false, StateType>,
          0,
          N);
      }
      // printf("\nRecommended block size: %d\n", blockSize);
      DoubleBuffer deviceResults;
      try {
        deviceResults = DoubleBuffer(N);
      }
      catch (const CudaError& e) {
        zeus::Result<ZEUS_DIM> result;
        result.status = (e.code() == cudaErrorMemoryAllocation) ? 3 : 4;
        return result;
      }
      dim3 optBlock(blockSize);
      dim3 optGrid((N + blockSize - 1) / blockSize);

      // Allocate and initialize BFGSContext
      util::BFGSContext* d_ctx;
      cudaMalloc(&d_ctx, sizeof(util::BFGSContext));
      util::BFGSContext h_ctx = {0, 0};
      cudaMemcpy(
        d_ctx, &h_ctx, sizeof(util::BFGSContext), cudaMemcpyHostToDevice);

      // metric buffers
      // ad
      unsigned long long* d_ad_cycles = nullptr;
      int* d_ad_calls = nullptr;
      cudaMalloc(&d_ad_cycles, N * sizeof(unsigned long long));
      cudaMalloc(&d_ad_calls, N * sizeof(int));
      cudaMemset(d_ad_cycles, 0, N * sizeof(unsigned long long));
      cudaMemset(d_ad_calls, 0, N * sizeof(int));
      // bfgs
      unsigned long long* d_bfgs_cycles = nullptr;
      int* d_bfgs_calls = nullptr;
      cudaMalloc(&d_bfgs_cycles, N * sizeof(unsigned long long));
      cudaMalloc(&d_bfgs_calls, N * sizeof(int));
      cudaMemset(d_bfgs_cycles, 0, N * sizeof(unsigned long long));
      cudaMemset(d_bfgs_calls, 0, N * sizeof(int));

      unsigned long long* d_total_cycles = nullptr;
      cudaMalloc(&d_total_cycles, N * sizeof(unsigned long long));
      cudaMemset(d_total_cycles, 0, N * sizeof(unsigned long long));

      // optimizeKernel time
      cudaEvent_t startOpt, stopOpt;
      cudaEventCreate(&startOpt);
      cudaEventCreate(&stopOpt);
      cudaEventRecord(startOpt);

      ResultBuffer<ZEUS_DIM> d_results;
      try {
        d_results = ResultBuffer<ZEUS_DIM>(N);
      }
      catch (const CudaError& e) {
        zeus::Result<ZEUS_DIM> result;
        result.status = (e.code() == cudaErrorMemoryAllocation) ? 3 : 4;
        return result;
      }
      if (save_trajectories) {
        optimize<Function, ZEUS_DIM, 128, true, StateType>
          <<<optGrid, optBlock>>>(f,
                                  lower,
                                  upper,
                                  pso_results_device,
                                  util::NonNull{deviceResults.data()},
                                  deviceTrajectoryCoords,
                                  deviceTrajectoryFval,
                                  deviceTrajectoryGrad,
                                  deviceStatus,
                                  (int)N,
                                  MAX_ITER,
                                  requiredConverged,
                                  tolerance,
                                  nzerosteps,
                                  util::NonNull{d_results.data()},
                                  util::NonNull{states},
                                  util::NonNull{d_ctx},
                                  d_ad_cycles,
                                  d_ad_calls,
                                  d_bfgs_cycles,
                                  d_bfgs_calls,
                                  d_total_cycles);
      } else {
        optimize<Function, ZEUS_DIM, 128, false, StateType>
          <<<optGrid, optBlock>>>(f,
                                  lower,
                                  upper,
                                  pso_results_device,
                                  util::NonNull{deviceResults.data()},
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  nullptr,
                                  (int)N,
                                  MAX_ITER,
                                  requiredConverged,
                                  tolerance,
                                  nzerosteps,
                                  util::NonNull{d_results.data()},
                                  util::NonNull{states},
                                  util::NonNull{d_ctx},
                                  d_ad_cycles,
                                  d_ad_calls,
                                  d_bfgs_cycles,
                                  d_bfgs_calls,
                                  d_total_cycles);
      }
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::fprintf(
          stderr, "BFGS kernel launch failed: %s\n", cudaGetErrorString(err));
        zeus::Result<ZEUS_DIM> result;
        result.status = 4;
        return result;
      }
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::fprintf(
          stderr, "bfgs kernel runtime error: %s\n", cudaGetErrorString(err));
        zeus::Result<ZEUS_DIM> result;
        result.status = 4;
        return result;
      }
      cudaEventRecord(stopOpt);
      cudaEventSynchronize(stopOpt);
      cudaEventElapsedTime(&ms_opt, startOpt, stopOpt);
      // printf("\nOptimization Kernel execution time = %.3f ms\n", ms_opt);
      cudaEventDestroy(startOpt);
      cudaEventDestroy(stopOpt);
      cudaFree(d_ctx);

      // calculate the AD timing
      int device_id = 0;
      cudaGetDevice(&device_id);
      Metrics ad_metrics = report_metrics_and_cleanup("AD",
                                                      N,
                                                      d_ad_cycles,
                                                      d_ad_calls,
                                                      ms_opt,
                                                      device_id,
                                                      optGrid,
                                                      optBlock,
                                                      d_total_cycles);
      Metrics bfgs_metrics = report_metrics_and_cleanup("BFGS",
                                                        N,
                                                        d_bfgs_cycles,
                                                        d_bfgs_calls,
                                                        ms_opt,
                                                        device_id,
                                                        optGrid,
                                                        optBlock,
                                                        d_total_cycles);

      std::vector<zeus::Result<ZEUS_DIM>> h_results(N);
      cudaMemcpy(h_results.data(),
                 d_results.data(),
                 N * sizeof(zeus::Result<ZEUS_DIM>),
                 cudaMemcpyDeviceToHost);
      Convergence c = util::dumpDataToFile<ZEUS_DIM>(
        N, h_results.data(), fun_name, pso_iter, run);

      zeus::Result<ZEUS_DIM> best =
        launchReduction<ZEUS_DIM>(N, deviceResults.data(), h_results.data());
      best.c = c;
      best.ad = ad_metrics;
      best.bfgs = bfgs_metrics;
      return best;
    }

  } // end sequential namespace
} // namespace bfgs
