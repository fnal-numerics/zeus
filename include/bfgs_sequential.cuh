#pragma once

#include <cuda_runtime.h>

#include "duals.cuh"
#include "utils.cuh"
#include "pso.cuh"
#include "traits.hpp"
#include "bfgs_common.cuh"

namespace bfgs {
  namespace sequential {
    template <typename Function, int DIM, unsigned int blockSize>
    __global__ void
    optimize(Function f,
             const double lower,
             const double upper,
             const double* pso_array, // pso initialized positions (optional)
             util::non_null<double*> deviceResults,
             double* deviceTrajectory,
             int N,
             const int MAX_ITER,
             const int requiredConverged,
             const double tolerance,
             util::non_null<Result<DIM>*> result,
             util::non_null<curandState*> states,
             util::non_null<util::BFGSContext*> ctx,
             bool save_trajectories = false,
             unsigned long long* ad_cycles_out = nullptr,
             int* ad_calls_out = nullptr,
             unsigned long long* bfgs_cycles_out = nullptr,
             int* bfgs_calls_out = nullptr,
             unsigned long long* total_cycles_out = nullptr)
    {
      static_assert(
        std::is_same_v<decltype(std::declval<Function>()(
                         std::declval<std::array<dual::DualNumber, DIM>>())),
                       dual::DualNumber>,
        "\n\n> This objective is not templated.\nMake it\n\ttemplate<class T> "
        "T "
        "fun(const std::array<T,N>) { ... }\n");
      int idx = blockIdx.x * blockDim.x + threadIdx.x;
      if (idx >= N)
        return;

      unsigned long long k_begin = clock64();

      curandState localState = states[idx];
      std::array<double, DIM> x_arr, x_new, g_arr, g_new, p_arr;
      DeviceMatrix<double> H(DIM, DIM);
      DeviceMatrix<double> Htmp(DIM, DIM);
      double delta_x[DIM], delta_g[DIM];
      Result<DIM> r;
      for (int i = 0; i < DIM; i++)
        x_arr[i] = 7.0;
      write_result<DIM>(r,
                        /*status=*/-1,
                        /*fval=*/333777.0,
                        /*coordinates=*/x_arr.data(),
                        /*norm=*/69.0,
                        /*iter*/ 0,
                        idx);
      util::initialize_identity_matrix(&H, DIM);
      int num_steps = 0, iter;
      double x_raw[DIM];
      // initialize x either from PSO array or fallback by RNG
#pragma unroll
      for (int d = 0; d < DIM; ++d) {
        x_raw[d] = pso_array ?
                     pso_array[idx * DIM + d] :
                     util::generate_random_double(&states[idx], lower, upper);
        x_arr[d] = x_raw[d];
        g_arr[d] = 0.0;
        states[idx] = localState;
      }

      double f0 = f(x_arr); // rosenbrock_device(x, DIM);
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
        // printf("inside BeeG File System");
        //  check if somebody already asked to stop
        if (atomicAdd(&ctx->stopFlag, 0) !=
            0) { // atomicAdd here just to get a strong read-barrier
          // CUDA will fetch a coherent copy of the integer from global memory.
          // as soon as one thread writes 1 into d_stopFlag via atomicExch,
          // the next time any thread does atomicAdd(&d_stopFlag, 0) it’ll see 1
          // and break. printf("thread %d get outta dodge cuz we converged...",
          // idx);
          write_result<DIM>(r,
                            2,
                            f(x_arr),
                            x_arr.data(),
                            util::calculate_gradient_norm<DIM>(g_arr),
                            iter,
                            idx);
          break;
        }
        num_steps++;
        util::compute_search_direction<DIM>(p_arr, &H, g_arr); // p = -H * g

        // use the alpha obtained from the line search
        double alpha =
          util::line_search<Function, DIM>(bestVal, x_arr, p_arr, g_arr, f);
        if (alpha == 0.0) {
          printf("Alpha is zero, no movement in iteration=%d\n", iter);
          alpha = 1e-3;
        }

        // update current point by taking a step size of alpha in the direction
        // p
        for (int i = 0; i < DIM; ++i) {
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
        for (int i = 0; i < DIM; ++i) {
          delta_g[i] =
            g_new[i] -
            g_arr[i]; // difference in gradient at the new point vs old point
        }

        // calculate the the dot product between the change in x and change in
        // gradient using new point
        double delta_dot = util::dot_product_device(delta_x, delta_g, DIM);

        unsigned long long b0 = clock64();
        // bfgs update on H
        util::bfgs_update<DIM>(&H, delta_x, delta_g, delta_dot, &Htmp);
        // only update x and g for next iteration if the new minima is smaller
        // than previous double min =
        unsigned long long b1 = clock64();
        bfgs_cycles += (b1 - b0);
        bfgs_calls += 1;

        if (fnew < bestVal) {
          bestVal = fnew;
          for (int i = 0; i < DIM; ++i) {
            x_arr[i] = x_new[i];
            g_arr[i] = g_new[i];
          }
        }
        // refactor? yes
        double grad_norm = util::calculate_gradient_norm<DIM>(g_arr);
        // catch not finite gradient norm or function value
        if (!isfinite(grad_norm) || !isfinite(fnew)) {
          write_result<DIM>(r, 5, fnew, x_arr.data(), grad_norm, iter, idx);
          break;
        }
        if (grad_norm < tolerance) {
          // atomically increment the converged counter
          int oldCount = atomicAdd(&ctx->convergedCount, 1);
          int newCount = oldCount + 1;
          double fcurr = f(x_arr);
          write_result<DIM>(r, 1, fcurr, x_arr.data(), grad_norm, iter, idx);
          // if we just hit the threshold set by the user, the VERY FIRST thread
          // to do so sets ctx->stopFlag=1 so everyone else exits on their next
          // check
          if (newCount == requiredConverged) {
            // flip the global stop flag
            atomicExch(&ctx->stopFlag, 1);
            __threadfence();
            printf("\nThread %d is the %d%s converged thread (iter=%d); fn = "
                   "%.6f.\n",
                   idx,
                   newCount,
                   (newCount == 1 ? "st" :
                    newCount == 2 ? "nd" :
                    newCount == 3 ? "rd" :
                                    "th"),
                   iter,
                   fcurr);
          }
          // in _any_ case, whether we were the last to converge or not,
          // we are individually done so break
          break;
        }

        /*  deviceTrajectory layout: idx * (MAX_ITER * DIM) + iter * DIM + i
        if (save_trajectories) {
          for (int i = 0; i < DIM; i++) {
            deviceTrajectory[idx * (MAX_ITER * DIM) + iter * DIM + i] =
        x_raw[i];
          }
        }*/

      } // end bfgs loop
      // if we broek out because we hit the max numberof iterations, then its a
      // surrender
      if (MAX_ITER == iter) {
        write_result<DIM>(r,
                          0,
                          f(x_arr),
                          x_arr.data(),
                          util::calculate_gradient_norm<DIM>(g_arr),
                          iter,
                          idx);
      }
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
              std::size_t DIM = zeus::fn_traits<Function>::arity>
    Result<DIM>
    launch(size_t N,
           const int pso_iter,
           const int MAX_ITER,
           const double upper,
           const double lower,
           double* pso_results_device,
           double* deviceTrajectory,
           const int requiredConverged,
           const double tolerance,
           bool save_trajectories,
           float& ms_opt,
           std::string fun_name,
           curandState* states,
           const int run,
           Function f)
    {
      int blockSize, minGridSize;

      cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, optimize<Function, DIM, 128>, 0, N);
      // printf("\nRecommended block size: %d\n", blockSize);
      dbuf deviceResults;
      try {
        deviceResults = dbuf(N);
      }
      catch (const cuda_exception<3>& e) {
        Result<DIM> result;
        result.status = 3;
        return result;
        // throw cuda_exception<3>(std::string("bfgs::launch: failed to
        // cudaMalloc deviceResults (") + e.what() + ")\n"); //allocation failed
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

      result_buffer<DIM> d_results;
      try {
        d_results = result_buffer<DIM>(N);
      }
      catch (const cuda_exception<3>& e) {
        Result<DIM> result;
        result.status = 3;
        return result;
        // throw; // allocation failed
      }
      if (save_trajectories) {
        optimize<Function, DIM, 128>
          <<<optGrid, optBlock>>>(f,
                                  lower,
                                  upper,
                                  pso_results_device,
                                  util::non_null{deviceResults.data()},
                                  deviceTrajectory,
                                  (int)N,
                                  MAX_ITER,
                                  requiredConverged,
                                  tolerance,
                                  util::non_null{d_results.data()},
                                  util::non_null{states},
                                  util::non_null{d_ctx},
                                  true,
                                  d_ad_cycles,
                                  d_ad_calls,
                                  d_bfgs_cycles,
                                  d_bfgs_calls,
                                  d_total_cycles);
      } else {
        optimize<Function, DIM, 128>
          <<<optGrid, optBlock>>>(f,
                                  lower,
                                  upper,
                                  pso_results_device,
                                  util::non_null{deviceResults.data()},
                                  nullptr,
                                  (int)N,
                                  MAX_ITER,
                                  requiredConverged,
                                  tolerance,
                                  util::non_null{d_results.data()},
                                  util::non_null{states},
                                  util::non_null{d_ctx},
                                  false,
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
        Result<DIM> result;
        result.status = 4;
        return result;
      }
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::fprintf(
          stderr, "bfgs kernel runtime error: %s\n", cudaGetErrorString(err));
        Result<DIM> result;
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

      std::vector<Result<DIM>> h_results(N);
      cudaMemcpy(h_results.data(),
                 d_results,
                 N * sizeof(Result<DIM>),
                 cudaMemcpyDeviceToHost);
      Convergence c = util::dump_data_2_file<DIM>(
        N, h_results.data(), fun_name, pso_iter, run);

      Result best =
        launch_reduction<DIM>(N, deviceResults.data(), h_results.data());
      best.c = c;
      best.ad = ad_metrics;
      best.bfgs = bfgs_metrics;
      return best;
    }

  } // end sequential namespace
} // namespace bfgs
