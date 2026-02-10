#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "duals.cuh"
#include "utils.cuh"
#include "pso.cuh"
#include "context.cuh"
#include "traits.hpp"

using namespace zeus;

namespace bfgs {

  inline curandState*
  initialize_states(int N, int seed, float& ms_rand)
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
    util::setup_curand_states<<<blocks, threads>>>(util::non_null{d_states}, seed, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_rand, t0, t1);
    cudaDeviceSynchronize();
    return d_states;
  }

  template <int DIM>
  __device__ void
  write_result(Result<DIM>& r,
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

  template <int DIM>
  Result<DIM>
  launch_reduction(int N, double* deviceResults, Result<DIM> const* h_results)
  {
    // ArgMin & final print
    cub::KeyValuePair<int, double>* deviceArgMin;
    cudaMalloc(&deviceArgMin, sizeof(*deviceArgMin));
    void* d_temp_storage = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::ArgMin(
      d_temp_storage, temp_bytes, deviceResults, deviceArgMin, N);
    cudaMalloc(&d_temp_storage, temp_bytes);
    cub::DeviceReduce::ArgMin(
      d_temp_storage, temp_bytes, deviceResults, deviceArgMin, N);

    cub::KeyValuePair<int, double> h_argMin;
    cudaMemcpy(
      &h_argMin, deviceArgMin, sizeof(h_argMin), cudaMemcpyDeviceToHost);

    int globalMinIndex = h_argMin.key;

    // print the “best” thread’s full record
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

    cudaFree(deviceArgMin);
    cudaFree(d_temp_storage);
    return best;
  }

  namespace sequential {
    template <typename Function, int DIM, unsigned int blockSize>
    __global__ void
    optimize(
      Function f,
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

    template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
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

  namespace parallel {
    namespace cg = cooperative_groups;

    // Parallel forward-mode AD across dimensions using a tile of size TS.
    // TS must divide 32 (so tiles don’t cross warp boundaries).
    template <class Function, int DIM, int TS>
    __device__ void
    grad_ad_tile(const Function& f,
                 const double* __restrict__ x_shared,
                 double* __restrict__ g_out,
                 cg::thread_block_tile<TS> tile,
                 double* __restrict__ scratch)
    {
      static_assert(32 % TS == 0, "TS must divide 32");

      // stream over dimensions in TS-sized chunks
      for (int base = 0; base < DIM; base += TS) {
        const int i = base + tile.thread_rank();
        if (i < DIM) {
          // std::array, not a C array
          std::array<dual::DualNumber, DIM> xDual;

#pragma unroll
          for (int j = 0; j < DIM; ++j) {
            xDual[j].real = x_shared[j];
            xDual[j].dual = 0.0;
          }
          xDual[i].dual = 1.0;

          // f(std::array<dual::DualNumber, DIM>) -> dual::DualNumber
          auto y = f(xDual);
          scratch[i] = y.dual; // delf/delx_i
        }
        tile.sync();
      }

      if (tile.thread_rank() == 0) {
#pragma unroll
        for (int j = 0; j < DIM; ++j)
          g_out[j] = scratch[j];
      }
      tile.sync();
    }

    // match the kernel's chooser
    template <int DIM>
    struct tile_size {
      static constexpr int value = (DIM >= 32 ? 32 :
                                    DIM >= 16 ? 16 :
                                    DIM >= 8  ? 8 :
                                    DIM >= 4  ? 4 :
                                    DIM >= 2  ? 2 :
                                                1);
    };

    // kernel for one tile = one BFGS optimization
    template <typename Function,
              std::size_t DIM = fn_traits<Function>::arity,
              int TS>
    __global__ void
    optimize_tiles(Function f,
                   const double lower,
                   const double upper,
                   const double* pso_array,
                   util::non_null<double*> deviceResults,
                   double* deviceTrajectory,
                   int N,
                   const int MAX_ITER,
                   const int requiredConverged,
                   const double tolerance,
                   util::non_null<Result<DIM>*> result,
                   util::non_null<curandState*> states,
                   util::non_null<util::BFGSContext*> ctx,
                   bool save_trajectories = false)
    {

      static_assert(
        std::is_same_v<decltype(std::declval<Function>()(
                         std::declval<std::array<dual::DualNumber, DIM>>())),
                       dual::DualNumber>,
        "Objective must be templated: template<class T> T f(const "
        "std::array<T,DIM>&)");

      // partition the block into tiles of size TS
      auto block = cg::this_thread_block();
      auto tile = cg::tiled_partition<TS>(block);

      // each tile owns one optimization index
      const int tiles_per_block = blockDim.x / TS;
      const int tile_local_id = threadIdx.x / TS;
      const int tile_global_id = blockIdx.x * tiles_per_block + tile_local_id;
      if (tile_global_id >= N)
        return;

      // per-tile shared memory layout:
      // [ x_shared (DIM) | grad_scratch (DIM) | done_flag (1 as int) ]
      extern __shared__ double smem[];
      const int per_tile_doubles =
        2 * DIM + 1; // last double used as an int flag
      double* base_ptr = smem + tile_local_id * per_tile_doubles;
      double* x_shared = base_ptr;
      double* grad_scratch = base_ptr + DIM;
      int* done_flag = reinterpret_cast<int*>(base_ptr + 2 * DIM);

      // Lane-0 state
      std::array<double, DIM> x_arr, x_new, g_arr, g_new, p_arr;
      DeviceMatrix<double> H(DIM, DIM), Htmp(DIM, DIM);
      Result<DIM> r;

      // init
      if (tile.thread_rank() == 0) {
        *done_flag = 0;
        util::initialize_identity_matrix(&H, DIM);

        curandState localState = states[tile_global_id];
        for (int d = 0; d < DIM; ++d) {
          double v = pso_array ?
                       pso_array[tile_global_id * DIM + d] :
                       util::generate_random_double(&localState, lower, upper);
          x_arr[d] = v;
          g_arr[d] = 0.0;
          x_shared[d] = v; // publish to tile
        }
        states[tile_global_id] = localState;
      }
      tile.sync();

      // first gradient at x_arr
      grad_ad_tile<Function, DIM, TS>(
        f, x_shared, grad_scratch, tile, grad_scratch);
      if (tile.thread_rank() == 0) {
        for (int d = 0; d < DIM; ++d)
          g_arr[d] = grad_scratch[d];
      }
      tile.sync();

      double bestVal = 0.0;
      if (tile.thread_rank() == 0)
        bestVal = f(x_arr);

      int iter = 0;
      for (; iter < MAX_ITER; ++iter) {
        // Global stop check
        if (tile.thread_rank() == 0) {
          if (atomicAdd(&ctx->stopFlag, 0) != 0) {
            write_result<DIM>(r,
                              2,
                              f(x_arr),
                              x_arr.data(),
                              util::calculate_gradient_norm<DIM>(g_arr),
                              iter,
                              tile_global_id);
            *done_flag = 1;
          }
        }
        tile.sync();
        if (*done_flag)
          break;

        // Lane 0: compute search direction & step
        if (tile.thread_rank() == 0) {
          util::compute_search_direction<DIM>(p_arr, &H, g_arr); // p = -H g
          double alpha =
            util::line_search<Function, DIM>(bestVal, x_arr, p_arr, g_arr, f);
          if (alpha == 0.0) {
            printf(
              "Alpha is zero, using fallback alpha=1e-3 (iter=%d, idx=%d)\n",
              iter,
              tile_global_id);
            alpha = 1e-3;
          }
          for (int d = 0; d < DIM; ++d) {
            x_new[d] = x_arr[d] + alpha * p_arr[d];
            x_shared[d] = x_new[d]; // publish to tile for gradient
          }
        }
        tile.sync();

        // New gradient at x_new
        grad_ad_tile<Function, DIM, TS>(
          f, x_shared, grad_scratch, tile, grad_scratch);
        if (tile.thread_rank() == 0) {
          for (int d = 0; d < DIM; ++d)
            g_new[d] = grad_scratch[d];

          // BFGS update
          double delta_x[DIM], delta_g[DIM];
          for (int d = 0; d < DIM; ++d) {
            delta_x[d] = x_new[d] - x_arr[d];
            delta_g[d] = g_new[d] - g_arr[d];
          }

          const double fnew = f(x_new);
          const double delta_dot =
            util::dot_product_device(delta_x, delta_g, DIM);
          util::bfgs_update<DIM>(&H, delta_x, delta_g, delta_dot, &Htmp);

          if (fnew < bestVal) {
            bestVal = fnew;
            for (int d = 0; d < DIM; ++d) {
              x_arr[d] = x_new[d];
              g_arr[d] = g_new[d];
            }
          }

          const double grad_norm = util::calculate_gradient_norm<DIM>(g_arr);
          if (!isfinite(grad_norm) || !isfinite(fnew)) {
            write_result<DIM>(
              r, 5, fnew, x_arr.data(), grad_norm, iter, tile_global_id);
            *done_flag = 1;
          } else if (grad_norm < tolerance) {
            const int oldCount = atomicAdd(&ctx->convergedCount, 1);
            const double fcurr = f(x_arr);
            write_result<DIM>(
              r, 1, fcurr, x_arr.data(), grad_norm, iter, tile_global_id);
            if (oldCount + 1 == requiredConverged) {
              atomicExch(&ctx->stopFlag, 1);
              __threadfence();
            }
            *done_flag = 1;
          }
        }
        tile.sync();
        if (*done_flag)
          break;

        // save_trajectories could go here; omitted for brevity
      }

      // Max-iters surrender
      if (tile.thread_rank() == 0) {
        if (!*done_flag && iter == MAX_ITER) {
          write_result<DIM>(r,
                            0,
                            f(x_arr),
                            x_arr.data(),
                            util::calculate_gradient_norm<DIM>(g_arr),
                            iter,
                            tile_global_id);
        }
        deviceResults[tile_global_id] = r.fval;
        result[tile_global_id] = r;

        // free exactly once
        H.release();
        Htmp.release();
      }
    }

    // helper to compute dynamic shmem for a given block size
    template <int DIM, int TS>
    static size_t
    smem_for_block(int block_threads)
    {
      const int tilesPerBlock = block_threads / TS;
      // [ x_shared (DIM) | grad_scratch (DIM) | done_flag (1) ] per tile
      const size_t perTileDoubles = 2 * DIM + 1;
      return size_t(tilesPerBlock) * perTileDoubles * sizeof(double);
    }

    template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
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
      // choose tile size per DIM and block siz
      constexpr int TS = tile_size<DIM>::value; // 1,2,4,8,16,32
      int dev = 0;
      cudaGetDevice(&dev);
      cudaDeviceProp props{};
      cudaGetDeviceProperties(&props, dev);

      // try a few sane candidates, cap at 256 to stay under reg budget
      int blockThreads = 256;
      for (int cand : {256, 192, 160, 128, 96, 64, 32}) {
        if (cand < TS)
          continue; // block must be ≥ tile
        if (cand % 32)
          continue; // multiple of warp
        blockThreads = cand;
        break;
      }

      const int tilesPerBlock = blockThreads / TS;
      const size_t shmemBytes = smem_for_block<(int)DIM, TS>(blockThreads);

      dim3 optBlock(blockThreads);
      dim3 optGrid(
        static_cast<unsigned>((N + tilesPerBlock - 1) / tilesPerBlock));
      dbuf deviceResults;
      try {
        deviceResults = dbuf(N);
      }
      catch (const cuda_exception<3>&) {
        Result<DIM> r{};
        r.status = 3;
        return r;
      }

      result_buffer<DIM> d_results;
      try {
        d_results = result_buffer<DIM>(N);
      }
      catch (const cuda_exception<3>&) {
        Result<DIM> r{};
        r.status = 3;
        return r;
      }

      cudaEvent_t startOpt, stopOpt;
      cudaEventCreate(&startOpt);
      cudaEventCreate(&stopOpt);
      cudaEventRecord(startOpt);

      // Allocate and initialize BFGSContext
      util::BFGSContext* d_ctx;
      cudaMalloc(&d_ctx, sizeof(util::BFGSContext));
      util::BFGSContext h_ctx = {0, 0};
      cudaMemcpy(
        d_ctx, &h_ctx, sizeof(util::BFGSContext), cudaMemcpyHostToDevice);

      // launch the tile-per-BFGS kernel
      optimize_tiles<Function, (int)DIM, TS><<<optGrid, optBlock, shmemBytes>>>(
        f,
        lower,
        upper,
        pso_results_device,
        util::non_null{deviceResults.data()},
        save_trajectories ? deviceTrajectory : nullptr,
        (int)N,
        MAX_ITER,
        requiredConverged,
        tolerance,
        util::non_null{d_results.data()},
        util::non_null{states},
        util::non_null{d_ctx},
        save_trajectories);

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::fprintf(
          stderr, "BFGS kernel launch failed: %s\n", cudaGetErrorString(err));
        Result<DIM> r{};
        r.status = 4;
        return r;
      }
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::fprintf(
          stderr, "BFGS kernel runtime error: %s\n", cudaGetErrorString(err));
        Result<DIM> r{};
        r.status = 4;
        return r;
      }

      cudaEventRecord(stopOpt);
      cudaEventSynchronize(stopOpt);
      cudaEventElapsedTime(&ms_opt, startOpt, stopOpt);
      cudaEventDestroy(startOpt);
      cudaEventDestroy(stopOpt);
      cudaFree(d_ctx);

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
      return best;
    }

  } // end parallel namespace

} // end namespace bfgs
