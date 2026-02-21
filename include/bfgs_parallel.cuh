#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "duals.cuh"
#include "utils.cuh"
#include "traits.hpp"
#include "bfgs_common.cuh"

namespace bfgs {
  namespace parallel {
    namespace cg = cooperative_groups;

    /// Parallel forward-mode AD across dimensions using a tile of size TS.
    /// TS must divide 32 (so tiles don’t cross warp boundaries).
    template <class Function, int ZEUS_DIM, int TS>
    __device__ void
    grad_ad_tile(const Function& f,
                 const double* __restrict__ x_shared,
                 double* __restrict__ g_out,
                 cg::thread_block_tile<TS> tile,
                 double* __restrict__ scratch)
    {
      static_assert(32 % TS == 0, "TS must divide 32");

      // stream over dimensions in TS-sized chunks
      for (int base = 0; base < ZEUS_DIM; base += TS) {
        const int i = base + tile.thread_rank();
        if (i < ZEUS_DIM) {
          // std::array, not a C array
          std::array<dual::DualNumber, ZEUS_DIM> xDual;

#pragma unroll
          for (int j = 0; j < ZEUS_DIM; ++j) {
            xDual[j].real = x_shared[j];
            xDual[j].dual = 0.0;
          }
          xDual[i].dual = 1.0;

          // f(std::array<dual::DualNumber, ZEUS_DIM>) -> dual::DualNumber
          auto y = f(xDual);
          scratch[i] = y.dual; // delf/delx_i
        }
        tile.sync();
      }

      if (tile.thread_rank() == 0) {
#pragma unroll
        for (int j = 0; j < ZEUS_DIM; ++j)
          g_out[j] = scratch[j];
      }
      tile.sync();
    }

    /// Compile-time tile size selection based on problem dimensionality.
    template <int ZEUS_DIM>
    struct tileSize {
      static constexpr int value = (ZEUS_DIM >= 32 ? 32 :
                                    ZEUS_DIM >= 16 ? 16 :
                                    ZEUS_DIM >= 8  ? 8 :
                                    ZEUS_DIM >= 4  ? 4 :
                                    ZEUS_DIM >= 2  ? 2 :
                                                     1);
    };

    /// Parallel BFGS kernel where each tile performs one optimization.
    template <typename Function,
              std::size_t ZEUS_DIM = zeus::FnTraits<Function>::arity,
              int TS>
    __global__ void
    optimizeTiles(Function f,
                  const double lower,
                  const double upper,
                  const double* pso_array,
                  util::NonNull<double*> deviceResults,
                  double* deviceTrajectory,
                  int N,
                  const int MAX_ITER,
                  const int requiredConverged,
                  const double tolerance,
                  util::NonNull<zeus::Result<ZEUS_DIM>*> result,
                  util::NonNull<curandState*> states,
                  util::NonNull<util::BFGSContext*> ctx,
                  bool save_trajectories = false)
    {

      static_assert(
        std::is_same_v<
          decltype(std::declval<Function>()(
            std::declval<std::array<dual::DualNumber, ZEUS_DIM>>())),
          dual::DualNumber>,
        "Objective must be templated: template<class T> T f(const "
        "std::array<T,ZEUS_DIM>&)");

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
      // [ x_shared (ZEUS_DIM) | grad_scratch (ZEUS_DIM) | done_flag (1 as int)
      // ]
      extern __shared__ double smem[];
      const int per_tile_doubles =
        2 * ZEUS_DIM + 1; // last double used as an int flag
      double* base_ptr = smem + tile_local_id * per_tile_doubles;
      double* x_shared = base_ptr;
      double* grad_scratch = base_ptr + ZEUS_DIM;
      int* done_flag = reinterpret_cast<int*>(base_ptr + 2 * ZEUS_DIM);

      // Lane-0 state
      std::array<double, ZEUS_DIM> x_arr, x_new, g_arr, g_new, p_arr;
      DeviceMatrix<double> H(ZEUS_DIM, ZEUS_DIM), Htmp(ZEUS_DIM, ZEUS_DIM);
      zeus::Result<ZEUS_DIM> r;

      // init
      if (tile.thread_rank() == 0) {
        *done_flag = 0;
        util::initializeIdentityMatrix(&H, ZEUS_DIM);

        curandState localState = states[tile_global_id];
        for (int d = 0; d < ZEUS_DIM; ++d) {
          double v = pso_array ?
                       pso_array[tile_global_id * ZEUS_DIM + d] :
                       util::generateRandomDouble(&localState, lower, upper);
          x_arr[d] = v;
          g_arr[d] = 0.0;
          x_shared[d] = v; // publish to tile
        }
        states[tile_global_id] = localState;
      }
      tile.sync();

      // first gradient at x_arr
      grad_ad_tile<Function, ZEUS_DIM, TS>(
        f, x_shared, grad_scratch, tile, grad_scratch);
      if (tile.thread_rank() == 0) {
        for (int d = 0; d < ZEUS_DIM; ++d)
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
            writeResult<ZEUS_DIM>(r,
                                  2,
                                  f(x_arr),
                                  x_arr.data(),
                                  util::calculateGradientNorm<ZEUS_DIM>(g_arr),
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
          util::computeSearchDirection<ZEUS_DIM>(p_arr, &H, g_arr); // p = -H g
          double alpha = util::lineSearch<Function, ZEUS_DIM>(
            bestVal, x_arr, p_arr, g_arr, f);
          if (alpha == 0.0) {
            printf(
              "Alpha is zero, using fallback alpha=1e-3 (iter=%d, idx=%d)\n",
              iter,
              tile_global_id);
            alpha = 1e-3;
          }
          for (int d = 0; d < ZEUS_DIM; ++d) {
            x_new[d] = x_arr[d] + alpha * p_arr[d];
            x_shared[d] = x_new[d]; // publish to tile for gradient
          }
        }
        tile.sync();

        // New gradient at x_new
        grad_ad_tile<Function, ZEUS_DIM, TS>(
          f, x_shared, grad_scratch, tile, grad_scratch);
        if (tile.thread_rank() == 0) {
          for (int d = 0; d < ZEUS_DIM; ++d)
            g_new[d] = grad_scratch[d];

          // BFGS update
          double delta_x[ZEUS_DIM], delta_g[ZEUS_DIM];
          for (int d = 0; d < ZEUS_DIM; ++d) {
            delta_x[d] = x_new[d] - x_arr[d];
            delta_g[d] = g_new[d] - g_arr[d];
          }

          const double fnew = f(x_new);
          const double delta_dot =
            util::dotProductDevice(delta_x, delta_g, ZEUS_DIM);
          util::bfgsUpdate<ZEUS_DIM>(&H, delta_x, delta_g, delta_dot, &Htmp);

          if (fnew < bestVal) {
            bestVal = fnew;
            for (int d = 0; d < ZEUS_DIM; ++d) {
              x_arr[d] = x_new[d];
              g_arr[d] = g_new[d];
            }
          }

          const double grad_norm = util::calculateGradientNorm<ZEUS_DIM>(g_arr);
          if (!isfinite(grad_norm) || !isfinite(fnew)) {
            writeResult<ZEUS_DIM>(
              r, 5, fnew, x_arr.data(), grad_norm, iter, tile_global_id);
            *done_flag = 1;
          } else if (grad_norm < tolerance) {
            const int oldCount = atomicAdd(&ctx->convergedCount, 1);
            const double fcurr = f(x_arr);
            writeResult<ZEUS_DIM>(
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
          writeResult<ZEUS_DIM>(r,
                                0,
                                f(x_arr),
                                x_arr.data(),
                                util::calculateGradientNorm<ZEUS_DIM>(g_arr),
                                iter,
                                tile_global_id);
        }
        deviceResults[tile_global_id] = r.fval;
        result[tile_global_id] = r;
      }
    }

    /// Calculate shared memory bytes needed for optimize_tiles kernel.
    template <int ZEUS_DIM, int TS>
    static size_t
    smem_for_block(int block_threads)
    {
      const int tilesPerBlock = block_threads / TS;
      // [ x_shared (ZEUS_DIM) | grad_scratch (ZEUS_DIM) | done_flag (1) ] per
      // tile
      const size_t perTileDoubles = 2 * ZEUS_DIM + 1;
      return size_t(tilesPerBlock) * perTileDoubles * sizeof(double);
    }

    /// Launch parallel BFGS optimization with N independent optimizations.
    /// Uses tile-based parallelism for cooperative gradient computation.
    template <typename Function,
              std::size_t ZEUS_DIM = zeus::FnTraits<Function>::arity>
    zeus::Result<ZEUS_DIM>
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
      // choose tile size per ZEUS_DIM and block siz
      constexpr int TS = tileSize<ZEUS_DIM>::value; // 1,2,4,8,16,32
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
      const size_t shmemBytes = smem_for_block<(int)ZEUS_DIM, TS>(blockThreads);

      dim3 optBlock(blockThreads);
      dim3 optGrid(
        static_cast<unsigned>((N + tilesPerBlock - 1) / tilesPerBlock));
      DoubleBuffer deviceResults;
      try {
        deviceResults = DoubleBuffer(N);
      }
      catch (const CudaError& e) {
        zeus::Result<ZEUS_DIM> r{};
        r.status = (e.code() == cudaErrorMemoryAllocation) ? 3 : 4;
        return r;
      }

      ResultBuffer<ZEUS_DIM> d_results;
      try {
        d_results = ResultBuffer<ZEUS_DIM>(N);
      }
      catch (const CudaError& e) {
        zeus::Result<ZEUS_DIM> r{};
        r.status = (e.code() == cudaErrorMemoryAllocation) ? 3 : 4;
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
      optimizeTiles<Function, (int)ZEUS_DIM, TS>
        <<<optGrid, optBlock, shmemBytes>>>(
          f,
          lower,
          upper,
          pso_results_device,
          util::NonNull{deviceResults.data()},
          save_trajectories ? deviceTrajectory : nullptr,
          (int)N,
          MAX_ITER,
          requiredConverged,
          tolerance,
          util::NonNull{d_results.data()},
          util::NonNull{states},
          util::NonNull{d_ctx},
          save_trajectories);

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        std::fprintf(
          stderr, "BFGS kernel launch failed: %s\n", cudaGetErrorString(err));
        zeus::Result<ZEUS_DIM> r{};
        r.status = 4;
        return r;
      }
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::fprintf(
          stderr, "BFGS kernel runtime error: %s\n", cudaGetErrorString(err));
        zeus::Result<ZEUS_DIM> r{};
        r.status = 4;
        return r;
      }

      cudaEventRecord(stopOpt);
      cudaEventSynchronize(stopOpt);
      cudaEventElapsedTime(&ms_opt, startOpt, stopOpt);
      cudaEventDestroy(startOpt);
      cudaEventDestroy(stopOpt);
      cudaFree(d_ctx);

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
      return best;
    }

  } // end parallel namespace
} // namespace bfgs
