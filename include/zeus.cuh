#pragma once

#include <type_traits>
#include <functional>

#include "fun.h"
#include "duals.cuh"
#include "pso.cuh"
#include "utils.cuh"
#include "bfgs.cuh"

namespace zeus {

  template <typename F, typename Arg> // F is callable type, Arg is element-type
                                      // inside vector
                                      auto
                                      fmap(F f, std::vector<Arg> const& xs)
  {
    using R = std::invoke_result_t<F, Arg>; // std trait that yields the type
                                            // F(Arg) returns
    std::vector<R> result;
    result.reserve(xs.size());
    for (auto const& x : xs) {
      result.push_back(f(x));
    }
    return result;
  }

  namespace detail {
    template <typename Function, size_t DIM>
    Result<DIM>
    ZeusImpl(Function f,
             double lower,
             double upper,
             double* hostResults,
             int N,
             int MAX_ITER,
             int PSO_ITER,
             int requiredConverged,
             std::string fun_name,
             double tolerance,
             int seed,
             int run)
    {
      int blockSize, minGridSize;
      cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        &blockSize,
        bfgs::optimizeKernel<Function, DIM, 128>,
        0,
        N);
      float ms_rand = 0.0f;
      curandState* states = bfgs::initialize_states(N, seed, ms_rand);
      // printf("Recommended block size: %d\n", blockSize);
      bool save_trajectories = util::askUser2saveTrajectories();
      double* deviceTrajectory = nullptr;
      double* pso_results_device = nullptr;
      float ms_init = 0.0f, ms_pso = 0.0f;
      if (PSO_ITER >= 0) {
        pso_results_device = pso::launch<Function, DIM>(
          PSO_ITER, N, lower, upper, ms_init, ms_pso, seed, states, f);
        // printf("pso init: %.2f main loop: %.2f", ms_init, ms_pso);
      } // end if pso_iter > 0
      if (!pso_results_device)
        std::cout << "still null" << std::endl;
      float ms_opt = 0.0f;
      Result best = bfgs::launch<Function, DIM>(N,
                                                PSO_ITER,
                                                MAX_ITER,
                                                upper,
                                                lower,
                                                pso_results_device,
                                                hostResults,
                                                deviceTrajectory,
                                                requiredConverged,
                                                tolerance,
                                                save_trajectories,
                                                ms_opt,
                                                fun_name,
                                                states,
                                                run,
                                                f);
      if (PSO_ITER >
          0) { // optimzation routine is finished, so we can free that
               // array on the device
        cudaFree(pso_results_device);
      }

      double error =
        util::calculate_euclidean_error(fun_name, best.coordinates, DIM);
      util::append_results_2_tsv(DIM,
                                 N,
                                 fun_name,
                                 ms_init,
                                 ms_pso,
                                 ms_opt,
                                 ms_rand,
                                 MAX_ITER,
                                 PSO_ITER,
                                 error,
                                 best.fval,
                                 best.coordinates,
                                 best.idx,
                                 best.status,
                                 best.gradientNorm,
                                 run,
                                 best.c.claimed,
                                 best.c.actual,
                                 best.c.surrendered,
                                 best.c.stopped);

      cudaError_t cuda_error = cudaGetLastError();
      if (cuda_error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
      }
      return best;
    } // end Zeus
  } // namespace detail

  template <typename Function, size_t DIM>
  Result<DIM>
  Zeus(Function f,
       double lower,
       double upper,
       double* hostResults,
       int N,
       int MAX_ITER,
       int PSO_ITER,
       int requiredConverged,
       std::string fun_name,
       double tolerance,
       int seed,
       int run)
  {
    return detail::ZeusImpl<Function, DIM>(std::move(f),
                                           lower,
                                           upper,
                                           hostResults,
                                           N,
                                           MAX_ITER,
                                           PSO_ITER,
                                           requiredConverged,
                                           std::move(fun_name),
                                           tolerance,
                                           seed,
                                           run);
  }

  /*
  template<class Function>
  constexpr size_t fun_dim_v = std::remove_cv_t<Function>::DIM;          //
  SFINAE-friendly alias
  */

  // concept HasDim = requires { { std::remove_cv_t<T>::DIM } ->
  // std::convertible_to<std::size_t>; };

  template <class Function>
    requires requires {
      std::remove_cv_t<Function>::DIM;
    } // contraint expression
  auto
  Zeus(Function f,
       double lower,
       double upper,
       double* hostResults,
       int N,
       int MAX_ITER,
       int PSO_ITER,
       int requiredConverged,
       std::string fun_name,
       double tolerance,
       int seed,
       int run)
    //-> Result< fun_dim_v<Function> >
    -> Result<std::remove_cv_t<Function>::DIM>
  {
    constexpr size_t DIM = std::remove_cv_t<Function>::DIM;
    // constexpr size_t DIM = fun_dim_v<Function>;
    return detail::ZeusImpl<Function, DIM>(std::move(f),
                                           lower,
                                           upper,
                                           hostResults,
                                           N,
                                           MAX_ITER,
                                           PSO_ITER,
                                           requiredConverged,
                                           std::move(fun_name),
                                           tolerance,
                                           seed,
                                           run);
  }

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
    util::setup_curand_states<<<blocks, threads>>>(d_states, seed, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_rand, t0, t1);
    cudaDeviceSynchronize();
    return d_states;
  }

} // end zeus namespace
