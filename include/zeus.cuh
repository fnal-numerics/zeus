#pragma once

#include <tuple>
#include <functional>

#include "duals.cuh"
#include "pso.cuh"
#include "utils.cuh"
#include "bfgs_common.cuh"
#include "bfgs_sequential.cuh"
#include "bfgs_parallel.cuh"
#include "traits.hpp"
#include "cuda_buffer.cuh"

__device__ int d_stopFlag = 0;
__device__ int d_convergedCount = 0;
__device__ int d_threadsRemaining = 0;

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

  namespace impl {
    template <typename Function,
              std::size_t ZEUS_DIM = FnTraits<Function>::arity>
    zeus::Result<ZEUS_DIM>
    Zeus(Function const& f,
         double lower,
         double upper,
         size_t N,
         int MAX_ITER,
         int PSO_ITER,
         int requiredConverged,
         std::string const& fun_name,
         double tolerance,
         int seed,
         int run,
         bool parallel)
    {
      util::setStackSize();
      float ms_rand = 0.0f;
      curandState* states = bfgs::initializeStates(N, seed, ms_rand);
      // save trajectories?
      bool save_trajectories = util::askUserToSaveTrajectories();
      double* deviceTrajectory = nullptr;
      // DoubleBuffer is CudaBuffer<double>
      DoubleBuffer trajBuffer(0);
      if (save_trajectories) {
        trajBuffer = DoubleBuffer(size_t(N) * MAX_ITER * ZEUS_DIM);
        deviceTrajectory = trajBuffer.data();
      }

      DoubleBuffer pso_results_device(0);
      float ms_init = 0.0f, ms_pso = 0.0f;
      if (PSO_ITER >= 0) {
        try {
          pso_results_device = pso::launch<Function, ZEUS_DIM>(
            PSO_ITER, N, lower, upper, ms_init, ms_pso, seed, states, f);
          // printf("pso init: %.2f main loop: %.2f", ms_init, ms_pso);
        }
        catch (const CudaError& e) {
          zeus::Result<ZEUS_DIM> r;
          r.status = (e.code() == cudaErrorMemoryAllocation) ? 3 : 4;
          return r;
        }

      } // end if pso_iter > 0

      zeus::Result<ZEUS_DIM> best;
      float ms_opt = 0.0f;
      if (run != 0) {
        std::cout << "parallel" << "\n";
        best =
          bfgs::parallel::launch<Function, ZEUS_DIM>(N,
                                                     PSO_ITER,
                                                     MAX_ITER,
                                                     upper,
                                                     lower,
                                                     pso_results_device.data(),
                                                     deviceTrajectory,
                                                     requiredConverged,
                                                     tolerance,
                                                     save_trajectories,
                                                     ms_opt,
                                                     fun_name,
                                                     states,
                                                     run,
                                                     f);
      } else {

        std::cout << "sequential" << "\n";
        best = bfgs::sequential::launch<Function, ZEUS_DIM>(
          N,
          PSO_ITER,
          MAX_ITER,
          upper,
          lower,
          pso_results_device.data(),
          deviceTrajectory,
          requiredConverged,
          tolerance,
          save_trajectories,
          ms_opt,
          fun_name,
          states,
          run,
          f);
      }
      double error =
        util::calculateEuclideanError(fun_name, best.coordinates, ZEUS_DIM);
      util::appendResultsToTsv(ZEUS_DIM,
                               N,
                               fun_name,
                               ms_init,
                               ms_pso,
                               ms_opt,
                               ms_rand,
                               MAX_ITER,
                               PSO_ITER,
                               error,
                               run,
                               best);
      cudaError_t cuda_error = cudaGetLastError();
      if (cuda_error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
      }
      return best;
    } // end Zeus
  } // namespace impl

  template <typename Function, std::size_t ZEUS_DIM = FnTraits<Function>::arity>
    requires ZeusObjective<Function, ZEUS_DIM>
  auto
  Zeus(Function const& f,
       double lower,
       double upper,
       size_t N,
       int MAX_ITER,
       int PSO_ITER,
       int requiredConverged,
       std::string fun_name,
       double tolerance,
       int seed,
       int run,
       bool parallel = true)
  {
    return impl::Zeus(f,
                      lower,
                      upper,
                      N,
                      MAX_ITER,
                      PSO_ITER,
                      requiredConverged,
                      std::move(fun_name),
                      tolerance,
                      seed,
                      run,
                      parallel);
  }

} // end zeus namespace
