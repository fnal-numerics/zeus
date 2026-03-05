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
#include "cuda_buffer.hpp"

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
              typename StateType,
              std::size_t ZEUS_DIM = FnTraits<Function>::arity>
    zeus::Result<ZEUS_DIM>
    ZeusInternal(Function const& f,
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
                 bool parallel,
                 std::string_view trajectory_file,
                 StateType* states,
                 float ms_rand)
    {
      if (N == 0) {
        return zeus::Result<ZEUS_DIM>{};
      }

      // save trajectories?
      bool save_trajectories = !trajectory_file.empty();
      double* deviceTrajectory = nullptr;
      int8_t* deviceStatus = nullptr;
      DoubleBuffer trajBuffer(0);
      CudaBuffer<int8_t> statusBuffer(0);
      if (save_trajectories) {
        size_t size = size_t(N) * MAX_ITER;
        size_t traj_size = size * (ZEUS_DIM + 2);
        trajBuffer = DoubleBuffer(traj_size);
        deviceTrajectory = trajBuffer.data();
        util::fillWithNaN(deviceTrajectory, traj_size);

        statusBuffer = CudaBuffer<int8_t>(size);
        deviceStatus = statusBuffer.data();
        cudaMemset(deviceStatus, -1, size);
      }

      DoubleBuffer pso_results_device(0);
      float ms_init = 0.0f, ms_pso = 0.0f;
      if (PSO_ITER >= 0) {
        try {
          pso_results_device = pso::launch<Function, ZEUS_DIM, StateType>(
            PSO_ITER, N, lower, upper, ms_init, ms_pso, seed, states, f);
        }
        catch (const CudaError& e) {
          zeus::Result<ZEUS_DIM> r;
          r.status = (e.code() == cudaErrorMemoryAllocation) ? 3 : 4;
          return r;
        }
      }

      zeus::Result<ZEUS_DIM> best;
      float ms_opt = 0.0f;
      if (run != 0) {
        std::cout << "parallel" << "\n";
        best = bfgs::parallel::launch<Function, ZEUS_DIM, StateType>(
          N,
          PSO_ITER,
          MAX_ITER,
          lower,
          upper,
          pso_results_device.data(),
          deviceTrajectory,
          deviceStatus,
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
        best = bfgs::sequential::launch<Function, ZEUS_DIM, StateType>(
          N,
          PSO_ITER,
          MAX_ITER,
          lower,
          upper,
          pso_results_device.data(),
          deviceTrajectory,
          deviceStatus,
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

      if (save_trajectories) {
        size_t n_steps = size_t(N) * MAX_ITER;
        std::vector<double> hostTrajectory(n_steps * (ZEUS_DIM + 2));
        std::vector<int8_t> hostStatus(n_steps);

        cudaMemcpy(hostTrajectory.data(),
                   deviceTrajectory,
                   hostTrajectory.size() * sizeof(double),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(hostStatus.data(),
                   deviceStatus,
                   hostStatus.size() * sizeof(int8_t),
                   cudaMemcpyDeviceToHost);

        util::writeTrajectoryData(hostTrajectory.data(),
                                  hostStatus.data(),
                                  N,
                                  MAX_ITER,
                                  ZEUS_DIM,
                                  trajectory_file);
      }

      return best;
    }

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
         bool parallel,
         PRNGType prng_type = PRNGType::XORWOW,
         std::string_view trajectory_file = {})
    {
      util::setStackSize();
      float ms_rand = 0.0f;

      switch (prng_type) {
        case PRNGType::PHILOX: {
          auto* states = bfgs::initializeStates<curandStatePhilox4_32_10_t>(
            N, seed, ms_rand);
          auto res =
            ZeusInternal<Function, curandStatePhilox4_32_10_t, ZEUS_DIM>(
              f,
              lower,
              upper,
              N,
              MAX_ITER,
              PSO_ITER,
              requiredConverged,
              fun_name,
              tolerance,
              seed,
              run,
              parallel,
              trajectory_file,
              states,
              ms_rand);
          cudaFree(states);
          return res;
        }
        case PRNGType::SOBOL: {
          auto* states = bfgs::initializeSobolStates(N, ms_rand);
          auto res = ZeusInternal<Function, curandStateSobol32_t, ZEUS_DIM>(
            f,
            lower,
            upper,
            N,
            MAX_ITER,
            PSO_ITER,
            requiredConverged,
            fun_name,
            tolerance,
            seed,
            run,
            parallel,
            trajectory_file,
            states,
            ms_rand);
          cudaFree(states);
          return res;
        }
        case PRNGType::XORWOW:
        default: {
          auto* states =
            bfgs::initializeStates<curandStateXORWOW_t>(N, seed, ms_rand);
          auto res = ZeusInternal<Function, curandStateXORWOW_t, ZEUS_DIM>(
            f,
            lower,
            upper,
            N,
            MAX_ITER,
            PSO_ITER,
            requiredConverged,
            fun_name,
            tolerance,
            seed,
            run,
            parallel,
            trajectory_file,
            states,
            ms_rand);
          cudaFree(states);
          return res;
        }
      }
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
       std::string_view fun_name,
       double tolerance,
       int seed,
       int run,
       bool parallel = true,
       PRNGType prng_type = PRNGType::XORWOW,
       std::string_view trajectory_file = {})
  {
    return impl::Zeus(f,
                      lower,
                      upper,
                      N,
                      MAX_ITER,
                      PSO_ITER,
                      requiredConverged,
                      std::string{fun_name},
                      tolerance,
                      seed,
                      run,
                      parallel,
                      prng_type,
                      trajectory_file);
  }

} // end zeus namespace
