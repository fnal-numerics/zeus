#pragma once

#include <tuple>
#include <functional>

#include "duals.cuh"
#include "pso.cuh"
#include "utils.cuh"
#include "bfgs.cuh"
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
    template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
    Result<DIM>
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
         int run)
    {
      float ms_rand = 0.0f;
      curandState* states = bfgs::initialize_states(N, seed, ms_rand);
      // save trajectories?
      bool save_trajectories = util::askUser2saveTrajectories();
      double* deviceTrajectory = nullptr;
      // dbuf is cuda_buffer<double>
      dbuf trajBuffer(0);
      if (save_trajectories) {
        trajBuffer = dbuf(size_t(N) * MAX_ITER * DIM);
        deviceTrajectory = trajBuffer.data();
      }

      dbuf pso_results_device(0);
      float ms_init = 0.0f, ms_pso = 0.0f;
      if (PSO_ITER >= 0) {
        try {
          pso_results_device = pso::launch(
            PSO_ITER, N, lower, upper, ms_init, ms_pso, seed, states, f);
          // printf("pso init: %.2f main loop: %.2f", ms_init, ms_pso);
        }
        catch (cuda_exception<3>&) {
          Result<DIM> r;
          r.status = 3;
          return r;
        }
        catch (cuda_exception<4>&) {
          Result<DIM> r;
          r.status = 4;
          return r;
        }
      } // end if pso_iter > 0

      float ms_opt = 0.0f;
      Result best = bfgs::launch(N,
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
  } // namespace impl

  template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
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
       int run)
  {
    static_assert(
      std::is_same_v<decltype(f(
                       std::declval<const std::array<double, DIM>&>())),
                     double>,
      "Your objective must be callable as f(std::array<double,DIM>) -> double");
    static_assert(
      std::is_same_v<decltype(std::declval<Function>()(
                       std::declval<std::array<dual::DualNumber, DIM>>())),
                     dual::DualNumber>,
      "\n\n> This objective is not templated.\nMake it\n\ttemplate<class T> T "
      "fun(const std::array<T,N>) { ... }\n");
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
                      run);
  }

} // end zeus namespace
