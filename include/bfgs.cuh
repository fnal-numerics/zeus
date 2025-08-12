#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "duals.cuh"
#include "utils.cuh"
#include "pso.cuh"
#include "traits.hpp"

using namespace zeus;

namespace bfgs {
  extern __device__ int d_stopFlag;
  extern __device__ int d_convergedCount;
  extern __device__ int d_threadsRemaining;

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

  template <int DIM>
  __device__ void write_result(Result<DIM>& r,int status,double fval,const double* coordinates,double gradientNorm,int iter, int idx) {
    r.status = status;
    r.iter = iter;
    r.fval = fval;
    r.idx = idx;
    r.gradientNorm = gradientNorm;
    for (int d = 0; d < DIM; ++d) {
      r.coordinates[d] = coordinates[d];
    }
  }

  template <typename Function, int DIM, unsigned int blockSize>
  __global__ void
  optimize(
    Function f,
    const double lower,
    const double upper,
    const double* __restrict__ pso_array, // pso initialized positions
    double* deviceResults,
    double* deviceTrajectory,
    int N,
    const int MAX_ITER,
    const int requiredConverged,
    const double tolerance,
    Result<DIM>* result,
    curandState* states,
    bool save_trajectories = false)
  {
    extern __device__ int d_stopFlag;
    extern __device__ int d_threadsRemaining;
    extern __device__ int d_convergedCount;

    static_assert(
      std::is_same_v<decltype(std::declval<Function>()(
                       std::declval<std::array<dual::DualNumber, DIM>>())),
                     dual::DualNumber>,
      "\n\n> This objective is not templated.\nMake it\n\ttemplate<class T> T "
      "fun(const std::array<T,N>) { ... }\n");
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
      return;

    curandState localState = states[idx];

    std::array<double, DIM> x_arr, x_new, g_arr, g_new, p_arr;

    Matrix<double> H(DIM, DIM);
    Matrix<double> Htmp(DIM, DIM);

    double delta_x[DIM], delta_g[DIM];

    Result<DIM> r;
    for(int i=0;i<DIM;i++)
      x_arr[i] = 7.0;
    write_result<DIM>(r,/*status=*/-1,/*fval=*/333777.0,/*coordinates=*/x_arr.data(),/*norm=*/69.0,/*iter*/0,idx);
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
    dual::calculateGradientUsingAD(f, x_arr, g_arr);
    for (iter = 0; iter < MAX_ITER; ++iter) {
      // printf("inside BeeG File System");
      //  check if somebody already asked to stop
      if (atomicAdd(&d_stopFlag, 0) !=
          0) { // atomicAdd here just to get a strong read-barrier
        // CUDA will fetch a coherent copy of the integer from global memory.
        // as soon as one thread writes 1 into d_stopFlag via atomicExch,
        // the next time any thread does atomicAdd(&d_stopFlag, 0) it’ll see 1
        // and break. printf("thread %d get outta dodge cuz we converged...",
        // idx);
	write_result<DIM>(r,2,f(x_arr),x_arr.data(),util::calculate_gradient_norm<DIM>(g_arr),iter,idx);
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
      dual::calculateGradientUsingAD(f, x_new, g_new);

      // calculate new delta_x and delta_g
      for (int i = 0; i < DIM; ++i) {
        delta_g[i] =
          g_new[i] -
          g_arr[i]; // difference in gradient at the new point vs old point
      }

      // calculate the the dot product between the change in x and change in
      // gradient using new point
      double delta_dot = util::dot_product_device(delta_x, delta_g, DIM);

      // bfgs update on H
      util::bfgs_update<DIM>(&H, delta_x, delta_g, delta_dot, &Htmp);
      // only update x and g for next iteration if the new minima is smaller
      // than previous double min =
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
      if(!isfinite(grad_norm) || !isfinite(fnew)) {
	write_result<DIM>(r,5,fnew,x_arr.data(),grad_norm,iter,idx);
	break;
      }
      if (grad_norm < tolerance) {
        // atomically increment the converged counter
        int oldCount = atomicAdd(&d_convergedCount, 1);
        int newCount = oldCount + 1;
        double fcurr = f(x_arr);
	write_result<DIM>(r,1,fcurr,x_arr.data(),grad_norm,iter,idx);
        // if we just hit the threshold set by the user, the VERY FIRST thread
        // to do so sets d_stopFlag=1 so everyone else exits on their next check
        if (newCount == requiredConverged) {
          // flip the global stop flag
          atomicExch(&d_stopFlag, 1);
          __threadfence();
          printf(
            "\nThread %d is the %d%s converged thread (iter=%d); fn = %.6f.\n",
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
          deviceTrajectory[idx * (MAX_ITER * DIM) + iter * DIM + i] = x_raw[i];
        }
      }*/

    } // end bfgs loop
    // if we broek out because we hit the max numberof iterations, then its a
    // surrender
    if (MAX_ITER == iter) {
      write_result<DIM>(r,0,f(x_arr),x_arr.data(),util::calculate_gradient_norm<DIM>(g_arr),iter,idx);
    }
    deviceResults[idx] = r.fval;
    result[idx] = r;
  } // end optimizerKernel

  template <int DIM>
  Result<DIM>
  launch_reduction(int N, double* deviceResults,Result<DIM> const* h_results)
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
         Function const& f)
  {
    int blockSize, minGridSize;

   cudaOccupancyMaxPotentialBlockSize(
      &minGridSize, &blockSize, optimize<Function, DIM, 128>, 0, N);
    // printf("\nRecommended block size: %d\n", blockSize);
   dbuf deviceResults;
    try {
      deviceResults = dbuf(N);
    } catch (const cuda_exception<3>& e)  {
       Result<DIM> result;
       result.status = 3;
       return result;
       //throw cuda_exception<3>(std::string("bfgs::launch: failed to cudaMalloc deviceResults (") + e.what() + ")\n"); //allocation failed
    }
    dim3 optBlock(blockSize);
    dim3 optGrid((N + blockSize - 1) / blockSize);

    // optimizeKernel time
    cudaEvent_t startOpt, stopOpt;
    cudaEventCreate(&startOpt);
    cudaEventCreate(&stopOpt);
    cudaEventRecord(startOpt);

    result_buffer<DIM> d_results;
    try { 
      d_results = result_buffer<DIM>(N);
    } catch (const cuda_exception<3>& e) {
      Result<DIM> result;
       result.status = 3;
       return result;
      //throw; // allocation failed
    }
    if (save_trajectories) {
      optimize<Function, DIM, 128>
        <<<optGrid, optBlock>>>(f,
                                lower,
                                upper,
                                pso_results_device,
                                deviceResults.data(),
                                deviceTrajectory,
                                N,
                                MAX_ITER,
                                requiredConverged,
                                tolerance,
                                d_results.data(),
                                states,
                                /*saveTraj=*/true);
    } else {
      optimize<Function, DIM, 128>
        <<<optGrid, optBlock>>>(f,
                                lower,
                                upper,
                                pso_results_device,
                                deviceResults.data(),
                                /*traj=*/nullptr,
                                N,
                                MAX_ITER,
                                requiredConverged,
                                tolerance,
                                d_results.data(),
                                states);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::fprintf(stderr,"BFGS kernel launch failed: %s\n",cudaGetErrorString(err));
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

    std::vector<Result<DIM>> h_results(N);
    cudaMemcpy(
      h_results.data(), d_results, N * sizeof(Result<DIM>), cudaMemcpyDeviceToHost);
    Convergence c = util::dump_data_2_file<DIM>(N, h_results.data(), fun_name, pso_iter, run);

    Result best = launch_reduction<DIM>(N, deviceResults.data(), h_results.data());
    best.c = c;
    return best;
  }

} // end namespace bfgs
