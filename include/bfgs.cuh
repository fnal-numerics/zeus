#pragma once

#include <cub/cub.cuh>
#include <cuda_runtime.h>

#include "fun.h"
#include "duals.cuh"
#include "utils.cuh"
#include "pso.cuh"
//#include "zeus.cuh"

namespace bfgs {

  __device__ int d_stopFlag         = 0;
  __device__ int d_convergedCount   = 0;
  __device__ int d_threadsRemaining = 0;


template <typename Function, int DIM, unsigned int blockSize>
__global__ void
optimizeKernel(
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

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N)
    return;

  curandState localState = states[idx];

  // int early_stopping = 0;
  double H[DIM * DIM];
  double g[DIM], x[DIM], x_new[DIM], p[DIM], g_new[DIM], delta_x[DIM],
    delta_g[DIM]; //, new_direction[DIM];
  // double tolerance = 1e-5;
  //  Line Search params

  Result<DIM> r;
  r.status = -1; // assume “not converged” by default
  r.fval = 333777.0;
  r.gradientNorm = 69.0;
  for (int d = 0; d < DIM; ++d) {
    r.coordinates[d] = 0.0;
  }
  r.iter = -1;
  r.idx = idx;
  util::initialize_identity_matrix(H, DIM);

  int num_steps = 0;

  // initialize x either from PSO array or fallback by RNG
  if (pso_array) {
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
      x[d] = pso_array[idx * DIM + d];
      // if(idx == 0) printf("x[%d]=%0.7f\n", d, x[d]);
    }
  } else {
// unsigned int seed = 456;
#pragma unroll
    for (int d = 0; d < DIM; ++d) {
      // x[d] = util::statelessUniform(idx,d,1,lower, upper, seed);
      x[d] = util::generate_random_double(&localState, lower, upper);
    }
    states[idx] = localState;
  }

  double f0 = Function::evaluate(x); // rosenbrock_device(x, DIM);
  deviceResults[idx] = f0;
  double bestVal = f0;
  // if (idx == 0) printf("\n\nf0 = %f", f0);
  int iter;
  dual::calculateGradientUsingAD<Function, DIM>(x, g);
  for (iter = 0; iter < MAX_ITER; ++iter) {
    // printf("inside BeeG File System");
    //  check if somebody already asked to stop
    if (atomicAdd(&d_stopFlag, 0) !=
        0) { // atomicAdd here just to get a strong read-barrier
      // CUDA will fetch a coherent copy of the integer from global memory.
      // as soon as one thread writes 1 into d_stopFlag via atomicExch,
      // the next time any thread does atomicAdd(&d_stopFlag, 0) it’ll see 1 and
      // break.
      // printf("thread %d get outta dodge cuz we converged...", idx);
      r.status = 2;
      r.iter = iter;
      r.fval = Function::evaluate(x);
      for (int d = 0; d < DIM; d++) {
        r.coordinates[d] = x[d];
      }
      r.gradientNorm = util::calculate_gradient_norm<DIM>(g);
      break;
    }
    num_steps++;

    util::compute_search_direction<DIM>(p, H, g); // p = -H * g

    // use the alpha obtained from the line search
    double alpha = util::line_search<Function, DIM>(bestVal, x, p, g);
    if (alpha == 0.0) {
      printf("Alpha is zero, no movement in iteration=%d\n", iter);
      alpha = 1e-3;
    }

    // update current point x by taking a step size of alpha in the direction p
    for (int i = 0; i < DIM; ++i) {
      x_new[i] = x[i] + alpha * p[i];
      delta_x[i] = x_new[i] - x[i];
    }

    double fnew = Function::evaluate(x_new);
    // get the new gradient g_new at x_new
    dual::calculateGradientUsingAD<Function, DIM>(x_new, g_new);

    // calculate new delta_x and delta_g
    for (int i = 0; i < DIM; ++i) {
      delta_g[i] =
        g_new[i] - g[i]; // difference in gradient at the new point vs old point
    }

    // calculate the the dot product between the change in x and change in
    // gradient using new point
    double delta_dot = util::dot_product_device(delta_x, delta_g, DIM);

    // bfgs update on H
    util::bfgs_update<DIM>(H, delta_x, delta_g, delta_dot);
    // only update x and g for next iteration if the new minima is smaller than
    // previous
    // double min = Function::evaluate(x_new);//rosenbrock_device(x_new, DIM);
    if (fnew < bestVal) {
      bestVal = fnew;
      for (int i = 0; i < DIM; ++i) {
        x[i] = x_new[i];
        g[i] = g_new[i];
      }
    }
    // refactor? yes
    double grad_norm = util::calculate_gradient_norm<DIM>(g);
    if (grad_norm < tolerance) {
      // atomically increment the converged counter
      int oldCount = atomicAdd(&d_convergedCount, 1);
      int newCount = oldCount + 1;
      double fcurr = Function::evaluate(x);
      // printf("\nconverged for %d at iter=%d); f = %.6f;",idx, iter,fcurr);
      // for (int d = 0; d < DIM; ++d) { printf(" % .6f", x[d]);}
      // printf(" ]\n");
      r.status = 1;
      r.gradientNorm = grad_norm;
      r.fval = Function::evaluate(x);
      r.iter = iter;
      for (int d = 0; d < DIM; ++d) {
        r.coordinates[d] = x[d];
      }
      // if we just hit the threshold set by the user, the VERY FIRST thread to
      // do so sets d_stopFlag=1 so everyone else exits on their next check
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

    //  deviceTrajectory layout: idx * (MAX_ITER * DIM) + iter * DIM + i
    if (save_trajectories) {
      for (int i = 0; i < DIM; i++) {
        deviceTrajectory[idx * (MAX_ITER * DIM) + iter * DIM + i] = x[i];
      }
    }

    // for(int i=0; i<DIM; ++i) {x[i] = x_new[i];}
  } // end bfgs loop
  // if we broek out because we hit the max numberof iterations, then its a
  // surrender
  if (MAX_ITER == iter) {
    r.status = 0; // surrender
    r.iter = iter;
    r.gradientNorm = util::calculate_gradient_norm<DIM>(g);
    r.fval = Function::evaluate(x);
    for (int d = 0; d < DIM; ++d) {
      r.coordinates[d] = x[d];
    }
  }
  deviceResults[idx] = Function::evaluate(x);
  result[idx] = r;
} // end optimizerKernel


template <int DIM>
Result<DIM>
launch_reduction(int N, double* deviceResults, Result<DIM>* h_results)
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
  cudaMemcpy(&h_argMin, deviceArgMin, sizeof(h_argMin), cudaMemcpyDeviceToHost);

  int globalMinIndex = h_argMin.key;

  // print the “best” thread’s full record
  Result best = h_results[globalMinIndex];
  printf("Global best summary:\n");
  printf("   idx          = %d\n", best.idx);
  printf("   status       = %d\n", best.status);
  printf("   fval         = %.6f\n", best.fval);
  printf("   gradientNorm = %.6f\n", best.gradientNorm);
  printf("   iter         = %d\n", best.iter);
  printf("   coords       = [");
  for (int d = 0; d < DIM; ++d) {
    printf(" %.7f", best.coordinates[d]);
  }
  printf(" ]\n");

  cudaFree(deviceResults);
  cudaFree(deviceArgMin);
  cudaFree(d_temp_storage);
  return best;
}




template <typename Function, int DIM>
Result<DIM>
launch(const int N,
            const int pso_iter,
            const int MAX_ITER,
            const double upper,
            const double lower,
            double* pso_results_device,
            double* hostResults,
            double* deviceTrajectory,
            const int requiredConverged,
            const double tolerance,
            bool save_trajectories,
            float& ms_opt,
            std::string fun_name,
            curandState* states,
            const int run)
{
  int blockSize, minGridSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize, optimizeKernel<Function, DIM, 128>, 0, N);
  // printf("\nRecommended block size: %d\n", blockSize);

  // prepare optimizer buffers & copy hostResults --> device
  double* deviceResults;
  cudaMalloc(&deviceResults, N * sizeof(double));
  cudaMemcpy(
    deviceResults, hostResults, N * sizeof(double), cudaMemcpyHostToDevice);

  dim3 optBlock(blockSize);
  dim3 optGrid((N + blockSize - 1) / blockSize);

  // optimizeKernel time
  cudaEvent_t startOpt, stopOpt;
  cudaEventCreate(&startOpt);
  cudaEventCreate(&stopOpt);
  cudaEventRecord(startOpt);

  Result<DIM>* h_results = new Result<DIM>[N]; // host copy
  Result<DIM>* d_results = nullptr;
  cudaMalloc(&d_results, N * sizeof(Result<DIM>));
  /*
  for(int i=0;i<DIM;i++){
      std::cout << hPBestX[i] << " ";
  }*/
  std::cout << std::endl;
  if (save_trajectories) {
    cudaMalloc(&deviceTrajectory, N * MAX_ITER * DIM * sizeof(double));
    optimizeKernel<Function, DIM, 128>
      <<<optGrid, optBlock>>>(lower,
                              upper,
                              pso_results_device,
                              deviceResults,
                              deviceTrajectory,
                              N,
                              MAX_ITER,
                              requiredConverged,
                              tolerance,
                              d_results,
                              states,
                              /*saveTraj=*/true);
  } else {
    optimizeKernel<Function, DIM, 128>
      <<<optGrid, optBlock>>>(lower,
                              upper,
                              pso_results_device,
                              deviceResults,
                              /*traj=*/nullptr,
                              N,
                              MAX_ITER,
                              requiredConverged,
                              tolerance,
                              d_results,
                              states);
  }
  cudaDeviceSynchronize();
  cudaEventRecord(stopOpt);
  cudaEventSynchronize(stopOpt);
  cudaEventElapsedTime(&ms_opt, startOpt, stopOpt);
  // printf("\nOptimization Kernel execution time = %.3f ms\n", ms_opt);
  cudaEventDestroy(startOpt);
  cudaEventDestroy(stopOpt);

  cudaMemcpy(
    h_results, d_results, N * sizeof(Result<DIM>), cudaMemcpyDeviceToHost);
  Convergence c = util::dump_data_2_file(h_results, fun_name, N, pso_iter, run);
  /*int countConverged = 0, surrender = 0, stopped = 0;
  for (int i = 0; i < N; ++i) {
      if (h_results[i].status == 1) {
          countConverged++;
      } else if(h_results[i].status == 2) { // particle was stopped early
          stopped++;
      } else {
          surrender++;
      }
  }*/
  // printf("\n%d converged, %d stopped early, %d surrendered\n",countConverged,
  // stopped, surrender);

  Result best = launch_reduction<DIM>(N, deviceResults, h_results);
  best.c = c;
  return best;
}



}//end namespace bfgs



