#include "fun.h"
#include "duals.cuh"
#include "pso.cuh"
#include "utils.cuh"


namespace zeus {

template <typename Function, int DIM>
Result<DIM>
Zeus(const double lower,
     const double upper,
     double* hostResults,
     int N,
     int MAX_ITER,
     int PSO_ITER,
     int requiredConverged,
     std::string fun_name,
     double tolerance,
     const int seed,
     const int run)
{
  int blockSize, minGridSize;
  cudaOccupancyMaxPotentialBlockSize(
    &minGridSize, &blockSize, optimizeKernel<Function, DIM, 128>, 0, N);
  float ms_rand = 0.0f;
  curandState* states = initialize_states(N, seed, ms_rand);
  // printf("Recommended block size: %d\n", blockSize);
  bool save_trajectories = askUser2saveTrajectories();
  double* deviceTrajectory = nullptr;
  double* pso_results_device = nullptr;
  float ms_init = 0.0f, ms_pso = 0.0f;
  if (PSO_ITER >= 0) {
    pso_results_device = pso::launch<Function, DIM>(
      PSO_ITER, N, lower, upper, ms_init, ms_pso, seed, states);
    // printf("pso init: %.2f main loop: %.2f", ms_init, ms_pso);
  } // end if pso_iter > 0
  if (!pso_results_device)
    std::cout << "still null" << std::endl;
  float ms_opt = 0.0f;
  Result best = launch_bfgs<Function, DIM>(N,
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
                                           run);
  if (PSO_ITER > 0) { // optimzation routine is finished, so we can free that
                      // array on the device
    cudaFree(pso_results_device);
  }

  double error = calculate_euclidean_error(fun_name, best.coordinates, DIM);
  append_results_2_tsv(DIM,
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


template <typename Function, int DIM>
void
runOptimizationKernel(double lower,
                      double upper,
                      double* hostResults,
                      int N,
                      int MAX_ITER,
                      int PSO_ITERS,
                      int requiredConverged,
                      std::string fun_name,
                      double tolerance,
                      int seed,
                      const int run)
{
  // void runOptimizationKernel(double* hostResults, int N, int dim) {
  /*printf("first 20 hostResults\n");
  for(int i=0;i<20;i++) {
     printf(" %f ",hostResults[i]);
  }
  printf("\n");
  */
  Result best = Zeus<Function, DIM>(lower,
                                    upper,
                                    hostResults,
                                    N,
                                    MAX_ITER,
                                    PSO_ITERS,
                                    requiredConverged,
                                    fun_name,
                                    tolerance,
                                    seed,
                                    run);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  // printf("Sorting the array with %d elements... ", N);
  cudaEventRecord(start);
  // quickSort(hostResults, 0, N - 1);
  cudaEventRecord(stop);
  float milli = 0;
  cudaEventElapsedTime(&milli, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  // printf("took %f ms\n",  milli);

  /*printf("first 20 function values in hostResults\n");
  for(int i=0;i<20;i++) {
     printf(" %f ",hostResults[i]);
  }*/
  printf("\n");
  // cudaMemGetInfo
}


}
