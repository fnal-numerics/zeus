#pragma once

#include "fun.h"
#include "duals.cuh"
#include "pso.cuh"
#include "utils.cuh"
#include "bfgs.cuh"

/*namespace bfgs {
  template <typename Function, int DIM, unsigned int blockSize>
  __global__ void optimizeKernel(
    const double lower,
    const double upper,
    const double* __restrict__ pso_array,
    double* deviceResults,
    double* deviceTrajectory,
    int N,
    const int MAX_ITER,
    const int requiredConverged,
    const double tolerance,
    Result<DIM>* result,
    curandState* states,
    bool save_trajectories);

  template <typename Function, int DIM>
  Result<DIM> launch(
    int N,
    int PSO_ITER,
    int MAX_ITER,
    double upper,
    double lower,
    double* pso_results_device,
    double* hostResults,
    double* deviceTrajectory,
    int requiredConverged,
    double tolerance,
    bool save_trajectories,
    float& ms_opt,
    const std::string& fun_name,
    curandState* states,
    int run);
}*/

namespace zeus {

/*
bool
askUser2saveTrajectories()
{
  std::cout << "Save optimization trajectories? (y/n): ";
  char ans;
  std::cin >> ans;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  return (ans == 'y' || ans == 'Y');
}

void
createOutputDirs(const std::string& path)
{
  std::filesystem::create_directories(path);
}

cudaError_t
writeTrajectoryData(double* hostTrajectory,
                    int N,
                    int MAX_ITER,
                    int DIM,
                    const std::string& fun_name,
                    const std::string& basePath)
{
  // construct the directory path and create it.
  std::string dirPath = basePath + "/" + fun_name + "/" + std::to_string(DIM) +
                        "d/" + std::to_string(MAX_ITER * N) + "/trajectories";
  std::filesystem::create_directories(dirPath);
  // createOutputDirs(dirPath);

  // the final filename.
  std::string filename = dirPath + "/" + std::to_string(MAX_ITER) + "it_" +
                         std::to_string(N) + ".tsv";

  std::ofstream stepOut(filename);
  stepOut << "OptIndex\tStep";
  for (int d = 0; d < DIM; d++)
    stepOut << "\tX_" << d;
  stepOut << "\n";
  stepOut << std::scientific << std::setprecision(17);
  for (int i = 0; i < N; i++) {
    for (int it = 0; it < MAX_ITER; it++) {
      stepOut << i << "\t" << it;
      for (int d = 0; d < DIM; d++) {
        stepOut << "\t" << hostTrajectory[i * (MAX_ITER * DIM) + it * DIM + d];
      }
      stepOut << "\n";
    }
  }
  stepOut.close();
  return cudaSuccess;
}


double
calculate_euclidean_error(const std::string fun_name,
                          const double* coordinates,
                          const int dim)
{
  double sum_sq = 0.0;

  if (fun_name == "rosenbrock") {
    for (int i = 0; i < dim; i++) {
      double diff = coordinates[i] - 1.0;
      sum_sq += diff * diff;
    }
  } else if (fun_name == "goldstein_price") {
    // Goldstein–Price is only defined in 2D (minimum at (0, -1))
    if (dim != 2) {
      fprintf(stderr, "Error: goldstein_price only defined for dim = 2\n");
      return NAN;
    }
    double dx = coordinates[0] - 0.0;
    double dy = coordinates[1] - (-1.0);
    sum_sq = dx * dx + dy * dy;
  } else if (fun_name == "rastrigin" ||
             fun_name == "ackley") { // both rastrigin and ackley have the same
                                     // coordinates for the global minimum
    for (int i = 0; i < dim; ++i) {
      sum_sq += coordinates[i] * coordinates[i];
    }
  }
  return std::sqrt(sum_sq);
} // end calculate_euclidean_error

// make it write to std::cout + dump to file
template <int DIM>
Convergence
dump_data_2_file(const Result<DIM>* h_results,
                 const std::string fun_name,
                 const int N,
                 const int PSO_ITER,
                 const int run)
{
    Convergence result;

    std::string tab = "\t";
    int actually_converged = 0;
    int countConverged = 0, surrender = 0, stopped = 0;
    for (int i = 0; i < N; ++i) {
        //outfile << fun_name << tab << run << tab << i << tab << std::scientific; 
        if (h_results[i].status == 1) {
            countConverged++;
            //outfile << 1 << tab;
            double error = calculate_euclidean_error(fun_name,  h_results[i].coordinates, DIM);
            if(error < 0.5) {actually_converged++;}
            //outfile << h_results[i].iter << tab << h_results[i].fval << tab << h_results[i].gradientNorm;
            //for(int d = 0; d < DIM; ++d) { outfile << "\t"<< h_results[i].coordinates[d]; }
            //outfile << std::endl;
        } else if(h_results[i].status == 2) { // particle was stopped early
            stopped++;
            //outfile << 2 << tab;
            //printf("Thread %d was stopped early (iter=%d)\n", i, h_results[i].iter);
        } else {
            surrender++;
            //outfile << 0 << tab;
        }
        //outfile << h_results[i].iter << tab << h_results[i].fval << tab << h_results[i].gradientNorm;
        //for(int d = 0; d < DIM; ++d) { outfile << "\t"<< h_results[i].coordinates[d]; }
        //outfile << std::endl;
    }
    result.actual = actually_converged;
    result.claimed = countConverged;
    result.surrendered = surrender;
    result.stopped = stopped;
    return result;
    //std::cout << "\ndumped data 2 "<< filename << "\n"<<countConverged <<" converged, "<<stopped << " stopped early, "<<surrender<<" surrendered\n"; 
    //printf("\ndumped data 2 %s\n%d converged, %d stopped early, %d surrendered\n",filename.c_str(),countConverged, stopped, surrender);
}
*/

void
append_results_2_tsv(const int dim,
                     const int N,
                     const std::string fun_name,
                     float ms_init,
                     float ms_pso,
                     float ms_opt,
                     float ms_rand,
                     const int max_iter,
                     const int pso_iter,
                     const double error,
                     const double globalMin,
                     double* hostCoordinates,
                     const int idx,
                     const int status,
                     const double norm,
                     const int run,
                     const int claimed,
                     const int actual,
                     const int surrendered,
                     const int stopped)
{
  std::string filename = "zeus_" + std::to_string(dim) + "d_results.tsv";
  std::ofstream outfile(filename, std::ios::app);

  bool file_exists = std::filesystem::exists(filename);
  bool file_empty =
    file_exists ? (std::filesystem::file_size(filename) == 0) : true;
  // std::ofstream outfile(filename, std::ios::app);
  if (!outfile.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }
  // if file is new or empty, let us write the header
  if (file_empty) {
    outfile << "fun\trun\tN\tclaimed\tactual\tsurrender\tstopped\tidx\tstatus\t"
               "bfgs_iter\tpso_iter\ttime\terror\tfval\tnorm";
    for (int i = 0; i < dim; i++)
      outfile << "\tcoord_" << i;
    outfile << std::endl;
  } // end if file is empty

  double time_seconds = std::numeric_limits<double>::infinity();
  if (pso_iter > 0) {
    time_seconds = (ms_init + ms_pso + ms_opt + ms_rand);
    // printf("total time = pso + bfgs = total time = %0.4f ms\n",
    // time_seconds);
  } else {
    time_seconds = (ms_opt + ms_rand);
    // printf("bfgs time = total time = %.4f ms\n", time_seconds);
  }
  outfile << fun_name << "\t" << run << "\t" << N << "\t" << claimed << "\t"
          << actual << "\t" << surrendered << "\t" << stopped << "\t" << idx
          << "\t" << status << "\t" << max_iter << "\t" << pso_iter << "\t"
          << time_seconds << "\t" << std::scientific << error << "\t"
          << globalMin << "\t" << norm << "\t";
  for (int i = 0; i < dim; i++) {
    outfile << hostCoordinates[i];
    if (i < dim - 1)
      outfile << "\t";
  }
  outfile << "\n";
  outfile.close();
  // printf("results are saved to %s", filename.c_str());
} // end append_results_2_tsv



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
    &minGridSize, &blockSize, bfgs::optimizeKernel<Function, DIM, 128>, 0, N);
  float ms_rand = 0.0f;
  curandState* states = initialize_states(N, seed, ms_rand);
  // printf("Recommended block size: %d\n", blockSize);
  bool save_trajectories = util::askUser2saveTrajectories();
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
                                           run);
  if (PSO_ITER > 0) { // optimzation routine is finished, so we can free that
                      // array on the device
    cudaFree(pso_results_device);
  }

  double error = util::calculate_euclidean_error(fun_name, best.coordinates, DIM);
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
