#pragma once

#include <cstdint>         // for uint64_t
#include <cstdio>          // for printf
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>      // for std::sqrt
#include <limits> 

struct Convergence {
  int actual;
  int claimed;
  int surrendered;
  int stopped;
};

template <int DIM>
struct Result {
  int idx;
  // 0 surrender (reached max iterations)
  // 1 if converged
  // 2 stopped_bc_someone_flipped_the_flag
  // 3 cudamemoryallocation failure
  // 4 cudaruntime error
  int status; 
  double fval; // function value
  double gradientNorm;
  double coordinates[DIM];
  int iter;
  Convergence c;
};

namespace util {

  bool askUser2saveTrajectories();


  double calculate_euclidean_error(const std::string fun_name,
                                   const double* coordinates,
                                   const int dim);

  void append_results_2_tsv(const int dim,
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
                            const int stopped);

  template <size_t DIM>
  Convergence
  dump_data_2_file(size_t N, Result<DIM>* h_results,
                   std::string fun_name,
                   const int PSO_ITER,
                   const int run)
  {
    Convergence result;

    std::string tab = "\t";
    int actually_converged = 0;
    int countConverged = 0, surrender = 0, stopped = 0;
    for (int i = 0; i < N; ++i) {
      // outfile << fun_name << tab << run << tab << i << tab <<
      // std::scientific;
      if (h_results[i].status == 1) {
        countConverged++;
        // outfile << 1 << tab;
        double error =
          calculate_euclidean_error(fun_name, h_results[i].coordinates, DIM);
        if (error < 0.5) {
          actually_converged++;
        }
        // outfile << h_results[i].iter << tab << h_results[i].fval << tab <<
        // h_results[i].gradientNorm; for(int d = 0; d < DIM; ++d) { outfile <<
        // "\t"<< h_results[i].coordinates[d]; } outfile << std::endl;
      } else if (h_results[i].status == 2) { // particle was stopped early
        stopped++;
        // outfile << 2 << tab;
        // printf("Thread %d was stopped early (iter=%d)\n", i,
        // h_results[i].iter);
      } else {
        surrender++;
        // outfile << 0 << tab;
      }
      // outfile << h_results[i].iter << tab << h_results[i].fval << tab <<
      // h_results[i].gradientNorm; for(int d = 0; d < DIM; ++d) { outfile <<
      // "\t"<< h_results[i].coordinates[d]; } outfile << std::endl;
    }
    result.actual = actually_converged;
    result.claimed = countConverged;
    result.surrendered = surrender;
    result.stopped = stopped;
    return result;
    // std::cout << "\ndumped data 2 "<< filename << "\n"<<countConverged <<"
    // converged, "<<stopped << " stopped early, "<<surrender<<" surrendered\n";
    // printf("\ndumped data 2 %s\n%d converged, %d stopped early, %d
    // surrendered\n",filename.c_str(),countConverged, stopped, surrender);
  }

}


