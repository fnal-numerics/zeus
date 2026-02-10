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

struct Metrics {
  // label for printing "AD", "BFGS"
  const char* label = "";
  double ms_per_call = 0.0;
  double calls_per_thread_mean = 0.0;
  double fraction_of_kernel = 0.0; // avg-per-thread / kernel
  double block95 = 0.0;
  double serialized = 0.0; // sum over threads / kernel
  double fraction_of_thread=0.0;
};

struct Convergence {
  int actual = 0;
  int claimed = 0;
  int surrendered = 0;
  int stopped = 0;
};

template <std::size_t DIM>
struct Result {
  int idx = -1;
  // 0 surrender (reached max iterations)
  // 1 if converged
  // 2 stopped_bc_someone_flipped_the_flag
  // 3 cudamemoryallocation failure
  // 4 cudaruntime error
  int status = -1; 
  double fval = 37.0; // function value
  double gradientNorm = 0.0;
  double coordinates[DIM]{};
  int iter=0;
  Convergence c;
  double ms_opt = 0.0; 

  Metrics ad;
  Metrics bfgs;
};

namespace util {

  bool askUser2saveTrajectories();


  double calculate_euclidean_error(const std::string fun_name,
                                   const double* coordinates,
                                   const int dim);

  template <std::size_t DIM>
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
                            const int run,
			    const Result<DIM>& best) {
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
      outfile
      << "fun\trun\tN\tkernel"
      << "\tad_ms_per_call\tad_calls_per_thread_mean\tad_fraction\tad_block95\tad_serialized"
      << "\tbfgs_ms_per_call\tbfgs_calls_per_thread_mean\tbfgs_fraction\tbfgs_block95\tbfgs_serialized"
      << "\tclaimed\tactual\tsurrender\tstopped\tidx\tstatus\tbfgs_iter\tpso_iter\ttime\terror\tfval\tnorm";
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
    outfile << fun_name << "\t" << run << "\t" << N << "\t" << ms_opt << "\t"
          << best.ad.ms_per_call << "\t" << best.ad.calls_per_thread_mean <<"\t" <<best.ad.fraction_of_thread <<  "\t" << best.ad.block95 << "\t" << best.ad.serialized << "\t"
          << best.bfgs.ms_per_call << "\t" << best.bfgs.calls_per_thread_mean << "\t" << best.bfgs.fraction_of_thread << "\t" << best.bfgs.block95 << "\t" << best.bfgs.serialized << "\t"
          << best.c.claimed << "\t" << best.c.actual << "\t" << best.c.surrendered << "\t" << best.c.stopped << "\t" << best.idx
          << "\t" << best.status << "\t" << max_iter << "\t" << pso_iter << "\t"
          << time_seconds << "\t" << std::scientific << error << "\t"
          << best.fval << "\t" << best.gradientNorm << "\t";
    std::cout << "claimed: " << best.c.claimed << "\tactual : " << best.c.actual << "\tsurrendered: " << best.c.surrendered << "\tstopped: " << best.c.stopped <<"\n";
    for (int i = 0; i < dim; i++) {
      outfile << best.coordinates[i];
      if (i < dim - 1)
        outfile << "\t";
    }
    outfile << "\n";
    outfile.close();
    // printf("results are saved to %s", filename.c_str());
  } // end append_results_2_tsv

			    
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


