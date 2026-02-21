#include "util.hpp"

namespace util {

  bool
  askUserToSaveTrajectories()
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

  double
  calculateEuclideanError(const std::string fun_name,
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
    } else { // if (fun_name == "rastrigin" ||
      //    fun_name ==
      //      "ackley") { // both rastrigin and ackley have the same
      //                  // coordinates for the global minimum
      for (int i = 0; i < dim; ++i) {
        sum_sq += coordinates[i] * coordinates[i];
      }
    }
    return std::sqrt(sum_sq);
  } // end calculateEuclideanError
  /*
  template <std::size_t DIM> void
  appendResultsToTsv(const int dim,
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
                       const Result<DIM>& best)
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
      outfile <<
  "\tBFGS_ms_per_call\tBFGS_calls_per_thread_mean\tBFGS_fraction\tBFGS_block95\tBFGS_serialized"
        <<
  "fun\trun\tN\tkernel\tms_per_call\tcalls_per_thread_mean\tad_fraction\tblock95\tserialized\tBFGS_ms_per_call\tBFGS_call_per_thread_mean\tBFGS_fraction\tBFGS_block95\tBFGS_serialized\tclaimed\tactual\tsurrender\tstopped\tidx\tstatus\t"
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
    outfile << fun_name << "\t" << run << "\t" << N << "\t" << ms_opt << "\t"
          << best.ad.ms_per_call << "\t" << best.ad.calls_per_thread_mean <<"\t"
  <<best.ad.fraction_of_kernel <<  "\t" << best.ad.block95 << "\t" <<
  best.ad.serialized << "\t"
          << "\t" << best.bfgs.ms_per_call << "\t" <<
  best.bfgs.calls_per_thread_mean << "\t" << best.bfgs.fraction_of_kernel <<
  "\t" << best.bfgs.block95 << "\t" << best.bfgs.serialized << "\t"
          << best.c.claimed << "\t" << best.c.actual << "\t" <<
  best.c.surrendered << "\t" << best.c.stopped << "\t" << best.idx
          << "\t" << best.status << "\t" << max_iter << "\t" << pso_iter << "\t"
          << time_seconds << "\t" << std::scientific << error << "\t"
          << best.fval << "\t" << best.gradientNorm << "\t";
    for (int i = 0; i < dim; i++) {
      outfile << best.coordinates[i];
      if (i < dim - 1)
        outfile << "\t";
    }
    outfile << "\n";
    outfile.close();
  } // end appendResultsToTsv
*/
} // end namespace util
