#include "optimization_utils.hpp"

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>

namespace zeus_examples {

bool
parse_args(int argc, char* argv[], OptimizationParams& params)
{
  if (argc < 10) {
    std::cerr << "Usage: " << argv[0]
              << " <lower_bound> <upper_bound> <max_iter> <pso_iters> "
                 "<converged> <num_optimizations> <tolerance> <seed> <run> "
                 "[--parallel] [--save-trajectories <filename>] [--prng <xorwow|philox|sobol>]"
                 " [--nzerosteps <n>]\n";
    return false;
  }

  try {
    params.lower_bound = std::atof(argv[1]);
    params.upper_bound = std::atof(argv[2]);
    params.max_iterations = std::stoi(argv[3]);
    params.pso_iterations = std::stoi(argv[4]);
    params.required_converged = std::stoi(argv[5]);
    params.num_optimizations = std::stoi(argv[6]);
    params.tolerance = std::stod(argv[7]);
    params.seed = std::stoi(argv[8]);
    params.run_id = std::stoi(argv[9]);

    for (int i = 10; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--parallel") {
        params.parallel = true;
      } else if (arg == "--save-trajectories" && i + 1 < argc) {
        params.trajectory_file = argv[++i];
      } else if (arg == "--prng" && i + 1 < argc) {
        std::string val = argv[++i];
        if (val == "xorwow")       params.prng_type = zeus::PRNGType::XORWOW;
        else if (val == "philox")  params.prng_type = zeus::PRNGType::PHILOX;
        else if (val == "sobol")   params.prng_type = zeus::PRNGType::SOBOL;
        else {
          std::cerr << "Unknown PRNG type: " << val << ". Using default xorwow.\n";
        }
      } else if (arg == "--nzerosteps" && i + 1 < argc) {
        params.nzerosteps = std::stoi(argv[++i]);
      } else {
        std::cerr << "Unknown or incomplete argument: " << arg << "\n";
        return false;
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "Error parsing arguments: " << e.what() << "\n";
    return false;
  }

  return true;
}

void
print_params(const OptimizationParams& params)
{
  std::cout << std::setprecision(10)
            << "Bounds: [" << params.lower_bound << ", " << params.upper_bound << "]\n"
            << "Max iterations: " << params.max_iterations << "\n"
            << "PSO iterations: " << params.pso_iterations << "\n"
            << "Required converged: " << params.required_converged << "\n"
            << "Number of optimizations: " << params.num_optimizations << "\n"
            << "Tolerance: " << params.tolerance << "\n"
            << "Seed: " << params.seed << "\n"
            << "Run ID: " << params.run_id << "\n"
            << "Algorithm: " << (params.parallel ? "parallel" : "sequential") << "\n";
  if (!params.trajectory_file.empty()) {
    std::cout << "Trajectory file: " << params.trajectory_file << "\n";
  }
  if (params.nzerosteps > 0) {
    std::cout << "Max consecutive zero steps: " << params.nzerosteps << "\n";
  }
}

void
print_banner(const std::string& title, int dim)
{
  std::cout << "\n=== " << title << " (" << dim << "D) ===\n\n";
}

void
print_result(int status, double fval, double gradientNorm,
             int iter, double ms_opt,
             const double* coords, int dim)
{
  std::cout << "\n=== Optimization Result ===\n";
  std::cout << "Status: " << status << " ";
  if (status == 0)
    std::cout << "(Surrendered - max iterations reached)";
  else if (status == 1)
    std::cout << "(Converged)";
  else if (status == 2)
    std::cout << "(Stopped by flag)";
  else if (status == 3)
    std::cout << "(CUDA memory allocation failure)";
  else if (status == 4)
    std::cout << "(CUDA runtime error)";
  else if (status == 5)
    std::cout << "(Non-finite values encountered)";
  else if (status == 6)
    std::cout << "(Consecutive zero-step limit reached)";
  std::cout << "\n";
  std::cout << "Function value: " << std::scientific << fval
            << std::defaultfloat << "\n";
  std::cout << "Gradient norm: " << std::scientific << gradientNorm
            << std::defaultfloat << "\n";
  std::cout << "Iterations: " << iter << "\n";
  std::cout << "Optimization time: " << ms_opt << " ms\n";
  std::cout << "Best coordinates: (";
  for (int i = 0; i < dim; ++i) {
    std::cout << std::setprecision(10) << coords[i];
    if (i < dim - 1)
      std::cout << ", ";
  }
  std::cout << ")\n\n";
}

}  // namespace zeus_examples
