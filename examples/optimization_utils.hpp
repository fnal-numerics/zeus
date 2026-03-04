#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>
#include "../include/traits.hpp"

namespace zeus_examples {

/// Parameters for optimization
struct OptimizationParams {
  double lower_bound;
  double upper_bound;
  int max_iterations;
  int pso_iterations;
  int required_converged;
  int num_optimizations;
  double tolerance;
  int seed;
  int run_id;
  zeus::PRNGType prng_type = zeus::PRNGType::XORWOW; // default
  std::string trajectory_file; // defaults to empty
};

/// Parse command-line arguments into OptimizationParams
inline bool parse_args(int argc, char* argv[], OptimizationParams& params)
{
  if (argc < 10) {
    std::cerr << "Usage: " << argv[0]
              << " <lower_bound> <upper_bound> <max_iter> <pso_iters> "
                 "<converged> <num_optimizations> <tolerance> <seed> <run> "
                 "[--save-trajectories <filename>] [--prng <xorwow|philox|sobol>]\n";
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

    // Parse optional arguments
    for (int i = 10; i < argc; ++i) {
      std::string arg = argv[i];
      if (arg == "--save-trajectories" && i + 1 < argc) {
        params.trajectory_file = argv[++i];
      } else if (arg == "--prng" && i + 1 < argc) {
        std::string val = argv[++i];
        if (val == "xorwow") params.prng_type = zeus::PRNGType::XORWOW;
        else if (val == "philox") params.prng_type = zeus::PRNGType::PHILOX;
        else if (val == "sobol") params.prng_type = zeus::PRNGType::SOBOL;
        else {
          std::cerr << "Unknown PRNG type: " << val << ". Using default xorwow.\n";
        }
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

/// Print optimization parameters to stdout
inline void print_params(const OptimizationParams& params)
{
  std::cout << std::setprecision(10)
            << "Bounds: [" << params.lower_bound << ", " << params.upper_bound << "]\n"
            << "Max iterations: " << params.max_iterations << "\n"
            << "PSO iterations: " << params.pso_iterations << "\n"
            << "Required converged: " << params.required_converged << "\n"
            << "Number of optimizations: " << params.num_optimizations << "\n"
            << "Tolerance: " << params.tolerance << "\n"
            << "Seed: " << params.seed << "\n"
            << "Run ID: " << params.run_id << "\n";
  if (!params.trajectory_file.empty()) {
    std::cout << "Trajectory file: " << params.trajectory_file << "\n";
  }
}

}  // namespace zeus_examples
