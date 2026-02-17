#pragma once

#include <string>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cstdlib>

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
};

/// Parse command-line arguments into OptimizationParams
inline bool parse_args(int argc, char* argv[], OptimizationParams& params)
{
  if (argc != 10) {
    std::cerr << "Usage: " << argv[0]
              << " <lower_bound> <upper_bound> <max_iter> <pso_iters> "
                 "<converged> <num_optimizations> <tolerance> <seed> <run>\n";
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
}

}  // namespace zeus_examples
