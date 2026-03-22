#pragma once

#include <string>
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
  bool parallel = false;
  zeus::PRNGType prng_type = zeus::PRNGType::XORWOW; // default
  std::string trajectory_file; // defaults to empty
  int nzerosteps = 0;           // 0 = disabled
};

/// Parse command-line arguments into OptimizationParams
bool parse_args(int argc, char* argv[], OptimizationParams& params);

/// Print optimization parameters to stdout
void print_params(const OptimizationParams& params);

/// Print the optimization banner, e.g. "=== Ackley Function Optimization (2D) ==="
void print_banner(const std::string& title, int dim);

/// Print the Zeus result block (status, fval, gradient norm, iterations, time, coordinates)
void print_result(int status, double fval, double gradientNorm,
                  int iter, double ms_opt,
                  const double* coords, int dim);

}  // namespace zeus_examples
