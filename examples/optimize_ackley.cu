#include <iostream>
#include <iomanip>
#include "zeus.cuh"
#include "sample_functions.hpp"
#include "optimization_utils.hpp"

using namespace zeus_examples;

int main(int argc, char* argv[])
{
  OptimizationParams params;
  if (!parse_args(argc, argv, params)) {
    return 1;
  }

  std::cout << "\n=== Ackley Function Optimization (2D) ===\n\n";
  print_params(params);
  std::cout << "\n";

  util::set_stack_size();
  
  auto f = util::Ackley<2>{};
  auto result = zeus::Zeus(f,
                           params.lower_bound,
                           params.upper_bound,
                           params.num_optimizations,
                           params.max_iterations,
                           params.pso_iterations,
                           params.required_converged,
                           "ackley",
                           params.tolerance,
                           params.seed,
                           params.run_id);

  return 0;
}
