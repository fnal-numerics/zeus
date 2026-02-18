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

  std::cout << "\n=== Goldstein-Price Function Optimization (2D) ===\n\n";
  print_params(params);
  std::cout << "\n";

  util::set_stack_size();
  
  auto f = util::GoldsteinPrice<2>{};
  auto result = zeus::Zeus(f,
                           params.lower_bound,
                           params.upper_bound,
                           params.num_optimizations,
                           params.max_iterations,
                           params.pso_iterations,
                           params.required_converged,
                           "goldstein_price",
                           params.tolerance,
                           params.seed,
                           params.run_id);

  std::cout << "\n=== Optimization Result ===\n";
  std::cout << "Status: " << result.status << " ";
  if (result.status == 0) std::cout << "(Surrendered - max iterations reached)";
  else if (result.status == 1) std::cout << "(Converged)";
  else if (result.status == 2) std::cout << "(Stopped by flag)";
  else if (result.status == 3) std::cout << "(CUDA memory allocation failure)";
  else if (result.status == 4) std::cout << "(CUDA runtime error)";
  std::cout << "\n";
  std::cout << "Function value: " << std::scientific << result.fval << std::defaultfloat << "\n";
  std::cout << "Gradient norm: " << std::scientific << result.gradientNorm << std::defaultfloat << "\n";
  std::cout << "Iterations: " << result.iter << "\n";
  std::cout << "Optimization time: " << result.ms_opt << " ms\n";
  std::cout << "Best coordinates: (";
  for (int i = 0; i < 2; ++i) {
    std::cout << std::setprecision(10) << result.coordinates[i];
    if (i < 1) std::cout << ", ";
  }
  std::cout << ")\n\n";

  return 0;
}
