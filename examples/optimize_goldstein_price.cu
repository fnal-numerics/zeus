#include "zeus.cuh"
#include "goldstein_price.hpp"
#include "optimization_utils.hpp"

using namespace zeus_examples;

int
main(int argc, char* argv[])
{
  OptimizationParams params;
  if (!parse_args(argc, argv, params)) {
    return 1;
  }

  print_banner("Goldstein-Price Function Optimization", 2);
  print_params(params);

  util::setStackSize();

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
                           params.run_id,
                           params.parallel,
                           params.prng_type,
                           params.trajectory_file,
                           params.nzerosteps);

  print_result(result.status, result.fval, result.gradientNorm,
               result.iter, result.ms_opt, result.coordinates, 2);

  return 0;
}
