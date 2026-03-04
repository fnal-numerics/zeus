#include <iostream>
#include <array>
#include <cstdlib>
#include <string>

#include "zeus.cuh"
#include "gaussian.hpp"
#include "nn.hpp"

// Dimension for the Gaussian example.
static constexpr std::size_t D = 5;

// ---------------------------------------------------------------------------
// Gaussian minimization
// ---------------------------------------------------------------------------
// Minimises f(x) = 0.5 * x^T C x where C is a D×D equicorrelation matrix
// with diagonal entries 1 + (D-1)*off and off-diagonal entries off.
// The unique global minimum is at x = 0 with f = 0.
static void
run_gaussian(std::size_t N,
             int bfgs,
             int run,
             zeus::PRNGType prng_type,
             std::string_view trajectory_file)
{
  using T = double;
  constexpr T off = T(0.5);

  std::array<std::array<T, D>, D> C;
  for (std::size_t i = 0; i < D; ++i)
    for (std::size_t j = 0; j < D; ++j)
      C[i][j] = (i == j ? T(1) + (D - 1) * off : off);

  Gaussian<D> g{C};

  std::cout << "--- " << D << "D Gaussian minimisation ---\n";
  auto res = zeus::Zeus(g,
                        -5.0,
                        5.0,
                        N,
                        bfgs,
                        10,
                        100,
                        "gaussian",
                        1e-8,
                        42,
                        run,
                        true,
                        prng_type,
                        trajectory_file);
  std::cout << "global minimum: " << res.fval << "  (expected 0.0)\n\n";
}

// ---------------------------------------------------------------------------
// Neural-network weight fitting
// ---------------------------------------------------------------------------
// Fits the weights of a small feedforward network NeuralNet<In,H,Out> to a
// single randomly-generated training example (x0, y0).  The problem is
// deliberately under-determined (one sample, many weights) to stress-test the
// PSO phase on a high-dimensional, flat loss surface.
static void
run_neural_net(std::size_t N,
               int bfgs,
               int run,
               zeus::PRNGType prng_type,
               std::string_view trajectory_file)
{
  constexpr size_t In = 5;
  constexpr size_t H = 15;
  constexpr size_t Out = 10;
  constexpr size_t P = NeuralNet<In, H, Out>::P;

  // Build a single toy training example.
  std::array<double, In> x0;
  std::array<double, Out> y0;
  for (size_t i = 0; i < In; ++i)
    x0[i] = double(i);
  for (size_t k = 0; k < Out; ++k)
    y0[k] = (std::rand() / double(RAND_MAX) - 0.5) * 0.5;

  NeuralNet<In, H, Out> objective{x0, y0};

  std::cout << "--- " << P << "D neural-net weight fitting ---\n";
  auto res = zeus::Zeus(objective,
                        -20.0,
                        20.0,
                        N,
                        bfgs,
                        2,
                        10,
                        "neural_net",
                        1e-6,
                        42,
                        run,
                        true,
                        prng_type,
                        trajectory_file);
  std::cout << "final loss: " << res.fval << "\n\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
  if (argc < 4) {
    std::cerr
      << "Usage: " << argv[0]
      << " <num_optimizations> <max_bfgs_iter> <run_id> "
         "[--save-trajectories <filename>] [--prng <xorwow|philox|sobol>]\n";
    return 1;
  }
  const std::size_t N = std::stoul(argv[1]);
  const int bfgs = std::stoi(argv[2]);
  const int run = std::stoi(argv[3]);

  std::string trajectory_file;
  zeus::PRNGType prng_type = zeus::PRNGType::XORWOW;
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--save-trajectories" && i + 1 < argc) {
      trajectory_file = argv[++i];
    } else if (arg == "--prng" && i + 1 < argc) {
      std::string val = argv[++i];
      if (val == "xorwow")
        prng_type = zeus::PRNGType::XORWOW;
      else if (val == "philox")
        prng_type = zeus::PRNGType::PHILOX;
      else if (val == "sobol")
        prng_type = zeus::PRNGType::SOBOL;
      else {
        std::cerr << "Unknown PRNG type: " << val
                  << ". Using default xorwow.\n";
      }
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  util::setStackSize();

  run_gaussian(N, bfgs, run, prng_type, trajectory_file);
  run_neural_net(N, bfgs, run, prng_type, trajectory_file);

  return 0;
}
