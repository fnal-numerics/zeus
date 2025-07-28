//#include "fun.h"
//#include "duals.cuh"
//#include "utils.cuh"
//#include "pso.cuh"
//#include "bfgs.cuh"
#include <vector>
#include <iostream>
#include "zeus.cuh"


template <int dim>
void
selectAndRunOptimization(double lower,
                         double upper,
                         double* hostResults,
                         int N,
                         int MAX_ITER,
                         int PSO_ITERS,
                         int requiredConverged,
                         double tolerance,
                         int seed,
                         const int run)
{
  double lo = lower;
  double hi = upper;

  int choice;
  std::cout << "\nSelect function to optimize:\n"
            << " 1. Rosenbrock\n"
            << " 2. Rastrigin\n"
            << " 3. Ackley\n";
  if constexpr (dim == 2) {
    std::cout << " 4. GoldsteinPrice\n"
              << " 5. Eggholder\n"
              << " 6. Himmelblau\n";
  }
  std::cout << " 7. Custom (user-defined objective)\n"
            << "Choice: ";
  std::cin >> choice;
  std::cin.ignore();
  std::array<double, dim> x0;
  for(int d=0;d<dim;d++)
    x0[d] = 0.5*(lo+hi);
  switch (choice) {
    case 1: {
      std::cout << "\n\n\tRosenbrock Function\n\n";
      auto f = util::Rosenbrock<dim>{};
      auto result = zeus::Zeus(f,x0, // deduce F = util::Rosenbrock<dim>
                               lo,
                               hi,
                               hostResults,
                               N,
                               MAX_ITER,
                               PSO_ITERS,
                               requiredConverged,
                               "rosenbrock",
                               tolerance,
                               seed,
                               run);
      break;
    }
    case 2: {
      std::cout << "\n\n\tRastrigin Function\n\n";
      auto f = util::Rastrigin<dim>{};
      auto result = zeus::Zeus(f,x0,
                               lo,
                               hi,
                               hostResults,
                               N,
                               MAX_ITER,
                               PSO_ITERS,
                               requiredConverged,
                               "rastrigin",
                               tolerance,
                               seed,
                               run);
      break;
    }
    case 3: {
      std::cout << "\n\n\tAckley Function\n\n";
      auto f = util::Ackley<dim>{};
      auto result = zeus::Zeus(f,x0,
                               lo,
                               hi,
                               hostResults,
                               N,
                               MAX_ITER,
                               PSO_ITERS,
                               requiredConverged,
                               "ackley",
                               tolerance,
                               seed,
                               run);
      break;
    }
    /*case 4:
      if constexpr (dim == 2) {
        std::cout << "\n\n\tGoldstein-Price Function\n\n";
        auto f = util::GoldsteinPrice<dim>{};
        auto result = zeus::Zeus(f,
                                 lo,
                                 hi,
                                 hostResults,
                                 N,
                                 MAX_ITER,
                                 PSO_ITERS,
                                 requiredConverged,
                                 "goldstein_price",
                                 tolerance,
                                 seed,
                                 run);
        break;
      }
    case 5:
      if constexpr (dim == 2) {
        std::cout << "\n\n\tEggholder Function\n\n";
        auto f = util::Eggholder<dim>{};
        auto result = zeus::Zeus(f,
                                 lo,
                                 hi,
                                 hostResults,
                                 N,
                                 MAX_ITER,
                                 PSO_ITERS,
                                 requiredConverged,
                                 "eggholder",
                                 tolerance,
                                 seed,
                                 run);
        break;
      }
    case 6:
      if constexpr (dim == 2) {
        std::cout << "\n\n\tHimmelblau Function\n\n";
        auto f = util::Himmelblau<dim>{};
        auto result = zeus::Zeus(f,
                                 lo,
                                 hi,
                                 hostResults,
                                 N,
                                 MAX_ITER,
                                 PSO_ITERS,
                                 requiredConverged,
                                 "himmelblau",
                                 tolerance,
                                 seed,
                                 run);
        break;
      }
    case 7: {
      std::cout << "\n\n\tCustom Function\n\n"
                << "Please implement a free function with signature\n"
                << "  double myfun(const double* x)\n"
                << "and then recompile.\n\n";
      break;
    }*/
    default:
      std::cerr << "Invalid choice\n";
  }
}

// #ifndef UNIT_TEST
// #ifndef NO_MAIN
#if !defined(UNIT_TEST) && !defined(TABLE_GEN)
int
main(int argc, char* argv[])
{
  if (argc != 10) {
    std::cerr
      << "Usage: " << argv[0]
      << " <lower_bound> <upper_bound> <max_iter> <pso_iters> <converged> "
         "<number_of_optimizations> <tolerance> <seed> <run>\n";
    return 1;
  }
  double lower = std::atof(argv[1]);
  double upper = std::atof(argv[2]);
  int MAX_ITER = std::stoi(argv[3]);
  int PSO_ITERS = std::stoi(argv[4]);
  int requiredConverged = std::stoi(argv[5]);
  int N = std::stoi(argv[6]);
  double tolerance = std::stod(argv[7]);
  int seed = std::stoi(argv[8]);
  int run = std::stoi(argv[9]);

  std::cout << "Tolerance: " << std::setprecision(10) << tolerance
            << "\nseed: " << seed << "\n";

  // const size_t N =
  // 128*4;//1024*128*16;//pow(10,5.5);//128*1024*3;//*1024*128;
  const int dim = 2;
  double hostResults[N]; // = new double[N];
  std::cout << "number of optimizations = " << N << " max_iter = " << MAX_ITER
            << " dim = " << dim << std::endl;

  double f0 = 333777; // initial function value
  util::set_stack_size();
  char cont = 'y';
  while (cont == 'y' || cont == 'Y') {
    for (int i = 0; i < N; i++) {
      hostResults[i] = f0;
    }
    selectAndRunOptimization<dim>(lower,
                                  upper,
                                  hostResults,
                                  N,
                                  MAX_ITER,
                                  PSO_ITERS,
                                  requiredConverged,
                                  tolerance,
                                  seed,
                                  run);
    std::cout << "\nDo you want to optimize another function? (y/n): ";
    std::cin >> cont;
    std::cin.ignore();
  }

  // for(int i=0; i<N; i++) {
  //     hostResults[i] = f0;
  // }
  // selectAndRunOptimization<dim>(lower, upper, hostResults, hostIndices,
  // hostCoordinates, N, MAX_ITER);
  return 0;
}
#endif
