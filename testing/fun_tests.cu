#include <cmath>
#include <array>
#include <cuda_runtime.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include "fun.h"
#include <iostream>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Approx;

// Value Test Kernel + Launcher
template <template <int> class F, int D>
__global__ void
valueKernel(const double* x, double* out, F<D> f)
{
  std::array<double, D> xa;
  for (int i = 0; i < D; ++i)
    xa[i] = x[i];
  out[0] = f(xa);
}

template <template <int> class F, int D>
double
run_value(const double x[D])
{
  double *dX, *dO, hO;
  cudaMalloc(&dX, D * sizeof(double));
  cudaMalloc(&dO, sizeof(double));
  cudaMemcpy(dX, x, D * sizeof(double), cudaMemcpyHostToDevice);

  valueKernel<F, D><<<1, 1>>>(dX, dO, F<D>());
  cudaMemcpy(&hO, dO, sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dX);
  cudaFree(dO);
  return hO;
}

// Gradient Test Kernel + Launcher
template <template <int> class F, int D>
__global__ void
gradKernel(const double* x, double* g)
{
  F<D> f;
  std::array<double, D> xa, ga;
  for (int i = 0; i < D; ++i) {
    xa[i] = x[i];
    ga[i] = 0.0;
  }
  dual::calculateGradientUsingAD<decltype(f), D>(f, xa, ga);
  for (int i = 0; i < D; ++i)
    g[i] = ga[i];
}

template <template <int> class F, int D>
void
run_grad(const double x[D], double out[D])
{
  double *dX, *dG;
  cudaMalloc(&dX, D * sizeof(double));
  cudaMalloc(&dG, D * sizeof(double));
  cudaMemcpy(dX, x, D * sizeof(double), cudaMemcpyHostToDevice);

  gradKernel<F, D><<<1, 1>>>(dX, dG);
  cudaMemcpy(out, dG, D * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(dX);
  cudaFree(dG);
}

// TEST_CASEs below

// Value tests
TEST_CASE("Rastrigin@origin", "[fun][value]")
{
  constexpr int D = 2;
  double x[D] = {0.0, 0.0};
  REQUIRE(run_value<util::Rastrigin, D>(x) == Approx(0.0));
}

TEST_CASE("Ackley@origin", "[fun][value]")
{
  constexpr int D = 2;
  double x[D] = {0.0, 0.0};
  REQUIRE(run_value<util::Ackley, D>(x) == Approx(0.0).margin(1e-6));
}

TEST_CASE("Rosenbrock@min", "[fun][value]")
{
  constexpr int D = 2;
  double x[D] = {1.0, 1.0};
  REQUIRE(run_value<util::Rosenbrock, D>(x) == Approx(0.0));
}

// Gradient tests
TEST_CASE("grad_Rastrigin@origin", "[fun][grad]")
{
  constexpr int D = 2;
  double x[D] = {0.0, 0.0};
  double expect[D] = {0.0, 0.0}, got[D];
  run_grad<util::Rastrigin, D>(x, got);
  for (int i = 0; i < D; ++i)
    REQUIRE(got[i] == Approx(expect[i]));
}

TEST_CASE("grad_Ackley@origin", "[fun][grad]")
{
  constexpr int D = 2;
  double x[D] = {-1e-17, 1e-17};
  double expect[D] = {-2.0, 2.0}, got[D];
  run_grad<util::Ackley, D>(x, got);
  for (int i = 0; i < D; ++i) {
    std::cout << "got[" << i << "]: " << got[i] << "expected[" << i
              << "]: " << expect[i] << std::endl;
    REQUIRE(got[i] == Approx(expect[i]));
  }
}

TEST_CASE("grad_Rosenbrock@min", "[fun][grad]")
{
  constexpr int D = 2;
  double x[D] = {1.0, 1.0};
  double expect[D] = {0.0, 0.0}, got[D];
  run_grad<util::Rosenbrock, D>(x, got);
  for (int i = 0; i < D; ++i)
    REQUIRE(got[i] == Approx(expect[i]));
}
