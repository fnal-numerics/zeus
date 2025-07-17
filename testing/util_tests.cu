#include <cuda_runtime.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
// #include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "fun.h"

#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Approx;

// extern device functions from util
extern "C" {
__device__ void vector_add(const double* a,
                           const double* b,
                           double* r,
                           int size);
__device__ void vector_scale(const double* a, double s, double* r, int dim);
}

// kernels to invoke the the functions
__global__ void
testVectorAddKernel(const double* a, const double* b, double* r, int n)
{
  vector_add(a, b, r, n);
}
__global__ void
testVectorScaleKernel(const double* a, double s, double* r, int n)
{
  vector_scale(a, s, r, n);
}

TEST_CASE("vector_add works on small arrays", "[utils][vector]")
{
  constexpr int N = 5;
  double hA[N] = {1, 2, 3, 4, 5}, hB[N] = {5, 4, 3, 2, 1}, hR[N] = {0};
  double *dA, *dB, *dR;
  cudaMalloc(&dA, N * sizeof(double));
  cudaMalloc(&dB, N * sizeof(double));
  cudaMalloc(&dR, N * sizeof(double));
  cudaMemcpy(dA, hA, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB, N * sizeof(double), cudaMemcpyHostToDevice);
  testVectorAddKernel<<<1, 1>>>(dA, dB, dR, N);
  cudaMemcpy(hR, dR, N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i)
    REQUIRE(hR[i] == Approx(hA[i] + hB[i]));

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dR);
}

TEST_CASE("vector_scale works for small arrays", "[utils][vector]")
{
  constexpr int N = 5;
  double hA[N] = {1, 2, 3, 4, 5}, hR[N] = {0};
  double *dA, *dR;
  cudaMalloc(&dA, N * sizeof(double));
  cudaMalloc(&dR, N * sizeof(double));
  cudaMemcpy(dA, hA, N * sizeof(double), cudaMemcpyHostToDevice);
  testVectorScaleKernel<<<1, 1>>>(dA, 2.0, dR, N);
  cudaMemcpy(hR, dR, N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i)
    REQUIRE(hR[i] == Approx(hA[i] * 2.0));

  cudaFree(dA);
  cudaFree(dR);
}
