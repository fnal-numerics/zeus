#include <cuda_runtime.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_approx.hpp>
#include "catch2/catch_template_test_macros.hpp"
#include "catch2/catch_test_macros.hpp"
// #include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "utils.cuh"

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


__host__ double* cleanupAndFail(double* a, double* b, double* c,
                                double* d, double* e, double* f)
{
    util::freeCudaPtrs(a, b, c, d, e, f);
    return KERNEL_ERROR;
}

TEST_CASE("Sentinel pointers are unique and stable")
{
    REQUIRE(MALLOC_ERROR  != nullptr);
    REQUIRE(KERNEL_ERROR  != nullptr);
    REQUIRE(MALLOC_ERROR  != KERNEL_ERROR);

    // Same addresses we expect:
    REQUIRE(MALLOC_ERROR  == const_cast<double*>(&malloc_error));
    REQUIRE(KERNEL_ERROR  == const_cast<double*>(&kernel_error));
}

TEST_CASE("freeCudaPtrs actually frees device allocations")
{
    double *p1 = nullptr, *p2 = nullptr, *p3 = nullptr;

    REQUIRE(cudaMalloc(&p1, sizeof(double)) == cudaSuccess);
    REQUIRE(cudaMalloc(&p2, sizeof(double)) == cudaSuccess);
    REQUIRE(cudaMalloc(&p3, sizeof(double)) == cudaSuccess);

    util::freeCudaPtrs(p1, p2, p3);

    // Second free must fail because the pointer was already released.
    REQUIRE(cudaFree(p1) != cudaSuccess);
    REQUIRE(cudaFree(p2) != cudaSuccess);
    REQUIRE(cudaFree(p3) != cudaSuccess);

    // Clear last error so it doesn’t bleed into other tests
    cudaGetLastError();
}

TEST_CASE("cleanupAndFail returns the kernel-error sentinel")
{
    double *d1 = nullptr, *d2 = nullptr, *d3 = nullptr,
           *d4 = nullptr, *d5 = nullptr, *d6 = nullptr;

    // Allocate six tiny buffers
    cudaMalloc(&d1, sizeof(double));
    cudaMalloc(&d2, sizeof(double));
    cudaMalloc(&d3, sizeof(double));
    cudaMalloc(&d4, sizeof(double));
    cudaMalloc(&d5, sizeof(double));
    cudaMalloc(&d6, sizeof(double));

    double* result = cleanupAndFail(d1, d2, d3, d4, d5, d6);

    REQUIRE(result == KERNEL_ERROR);
}
