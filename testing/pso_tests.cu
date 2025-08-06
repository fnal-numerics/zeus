// pso_tests.cu

#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <array>
#include <algorithm>

#include "utils.cuh"
#include "pso.cuh"
#include "fun.h"
#include "bfgs.cuh"

using namespace zeus;

// helper to copy device→host
template <typename T>
void
copyDevice(const T* dptr, T* hptr, size_t n)
{
  cudaMemcpy(hptr, dptr, n * sizeof(T), cudaMemcpyDeviceToHost);
}

TEST_CASE("pso::initKernel sets pBest & gBest for util::Rastrigin<2>",
          "[pso][init]")
{
  constexpr int N = 1, DIM = 2;
  const double lower = -5.0, upper = 5.0;
  const uint64_t seed = 42;

  // curand states
  float ms_rand;
  curandState* d_states = bfgs::initialize_states(N, int(seed), ms_rand);

  // allocate PSO buffers
  double *dX, *dV, *dPBestX, *dPBestVal, *dGBestX, *dGBestVal;
  cudaMalloc(&dX, N * DIM * sizeof(double));
  cudaMalloc(&dV, N * DIM * sizeof(double));
  cudaMalloc(&dPBestX, N * DIM * sizeof(double));
  cudaMalloc(&dPBestVal, N * sizeof(double));
  cudaMalloc(&dGBestX, DIM * sizeof(double));
  cudaMalloc(&dGBestVal, sizeof(double));
  {
    double inf = std::numeric_limits<double>::infinity();
    cudaMemcpy(dGBestVal, &inf, sizeof(double), cudaMemcpyHostToDevice);
  }

  // run initKernel<Function,DIM>
  pso::initKernel<util::Rastrigin<DIM>, DIM><<<1, N>>>(util::Rastrigin<DIM>(),
                                                       lower,
                                                       upper,
                                                       dX,
                                                       dV,
                                                       dPBestX,
                                                       dPBestVal,
                                                       dGBestX,
                                                       dGBestVal,
                                                       N,
                                                       seed,
                                                       d_states);
  cudaDeviceSynchronize();

  // copy back
  double hPVal, hGVal;
  double hPX[DIM], hGX[DIM];
  copyDevice(dPBestVal, &hPVal, 1);
  copyDevice(dGBestVal, &hGVal, 1);
  copyDevice(dPBestX, hPX, DIM);
  copyDevice(dGBestX, hGX, DIM);

  // compute expected f(pBestX) on host
  std::array<double, DIM> arr;
  std::copy(hPX, hPX + DIM, arr.begin());
  double expected = util::Rastrigin<DIM>()(arr);

  REQUIRE(hPVal == Approx(expected).margin(1e-6));
  REQUIRE(hGVal == Approx(expected).margin(1e-6));
  for (int d = 0; d < DIM; ++d)
    REQUIRE(hGX[d] == Approx(hPX[d]).margin(1e-6));

  // cleanup
  cudaFree(dX);
  cudaFree(dV);
  cudaFree(dPBestX);
  cudaFree(dPBestVal);
  cudaFree(dGBestX);
  cudaFree(dGBestVal);
  cudaFree(d_states);
}

TEST_CASE("pso::iterKernel inertia‐only updates X and V for 4 particles in 1D",
          "[pso][component][iter]")
{
  constexpr int N = 4, DIM = 1;
  const double lower = 0.0, upper = 1.0;
  const uint64_t seed = 0;

  // init curand states (not actually used when c1=c2=0)
  float ms_rand;
  curandState* d_states = bfgs::initialize_states(N, int(seed), ms_rand);

  // allocate everything
  double *dX, *dV, *dPBestX, *dPBestVal, *dGBestX, *dGBestVal;
  cudaMalloc(&dX, N * DIM * sizeof(double));
  cudaMalloc(&dV, N * DIM * sizeof(double));
  cudaMalloc(&dPBestX, N * DIM * sizeof(double));
  cudaMalloc(&dPBestVal, N * sizeof(double));
  cudaMalloc(&dGBestX, DIM * sizeof(double));
  cudaMalloc(&dGBestVal, sizeof(double));

  // host‐side initial values
  double hX[N] = {0.0, 1.0, 2.0, 3.0};
  double hV[N] = {10.0, 20.0, 30.0, 40.0};
  // personal & global bests start out large so they won't interfere
  double hPBX[N];
  std::copy(hX, hX + N, hPBX);
  double hPBV[N];
  std::fill(hPBV, hPBV + N, 1e6);
  double hGBX[DIM] = {0.0};
  double hGBV = 1e6;

  cudaMemcpy(dX, hX, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dPBestX, hPBX, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dPBestVal, hPBV, N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dGBestX, hGBX, DIM * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dGBestVal, &hGBV, sizeof(double), cudaMemcpyHostToDevice);

  // launch one iteration: w=0.5, c1=c2=0
  pso::iterKernel<util::Rastrigin<DIM>, DIM><<<1, N>>>(util::Rastrigin<DIM>(),
                                                       lower,
                                                       upper,
                                                       0.5, // inertia
                                                       0.0, // cognitive
                                                       0.0, // social
                                                       dX,
                                                       dV,
                                                       dPBestX,
                                                       dPBestVal,
                                                       dGBestX,
                                                       dGBestVal,
                                                       nullptr,
                                                       false,
                                                       N,
                                                       /*iter=*/0,
                                                       seed,
                                                       d_states);
  cudaDeviceSynchronize();

  // 5) copy back and verify: v1 = 0.5*v0, x1 = x0 + v1
  double hX1[N], hV1[N];
  cudaMemcpy(hX1, dX, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hV1, dV, N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; ++i) {
    double expectedV = 0.5 * hV[i];
    double expectedX = hX[i] + expectedV;
    REQUIRE(hV1[i] == Approx(expectedV).margin(1e-12));
    REQUIRE(hX1[i] == Approx(expectedX).margin(1e-12));
  }

  // cleanup
  cudaFree(dX);
  cudaFree(dV);
  cudaFree(dPBestX);
  cudaFree(dPBestVal);
  cudaFree(dGBestX);
  cudaFree(dGBestVal);
  cudaFree(d_states);
}

TEST_CASE("pso::iterKernel with zero w,c1,c2 leaves X unchanged and V zero",
          "[pso][iter]")
{
  constexpr int N = 1, DIM = 2;
  const double lower = -5.0, upper = 5.0;
  const uint64_t seed = 42;

  // curand states & buffers
  float ms_rand;
  curandState* d_states = bfgs::initialize_states(N, int(seed), ms_rand);

  double *dX, *dV, *dPBestX, *dPBestVal, *dGBestX, *dGBestVal;
  cudaMalloc(&dX, N * DIM * sizeof(double));
  cudaMalloc(&dV, N * DIM * sizeof(double));
  cudaMalloc(&dPBestX, N * DIM * sizeof(double));
  cudaMalloc(&dPBestVal, N * sizeof(double));
  cudaMalloc(&dGBestX, DIM * sizeof(double));
  cudaMalloc(&dGBestVal, sizeof(double));

  // init
  pso::initKernel<util::Rastrigin<DIM>, DIM><<<1, N>>>(util::Rastrigin<DIM>(),
                                                       lower,
                                                       upper,
                                                       dX,
                                                       dV,
                                                       dPBestX,
                                                       dPBestVal,
                                                       dGBestX,
                                                       dGBestVal,
                                                       N,
                                                       seed,
                                                       d_states);
  cudaDeviceSynchronize();

  // 3) snapshot pre‐iter
  double hX0[DIM], hV0[DIM], hPB0[DIM], hGB0[DIM], hPV0, hGV0;
  copyDevice(dX, hX0, DIM);
  copyDevice(dV, hV0, DIM);
  copyDevice(dPBestX, hPB0, DIM);
  copyDevice(dGBestX, hGB0, DIM);
  copyDevice(dPBestVal, &hPV0, 1);
  copyDevice(dGBestVal, &hGV0, 1);

  // 4) one iteration with w=c1=c2=0
  pso::iterKernel<util::Rastrigin<DIM>, DIM><<<1, N>>>(util::Rastrigin<DIM>(),
                                                       lower,
                                                       upper,
                                                       0.0,
                                                       0.0,
                                                       0.0, // w, c1, c2
                                                       dX,
                                                       dV,
                                                       dPBestX,
                                                       dPBestVal,
                                                       dGBestX,
                                                       dGBestVal,
                                                       nullptr,
                                                       false,
                                                       N,
                                                       0,
                                                       seed,
                                                       d_states);
  cudaDeviceSynchronize();

  // 5) snapshot post‐iter
  double hX1[DIM], hV1[DIM], hPB1[DIM], hGB1[DIM], hPV1, hGV1;
  copyDevice(dX, hX1, DIM);
  copyDevice(dV, hV1, DIM);
  copyDevice(dPBestX, hPB1, DIM);
  copyDevice(dGBestX, hGB1, DIM);
  copyDevice(dPBestVal, &hPV1, 1);
  copyDevice(dGBestVal, &hGV1, 1);

  // 6) assertions
  for (int d = 0; d < DIM; ++d) {
    REQUIRE(hX1[d] == Approx(hX0[d]).margin(1e-12));
    REQUIRE(hV1[d] == Approx(0.0).margin(1e-12));
    REQUIRE(hPB1[d] == Approx(hPB0[d]).margin(1e-12));
    REQUIRE(hGB1[d] == Approx(hGB0[d]).margin(1e-12));
  }
  REQUIRE(hPV1 == Approx(hPV0).margin(1e-12));
  REQUIRE(hGV1 == Approx(hGV0).margin(1e-12));

  // cleanup
  cudaFree(dX);
  cudaFree(dV);
  cudaFree(dPBestX);
  cudaFree(dPBestVal);
  cudaFree(dGBestX);
  cudaFree(dGBestVal);
  cudaFree(d_states);
}

inline int pointerStatus(double* p)
{
    if (p == MALLOC_ERROR)  return 3;
    if (p == KERNEL_ERROR)  return 4;
    return 0;                               // success
}

TEST_CASE("pso::launch returns 4: KERNEL_ERROR when overflowing the memory")
{
    // exhaust almost all free memory on the device ──────────────
    std::vector<void*> scraps;
    size_t freeB  = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);

    // Keep at least 16 MiB so the runtime itself can breathe
    const size_t CHUNK = (freeB > 32ull << 20) ? (freeB - (16ull << 20)) : freeB / 2;

    while (true) {
        void* p = nullptr;
        if (cudaMalloc(&p, CHUNK) != cudaSuccess) break;
        scraps.push_back(p);
    }

    // now launch PSO with a 'normal' problem size
    using Fn  = util::Rosenbrock<2>;
    constexpr int DIM = 2;
    float ms0 = 0.0f, ms1 = 0.0f;
    curandState* states = nullptr;

    double* ptr = pso::launch<Fn, DIM>(/*PSO_ITER*/ 10,
                                       /*N*/         512,
                                       /*lower*/    -2.0,
                                       /*upper*/     2.0,
                                       ms0, ms1,
                                       42, states,
                                       Fn{});

    REQUIRE(ptr == KERNEL_ERROR);

    // clean up the scrap buffers so the rest of the suite runs ‐─
    for (void* p : scraps) cudaFree(p);
}

TEST_CASE("pso::launch returns 3 (MALLOC_ERROR) when cudaMalloc fails",
          "[pso][malloc-error]")
{
    using Fn  = util::Rosenbrock<2>;
    constexpr int DIM       = 2;
    constexpr int PSO_ITER  = 10;

    // pick an absurdly large N so that at least one cudaMalloc fails
    const int N = std::numeric_limits<int>::max() / DIM;   // ≈1 G x DIM doubles

    float ms_init = 0.0f, ms_pso = 0.0f;
    curandState* states = nullptr;

    double* ptr = pso::launch<Fn, DIM>(PSO_ITER, N, -2.0, 2.0,
                                       ms_init, ms_pso, 42, states, Fn{});

    REQUIRE(pointerStatus(ptr) == 3);
}

TEST_CASE("pso::launch returns 4 (KERNEL_ERROR) when the kernel launch fails",
          "[pso][kernel-error]")
{
    using Fn  = util::Rosenbrock<2>;
    constexpr int DIM       = 2;
    constexpr int PSO_ITER  = 10;

    // This deliberately supplies no RNG state array.
    // the kernel dereferences it, so the first launch triggers an invalid-device-pointer
    // error which `pso::launch` must translate to KERNEL_ERROR.
    curandState* states = nullptr;

    float ms_init = 0.0f, ms_pso = 0.0f;
    const int N = 512;

    double* ptr = pso::launch<Fn, DIM>(PSO_ITER, N, -2.0, 2.0,
                                       ms_init, ms_pso, 99, states, Fn{});

    REQUIRE(pointerStatus(ptr) == 4);
}

