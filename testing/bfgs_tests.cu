#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>

#include "bfgs.cuh"
#include "fun.h"
#include "utils.cuh"

// 1D quadratic function for testing
struct Quadratic {
  __device__ double
  operator()(const std::array<double, 1>& x) const
  {
    return x[0] * x[0];
  }
};

// Kernel to wrap util::line_search in 1D
__global__ void
test_line_search(const double* x,
                 const double* p,
                 const double* g,
                 double f0,
                 double* out)
{
  out[0] = util::line_search<Quadratic, 1>(f0, x, p, g, Quadratic());
}

// Kernel to wrap gradient‐norm
__global__ void
test_grad_norm(const double* g, double* out)
{
  out[0] = util::calculate_gradient_norm<2>(g);
}

// Kernel to wrap identity initialization
__global__ void
test_identity(double* H, int dim)
{
  util::initialize_identity_matrix(H, dim);
}

// Kernel to wrap search direction (H * -g)
__global__ void
test_search_dir(const double* H, const double* g, double* p)
{
  util::compute_search_direction<3>(p, H, g);
}

// Kernel to wrap BFGS update
__global__ void
test_bfgs_update(double* H, const double* dx, const double* dg, double dot)
{
  util::bfgs_update<2>(H, dx, dg, dot);
}

/*
under the Armijo rule with c1 = 0.3, for f(x)=x^2 at x0=1 and search direction
p=–2, you get:

Initial alpha=1
New point x1 = 1 + 1 x (–2) = –1
f(–1) = 1
Armijo RHS = f0 + c1 x alpha x(g x p) = 1 + 0.3 x 1 x (2 x –2) = 1 – 1.2 = –0.2
Check: 1 <= –0.2 ? No, so halve alpha.

Next alpha=0.5
x1 = 1 + 0.5·(–2) = 0
f(0) = 0
Armijo RHS = 1 + 0.3·0.5·(–4) = 1 – 0.6 = 0.4
Check: 0 ≤ 0.4 ? Yes, so we stop with α=0.5.

Thus the correct expected step-size is 0.5.
*/
TEST_CASE("device line_search on x^2 from x=1 gives α=1", "[bfgs][line_search]")
{
  double hX[1] = {1.0}, hP[1] = {-2.0}, hG[1] = {2.0};
  double *dX, *dP, *dG, *dOut;
  cudaMalloc(&dX, sizeof(hX));
  cudaMalloc(&dP, sizeof(hP));
  cudaMalloc(&dG, sizeof(hG));
  cudaMalloc(&dOut, sizeof(double));
  cudaMemcpy(dX, hX, sizeof(hX), cudaMemcpyHostToDevice);
  cudaMemcpy(dP, hP, sizeof(hP), cudaMemcpyHostToDevice);
  cudaMemcpy(dG, hG, sizeof(hG), cudaMemcpyHostToDevice);

  double f0 = 1.0;
  test_line_search<<<1, 1>>>(dX, dP, dG, f0, dOut);
  cudaDeviceSynchronize();

  double alpha;
  cudaMemcpy(&alpha, dOut, sizeof(double), cudaMemcpyDeviceToHost);
  REQUIRE(alpha == Approx(0.5).margin(1e-8));

  cudaFree(dX);
  cudaFree(dP);
  cudaFree(dG);
  cudaFree(dOut);
}

TEST_CASE("device gradient‐norm of [3,4] is 5", "[bfgs][norm]")
{
  double hG[2] = {3.0, 4.0}, out;
  double *dG, *dOut;
  cudaMalloc(&dG, sizeof(hG));
  cudaMalloc(&dOut, sizeof(double));
  cudaMemcpy(dG, hG, sizeof(hG), cudaMemcpyHostToDevice);

  test_grad_norm<<<1, 1>>>(dG, dOut);
  cudaDeviceSynchronize();
  cudaMemcpy(&out, dOut, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(out == Approx(5.0).margin(1e-12));

  cudaFree(dG);
  cudaFree(dOut);
}

TEST_CASE("device identity init sets H=I for 3×3", "[bfgs][identity]")
{
  constexpr int DIM = 3;
  double hH[DIM * DIM];
  double* dH;
  cudaMalloc(&dH, sizeof(hH));

  test_identity<<<1, 1>>>(dH, DIM);
  cudaDeviceSynchronize();
  cudaMemcpy(hH, dH, sizeof(hH), cudaMemcpyDeviceToHost);

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      REQUIRE(hH[i * DIM + j] == Approx(i == j ? 1.0 : 0.0).margin(1e-12));
    }
  }
  cudaFree(dH);
}

TEST_CASE("device compute_search_direction uses H=I -> p=-g for 3D",
          "[bfgs][direction]")
{
  constexpr int DIM = 3;
  double hH[DIM * DIM], hG[DIM] = {1.0, -2.0, 0.5}, hP[DIM];
  double *dH, *dG, *dP;
  // host‐side identity fill
  for (int i = 0; i < DIM; ++i)
    for (int j = 0; j < DIM; ++j)
      hH[i * DIM + j] = (i == j ? 1.0 : 0.0);
  // util::initialize_identity_matrix(hH, DIM);
  cudaMalloc(&dH, sizeof(hH));
  cudaMalloc(&dG, sizeof(hG));
  cudaMalloc(&dP, sizeof(hP));
  cudaMemcpy(dH, hH, sizeof(hH), cudaMemcpyHostToDevice);
  cudaMemcpy(dG, hG, sizeof(hG), cudaMemcpyHostToDevice);

  test_search_dir<<<1, 1>>>(dH, dG, dP);
  cudaDeviceSynchronize();
  cudaMemcpy(hP, dP, sizeof(hP), cudaMemcpyDeviceToHost);

  for (int i = 0; i < DIM; i++) {
    REQUIRE(hP[i] == Approx(-hG[i]).margin(1e-12));
  }
  cudaFree(dH);
  cudaFree(dG);
  cudaFree(dP);
}

TEST_CASE("device bfgs_update updates H correctly for a 2D step",
          "[bfgs][update]")
{
  constexpr int DIM = 2;
  double hH[DIM * DIM];
  for (int i = 0; i < DIM; ++i)
    for (int j = 0; j < DIM; ++j)
      hH[i * DIM + j] = (i == j ? 1.0 : 0.0);
  // util::initialize_identity_matrix(hH,DIM);
  double dx[DIM] = {1.0, 0.0}, dg[DIM] = {2.0, 0.0};
  double *dH, *ddx, *ddg;
  cudaMalloc(&dH, sizeof(hH));
  cudaMalloc(&ddx, sizeof(dx));
  cudaMalloc(&ddg, sizeof(dg));
  cudaMemcpy(dH, hH, sizeof(hH), cudaMemcpyHostToDevice);
  cudaMemcpy(ddx, dx, sizeof(dx), cudaMemcpyHostToDevice);
  cudaMemcpy(ddg, dg, sizeof(dg), cudaMemcpyHostToDevice);

  double dot = 2.0;
  test_bfgs_update<<<1, 1>>>(dH, ddx, ddg, dot);
  cudaDeviceSynchronize();
  cudaMemcpy(hH, dH, sizeof(hH), cudaMemcpyDeviceToHost);

  REQUIRE(hH[0] == Approx(0.5).margin(1e-12));
  REQUIRE(hH[1] == Approx(0.0).margin(1e-12));
  REQUIRE(hH[2] == Approx(0.0).margin(1e-12));
  REQUIRE(hH[3] == Approx(1.0).margin(1e-12));

  cudaFree(dH);
  cudaFree(ddx);
  cudaFree(ddg);
}

TEST_CASE("bfgs::launch converges immediately for util::Rastrigin<2>",
          "[bfgs][optimize]")
{
  constexpr int N = 1, DIM = 2;
  const double lower = -5.0, upper = 5.0;
  const int MAX_ITER = 10, requiredConverged = 1;
  const double tolerance = 1e-6;
  const uint64_t seed = 123;

  // prepare PSO‐init at exact minimizer {0,0}
  double hInit[DIM] = {0.0, 0.0};
  double* dPSOInit = nullptr;
  cudaMalloc(&dPSOInit, N * DIM * sizeof(double));
  cudaMemcpy(dPSOInit, hInit, DIM * sizeof(double), cudaMemcpyHostToDevice);

  // hostResults placeholder
  double* hostResults = new double[N];

  // curand states
  float ms_rand = 0.0f;
  curandState* d_states = bfgs::initialize_states(N, int(seed), ms_rand);

  // correctly typed args
  double* deviceTrajectory = nullptr;
  float ms_opt = 0.0f;
  std::string fun_name = "rastrigin-bfgs-test";

  // invoke bfgs::launch<Function,DIM>
  auto best = bfgs::launch<util::Rastrigin<DIM>, DIM>(
    /*N*/ N,
    /*pso_iter*/ 0,
    /*MAX_ITER*/ MAX_ITER,
    /*upper*/ upper,
    /*lower*/ lower,
    /*pso_results_device*/ dPSOInit,
    /*hostResults*/ hostResults,
    /*deviceTrajectory*/ deviceTrajectory,
    /*requiredConverged*/ requiredConverged,
    /*tolerance*/ tolerance,
    /*save_trajectories*/ false,
    /*ms_opt*/ ms_opt,
    /*fun_name*/ fun_name,
    /*states*/ d_states,
    /*run*/ 0,
    /*f*/ util::Rastrigin<DIM>());

  REQUIRE(best.status == 1);
  REQUIRE(best.iter == 0);
  REQUIRE(best.fval == Approx(0.0).margin(tolerance));
  for (int d = 0; d < DIM; ++d)
    REQUIRE(best.coordinates[d] == Approx(hInit[d]).margin(1e-6));

  // cleanup
  delete[] hostResults;
  cudaFree(dPSOInit);
  cudaFree(d_states);
}

struct Quad {
  template <class T, std::size_t N>
  __device__ T
  operator()(const std::array<T, N>& x) const
  {
    return x[0] * x[0] + x[1] * x[1];
  }
};

TEST_CASE("bfgs::launch converges for Quad<2>", "[bfgs][opt]")
{
  constexpr int N = 1, DIM = 2;
  double hInit[DIM] = {4.5, 4.5};
  double* dInit;
  cudaMalloc(&dInit, sizeof(hInit));
  cudaMemcpy(dInit, hInit, sizeof(hInit), cudaMemcpyHostToDevice);

  double* hostResults = new double[N];
  float ms_rand = 0.0f;
  auto d_states = bfgs::initialize_states(N, /*seed=*/123, ms_rand);

  double* deviceTrajectory = nullptr;
  float ms_opt = 0;
  std::string fun_name = "quad-test";

  auto best = bfgs::launch<Quad, DIM>(
    /*N*/ N,
    /*pso_iter*/ 0,
    /*MAX_ITER*/ 100,
    /*upper*/ 10.0,
    /*lower*/ -10.0,
    /*pso_results_device*/ dInit,
    /*hostResults*/ hostResults,
    /*deviceTrajectory*/ deviceTrajectory,
    /*requiredConverged*/ 1,
    /*tolerance*/ 1e-6,
    /*save_trajectories*/ false,
    /*ms_opt*/ ms_opt,
    /*fun_name*/ fun_name,
    /*states*/ d_states,
    /*run*/ 0,
    /*f*/ Quad{});

  // quad minimum is at (0,0), fval=0
  REQUIRE(best.status == 1);
  REQUIRE(best.fval == Approx(0.0).margin(1e-6));
  REQUIRE(best.coordinates[0] == Approx(0.0).margin(1e-6));
  REQUIRE(best.coordinates[1] == Approx(0.0).margin(1e-6));

  delete[] hostResults;
  cudaFree(dInit);
  cudaFree(d_states);
}

TEST_CASE("bfgs::launch converges for util::Rosenbrock<2>", "[bfgs][optimize]")
{
  constexpr int N = 1, DIM = 2;
  const double lower = -5.0, upper = 5.0;
  const int MAX_ITER = 1000, requiredConverged = 1;
  const double tolerance = 1e-17;
  const uint64_t seed = 123;

  // prepare PSO‐init at exact minimizer {0,0}
  double hInit[DIM] = {4.5, 4.5};
  double* dPSOInit = nullptr;
  cudaMalloc(&dPSOInit, N * DIM * sizeof(double));
  cudaMemcpy(dPSOInit, hInit, DIM * sizeof(double), cudaMemcpyHostToDevice);

  // hostResults placeholder
  double* hostResults = new double[N];

  // curand states
  float ms_rand = 0.0f;
  curandState* d_states = bfgs::initialize_states(N, int(seed), ms_rand);

  // correctly typed args
  double* deviceTrajectory = nullptr;
  float ms_opt = 0.0f;
  std::string fun_name = "rosenbrock-bfgs-test";

  // invoke bfgs::launch<Function,DIM>
  auto best = bfgs::launch<util::Rosenbrock<DIM>, DIM>(
    /*N*/ N,
    /*pso_iter*/ 0,
    /*MAX_ITER*/ MAX_ITER,
    /*upper*/ upper,
    /*lower*/ lower,
    /*pso_results_device*/ dPSOInit,
    /*hostResults*/ hostResults,
    /*deviceTrajectory*/ deviceTrajectory,
    /*requiredConverged*/ requiredConverged,
    /*tolerance*/ tolerance,
    /*save_trajectories*/ false,
    /*ms_opt*/ ms_opt,
    /*fun_name*/ fun_name,
    /*states*/ d_states,
    /*run*/ 0,
    /*f*/ util::Rosenbrock<DIM>());

  REQUIRE(best.status == 1);
  REQUIRE(best.iter < MAX_ITER);
  REQUIRE(best.fval == Approx(0.0).margin(1e-4));
  for (int d = 0; d < DIM; ++d)
    REQUIRE(best.coordinates[d] == Approx(1.0).margin(1e-6));

  // cleanup
  delete[] hostResults;
  cudaFree(dPSOInit);
  cudaFree(d_states);
}

// dim we’ll test with
constexpr int DIM = 2;

struct GoodObjective {
  // templated call operator: accepts an array of DualNumber<DIM>
  template <class T>
  __device__ T
  operator()(const std::array<T, DIM>& x) const
  {
    // just sum coordinates
    T sum = T(0);
    for (int i = 0; i < DIM; ++i)
      sum = sum + x[i];
    return sum;
  }
};

struct BadObjective {
  // takes array of doubles, not DualNumber!
  __device__ double
  operator()(const std::array<double, DIM>& x) const
  {
    return x[0] * x[0] + x[1] * x[1];
  }
};

TEST_CASE("good/bad objective test", "[bfgs][objective]")
{
  curandState* states = nullptr;
  // dummy device pointers:
  double* d_pso = nullptr;
  double* d_results = nullptr;
  const int N = 1, MAX_ITER = 1, requiredConverged = 1;
  const double lower = 0.0, upper = 1.0, tolerance = 1e-6;
  Result<DIM>* d_out = nullptr;

  // This **should compile** without error:
  bfgs::optimizeKernel<GoodObjective, DIM, 128><<<1, 128>>>(GoodObjective(),
                                                            lower,
                                                            upper,
                                                            d_pso,
                                                            d_results,
                                                            nullptr,
                                                            N,
                                                            MAX_ITER,
                                                            requiredConverged,
                                                            tolerance,
                                                            d_out,
                                                            states);

  // This **must fail** to compile, triggering static_assert:
#if (0)
  bfgs::optimizeKernel<BadObjective, DIM, 128><<<1, 128>>>(BadObjective(),
                                                           lower,
                                                           upper,
                                                           d_pso,
                                                           d_results,
                                                           nullptr,
                                                           N,
                                                           MAX_ITER,
                                                           requiredConverged,
                                                           tolerance,
                                                           d_out,
                                                           states);
  dual::calculateGradientUsingAD(BadObjective, , ga);
#endif
}
