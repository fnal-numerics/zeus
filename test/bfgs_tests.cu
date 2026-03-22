#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>

#include "bfgs_sequential.cuh"
#include "fun.h"
#include "utils.cuh"

using namespace zeus;

namespace {

  void
  require_cuda_success(cudaError_t status, const char* action)
  {
    INFO(action << ": " << cudaGetErrorString(status));
    REQUIRE(status == cudaSuccess);
  }

  void
  require_cuda_launch_success(const char* kernel_name)
  {
    require_cuda_success(cudaGetLastError(), kernel_name);
  }

  template <typename T>
  struct DeviceBuffer {
    T* ptr = nullptr;

    ~DeviceBuffer()
    {
      if (ptr != nullptr)
        cudaFree(ptr);
    }

    T**
    out()
    {
      return &ptr;
    }
    T*
    get() const
    {
      return ptr;
    }
  };

} // namespace

// 1D quadratic function for testing
struct Quadratic {
  static constexpr std::size_t arity = 1;
  __device__ double
  operator()(const std::array<double, 1>& x) const
  {
    return x[0] * x[0];
  }
};

// Kernel to wrap util::line_search in 1D
__global__ void
testLineSearch(const double* x,
               const double* p,
               const double* g,
               double f0,
               double* out)
{
  out[0] = util::lineSearch<Quadratic, 1>(f0, x, p, g, Quadratic());
}

// Kernel to wrap gradient‐norm
__global__ void
testGradNorm(const double* g, double* out)
{
  out[0] = util::calculateGradientNorm<2>(g);
}

// Kernel to wrap identity initialization
__global__ void
testIdentity(double* H, int dim)
{
  util::initializeIdentityMatrix(H, dim);
}

// Kernel to wrap search direction (H * -g)
__global__ void
testSearchDir(const double* H, const double* g, double* p)
{
  util::computeSearchDirection<3>(p, H, g);
}

// Kernel to wrap BFGS update
__global__ void
testBfgsUpdate(double* H, const double* dx, const double* dg, double dot)
{
  util::bfgsUpdate<2>(H, dx, dg, dot);
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
TEST_CASE("device lineSearch on x^2 from x=1 gives α=1",
          "[bfgs][line_search][gpu]")
{
  double hX[1] = {1.0}, hP[1] = {-2.0}, hG[1] = {2.0};
  DeviceBuffer<double> dX, dP, dG, dOut;
  require_cuda_success(cudaMalloc(dX.out(), sizeof(hX)), "cudaMalloc dX");
  require_cuda_success(cudaMalloc(dP.out(), sizeof(hP)), "cudaMalloc dP");
  require_cuda_success(cudaMalloc(dG.out(), sizeof(hG)), "cudaMalloc dG");
  require_cuda_success(cudaMalloc(dOut.out(), sizeof(double)),
                       "cudaMalloc dOut");
  require_cuda_success(
    cudaMemcpy(dX.get(), hX, sizeof(hX), cudaMemcpyHostToDevice),
    "cudaMemcpy hX -> dX");
  require_cuda_success(
    cudaMemcpy(dP.get(), hP, sizeof(hP), cudaMemcpyHostToDevice),
    "cudaMemcpy hP -> dP");
  require_cuda_success(
    cudaMemcpy(dG.get(), hG, sizeof(hG), cudaMemcpyHostToDevice),
    "cudaMemcpy hG -> dG");

  double f0 = 1.0;
  testLineSearch<<<1, 1>>>(dX.get(), dP.get(), dG.get(), f0, dOut.get());
  require_cuda_launch_success("testLineSearch launch");
  require_cuda_success(cudaDeviceSynchronize(), "testLineSearch sync");

  double alpha;
  require_cuda_success(
    cudaMemcpy(&alpha, dOut.get(), sizeof(double), cudaMemcpyDeviceToHost),
    "cudaMemcpy dOut -> alpha");
  REQUIRE(alpha == Approx(0.5).margin(1e-8));
}

TEST_CASE("device gradient‐norm of [3,4] is 5", "[bfgs][norm][gpu]")
{
  double hG[2] = {3.0, 4.0}, out;
  DeviceBuffer<double> dG, dOut;
  require_cuda_success(cudaMalloc(dG.out(), sizeof(hG)), "cudaMalloc dG");
  require_cuda_success(cudaMalloc(dOut.out(), sizeof(double)),
                       "cudaMalloc dOut");
  require_cuda_success(
    cudaMemcpy(dG.get(), hG, sizeof(hG), cudaMemcpyHostToDevice),
    "cudaMemcpy hG -> dG");

  testGradNorm<<<1, 1>>>(dG.get(), dOut.get());
  require_cuda_launch_success("testGradNorm launch");
  require_cuda_success(cudaDeviceSynchronize(), "testGradNorm sync");
  require_cuda_success(
    cudaMemcpy(&out, dOut.get(), sizeof(double), cudaMemcpyDeviceToHost),
    "cudaMemcpy dOut -> out");

  REQUIRE(out == Approx(5.0).margin(1e-12));
}

TEST_CASE("device identity init sets H=I for 3×3", "[bfgs][identity][gpu]")
{
  constexpr int DIM = 3;
  double hH[DIM * DIM];
  DeviceBuffer<double> dH;
  require_cuda_success(cudaMalloc(dH.out(), sizeof(hH)), "cudaMalloc dH");

  testIdentity<<<1, 1>>>(dH.get(), DIM);
  require_cuda_launch_success("testIdentity launch");
  require_cuda_success(cudaDeviceSynchronize(), "testIdentity sync");
  require_cuda_success(
    cudaMemcpy(hH, dH.get(), sizeof(hH), cudaMemcpyDeviceToHost),
    "cudaMemcpy dH -> hH");

  for (int i = 0; i < DIM; i++) {
    for (int j = 0; j < DIM; j++) {
      REQUIRE(hH[i * DIM + j] == Approx(i == j ? 1.0 : 0.0).margin(1e-12));
    }
  }
}

TEST_CASE("device computeSearchDirection uses H=I -> p=-g for 3D",
          "[bfgs][direction][gpu]")
{
  constexpr int DIM = 3;
  double hH[DIM * DIM], hG[DIM] = {1.0, -2.0, 0.5}, hP[DIM];
  DeviceBuffer<double> dH, dG, dP;
  // host‐side identity fill
  for (int i = 0; i < DIM; ++i)
    for (int j = 0; j < DIM; ++j)
      hH[i * DIM + j] = (i == j ? 1.0 : 0.0);
  // util::initialize_identity_matrix(hH, DIM);
  require_cuda_success(cudaMalloc(dH.out(), sizeof(hH)), "cudaMalloc dH");
  require_cuda_success(cudaMalloc(dG.out(), sizeof(hG)), "cudaMalloc dG");
  require_cuda_success(cudaMalloc(dP.out(), sizeof(hP)), "cudaMalloc dP");
  require_cuda_success(
    cudaMemcpy(dH.get(), hH, sizeof(hH), cudaMemcpyHostToDevice),
    "cudaMemcpy hH -> dH");
  require_cuda_success(
    cudaMemcpy(dG.get(), hG, sizeof(hG), cudaMemcpyHostToDevice),
    "cudaMemcpy hG -> dG");

  testSearchDir<<<1, 1>>>(dH.get(), dG.get(), dP.get());
  require_cuda_launch_success("testSearchDir launch");
  require_cuda_success(cudaDeviceSynchronize(), "testSearchDir sync");
  require_cuda_success(
    cudaMemcpy(hP, dP.get(), sizeof(hP), cudaMemcpyDeviceToHost),
    "cudaMemcpy dP -> hP");

  for (int i = 0; i < DIM; i++) {
    REQUIRE(hP[i] == Approx(-hG[i]).margin(1e-12));
  }
}

TEST_CASE("device bfgsUpdate updates H correctly for a 2D step",
          "[bfgs][update][gpu]")
{
  constexpr int DIM = 2;
  double hH[DIM * DIM];
  for (int i = 0; i < DIM; ++i)
    for (int j = 0; j < DIM; ++j)
      hH[i * DIM + j] = (i == j ? 1.0 : 0.0);
  // util::initialize_identity_matrix(hH,DIM);
  double dx[DIM] = {1.0, 0.0}, dg[DIM] = {2.0, 0.0};
  DeviceBuffer<double> dH, ddx, ddg;
  require_cuda_success(cudaMalloc(dH.out(), sizeof(hH)), "cudaMalloc dH");
  require_cuda_success(cudaMalloc(ddx.out(), sizeof(dx)), "cudaMalloc ddx");
  require_cuda_success(cudaMalloc(ddg.out(), sizeof(dg)), "cudaMalloc ddg");
  require_cuda_success(
    cudaMemcpy(dH.get(), hH, sizeof(hH), cudaMemcpyHostToDevice),
    "cudaMemcpy hH -> dH");
  require_cuda_success(
    cudaMemcpy(ddx.get(), dx, sizeof(dx), cudaMemcpyHostToDevice),
    "cudaMemcpy dx -> ddx");
  require_cuda_success(
    cudaMemcpy(ddg.get(), dg, sizeof(dg), cudaMemcpyHostToDevice),
    "cudaMemcpy dg -> ddg");

  double dot = 2.0;
  testBfgsUpdate<<<1, 1>>>(dH.get(), ddx.get(), ddg.get(), dot);
  require_cuda_launch_success("testBfgsUpdate launch");
  require_cuda_success(cudaDeviceSynchronize(), "testBfgsUpdate sync");
  require_cuda_success(
    cudaMemcpy(hH, dH.get(), sizeof(hH), cudaMemcpyDeviceToHost),
    "cudaMemcpy dH -> hH");

  REQUIRE(hH[0] == Approx(0.5).margin(1e-12));
  REQUIRE(hH[1] == Approx(0.0).margin(1e-12));
  REQUIRE(hH[2] == Approx(0.0).margin(1e-12));
  REQUIRE(hH[3] == Approx(1.0).margin(1e-12));
}

TEST_CASE("bfgs::launch converges immediately for util::Rastrigin<2>",
          "[bfgs][optimize][gpu]")
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
  curandStateXORWOW_t* d_states =
    bfgs::initializeStates<curandStateXORWOW_t>(N, int(seed), ms_rand);

  // correctly typed args
  double* deviceTrajectoryCoords = nullptr;
  double* deviceTrajectoryFval = nullptr;
  double* deviceTrajectoryGrad = nullptr;
  float ms_opt = 0.0f;
  std::string fun_name = "rastrigin-bfgs-test";

  // invoke bfgs::launch<Function,DIM,StateType>
  auto best =
    bfgs::sequential::launch<util::Rastrigin<DIM>, DIM, curandStateXORWOW_t>(
      /*N*/ N,
      /*pso_iter*/ 0,
      /*MAX_ITER*/ MAX_ITER,
      /*upper*/ upper,
      /*lower*/ lower,
      /*pso_results_device*/ dPSOInit,
      /*deviceTrajectoryCoords*/ deviceTrajectoryCoords,
      /*deviceTrajectoryFval*/ deviceTrajectoryFval,
      /*deviceTrajectoryGrad*/ deviceTrajectoryGrad,
      /*deviceStatus*/ nullptr,
      /*requiredConverged*/ requiredConverged,
      /*tolerance*/ tolerance,
      /*nzerosteps*/ 0,
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

TEST_CASE("bfgs::launch converges for Quad<2>", "[bfgs][opt][gpu]")
{
  constexpr int N = 1, DIM = 2;
  double hInit[DIM] = {4.5, 4.5};
  double* dInit;
  cudaMalloc(&dInit, sizeof(hInit));
  cudaMemcpy(dInit, hInit, sizeof(hInit), cudaMemcpyHostToDevice);

  double* hostResults = new double[N];
  float ms_rand = 0.0f;
  auto d_states =
    bfgs::initializeStates<curandStateXORWOW_t>(N, /*seed=*/123, ms_rand);

  double* deviceTrajectoryCoords = nullptr;
  double* deviceTrajectoryFval = nullptr;
  double* deviceTrajectoryGrad = nullptr;
  float ms_opt = 0;
  std::string fun_name = "quad-test";

  auto best = bfgs::sequential::launch<Quad, DIM, curandStateXORWOW_t>(
    /*N*/ N,
    /*pso_iter*/ 0,
    /*MAX_ITER*/ 100,
    /*upper*/ 10.0,
    /*lower*/ -10.0,
    /*pso_results_device*/ dInit,
    /*deviceTrajectoryCoords*/ deviceTrajectoryCoords,
    /*deviceTrajectoryFval*/ deviceTrajectoryFval,
    /*deviceTrajectoryGrad*/ deviceTrajectoryGrad,
    /*deviceStatus*/ nullptr,
    /*requiredConverged*/ 1,
    /*tolerance*/ 1e-6,
    /*nzerosteps*/ 0,
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

TEST_CASE("bfgs::launch converges for util::Rosenbrock<2>",
          "[bfgs][optimize][gpu]")
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
  curandState* d_states = bfgs::initializeStates(N, int(seed), ms_rand);

  // correctly typed args
  double* deviceTrajectoryCoords = nullptr;
  double* deviceTrajectoryFval = nullptr;
  double* deviceTrajectoryGrad = nullptr;
  float ms_opt = 0.0f;
  std::string fun_name = "rosenbrock-bfgs-test";

  // invoke bfgs::launch<Function,DIM,StateType>
  auto best =
    bfgs::sequential::launch<util::Rosenbrock<DIM>, DIM, curandStateXORWOW_t>(
      /*N*/ N,
      /*pso_iter*/ 0,
      /*MAX_ITER*/ MAX_ITER,
      /*upper*/ upper,
      /*lower*/ lower,
      /*pso_results_device*/ dPSOInit,
      /*deviceTrajectoryCoords*/ deviceTrajectoryCoords,
      /*deviceTrajectoryFval*/ deviceTrajectoryFval,
      /*deviceTrajectoryGrad*/ deviceTrajectoryGrad,
      /*deviceStatus*/ nullptr,
      /*requiredConverged*/ requiredConverged,
      /*tolerance*/ tolerance,
      /*nzerosteps*/ 0,
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
  static constexpr std::size_t arity = DIM;
  // templated call operator: accepts an array of DualNumber<DIM>
  template <class T>
  __device__ T
  operator()(const std::array<T, DIM>& x) const
  {
    // just sum coordinates
    T sum = T(0);
    for (int i = 0; i < DIM; ++i)
      sum = sum + x[i] * x[i];
    return sum;
  }
};

struct BadObjective {
  static constexpr std::size_t arity = DIM;
  // takes array of doubles, not DualNumber!
  __device__ double
  operator()(const std::array<double, DIM>& x) const
  {
    return x[0] * x[0] + x[1] * x[1];
  }
};

TEST_CASE("good/bad objective test", "[bfgs][objective][gpu]")
{
  const int N = 1, MAX_ITER = 1, requiredConverged = 1;
  const double lower = 0.0, upper = 1.0, tolerance = 1e-6;

  // Allocate required device pointers
  float ms_rand = 0.0f;
  curandStateXORWOW_t* states =
    bfgs::initializeStates<curandStateXORWOW_t>(N, 42, ms_rand);

  double* d_pso;
  cudaMalloc(&d_pso, N * DIM * sizeof(double));

  double* d_results;
  cudaMalloc(&d_results, N * sizeof(double));

  zeus::Result<DIM>* d_out;
  cudaMalloc(&d_out, N * sizeof(zeus::Result<DIM>));

  // Allocate context
  util::BFGSContext* d_ctx;
  cudaMalloc(&d_ctx, sizeof(util::BFGSContext));
  util::BFGSContext h_ctx = {0, 0};
  cudaMemcpy(d_ctx, &h_ctx, sizeof(util::BFGSContext), cudaMemcpyHostToDevice);

  // This **should compile** without error:
  bfgs::sequential::
    optimize<GoodObjective, DIM, 128, false, curandStateXORWOW_t>
    <<<1, 128>>>(GoodObjective(),
                 lower,
                 upper,
                 d_pso,
                 util::NonNull{d_results},
                 nullptr,
                 nullptr,
                 nullptr,
                 nullptr,
                 N,
                 MAX_ITER,
                 requiredConverged,
                 tolerance,
                 /*nzerosteps=*/0,
                 util::NonNull{d_out},
                 util::NonNull{states},
                 util::NonNull{d_ctx});

  cudaFree(d_ctx);
  cudaFree(d_out);
  cudaFree(d_results);
  cudaFree(d_pso);
  cudaFree(states);

  // This **must fail** to compile, triggering static_assert:
#if (0)
  bfgs::sequential::optimize<BadObjective, DIM, 128, false>
    <<<1, 128>>>(BadObjective(),
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

TEST_CASE("Trajectory saving writes to buffer", "[bfgs][trajectory][gpu]")
{
  // Use a large MAX_ITER so BFGS converges well before the buffer fills.
  // We then verify: slots up to convergence have valid data; slots after have
  // NaN.
  const int N = 1, MAX_ITER = 50, requiredConverged = 1;
  const double lower = -5.0, upper = 5.0, tolerance = 1e-6;

  float ms_rand = 0.0f;
  curandStateXORWOW_t* states =
    bfgs::initializeStates<curandStateXORWOW_t>(N, 42, ms_rand);

  // Start from a known point in the interior; GoodObjective = sum(xi^2),
  // minimum at origin. BFGS will converge in well under 50 iterations.
  double h_pso[DIM] = {1.0, 1.0};
  double* d_pso;
  cudaMalloc(&d_pso, N * DIM * sizeof(double));
  cudaMemcpy(d_pso, h_pso, N * DIM * sizeof(double), cudaMemcpyHostToDevice);

  double* d_results;
  cudaMalloc(&d_results, N * sizeof(double));

  const size_t traj_coords = size_t(N) * DIM * MAX_ITER;
  const size_t traj_scalars = size_t(N) * MAX_ITER;

  double* d_trajectory_coords;
  cudaMalloc(&d_trajectory_coords, traj_coords * sizeof(double));
  util::fillWithNaN(d_trajectory_coords, traj_coords);

  double* d_trajectory_fval;
  cudaMalloc(&d_trajectory_fval, traj_scalars * sizeof(double));
  util::fillWithNaN(d_trajectory_fval, traj_scalars);

  double* d_trajectory_grad;
  cudaMalloc(&d_trajectory_grad, traj_scalars * sizeof(double));
  util::fillWithNaN(d_trajectory_grad, traj_scalars);

  // Allocate and zero-initialize the status buffer (will be overwritten by
  // kernel).
  int8_t* d_status;
  cudaMalloc(&d_status, traj_scalars * sizeof(int8_t));
  cudaMemset(d_status, -1, traj_scalars * sizeof(int8_t));

  zeus::Result<DIM>* d_out;
  cudaMalloc(&d_out, N * sizeof(zeus::Result<DIM>));

  util::BFGSContext* d_ctx;
  cudaMalloc(&d_ctx, sizeof(util::BFGSContext));
  util::BFGSContext h_ctx = {0, 0};
  cudaMemcpy(d_ctx, &h_ctx, sizeof(util::BFGSContext), cudaMemcpyHostToDevice);

  bfgs::sequential::optimize<GoodObjective, DIM, 128, true, curandStateXORWOW_t>
    <<<1, 128>>>(GoodObjective(),
                 lower,
                 upper,
                 d_pso,
                 util::NonNull{d_results},
                 d_trajectory_coords,
                 d_trajectory_fval,
                 d_trajectory_grad,
                 d_status,
                 N,
                 MAX_ITER,
                 requiredConverged,
                 tolerance,
                 /*nzerosteps=*/0,
                 util::NonNull{d_out},
                 util::NonNull{states},
                 util::NonNull{d_ctx});

  cudaDeviceSynchronize();

  // Read back trajectory and status buffers.
  std::vector<double> h_traj_fval(traj_scalars);
  cudaMemcpy(h_traj_fval.data(),
             d_trajectory_fval,
             traj_scalars * sizeof(double),
             cudaMemcpyDeviceToHost);

  std::vector<int8_t> h_status(traj_scalars);
  cudaMemcpy(h_status.data(),
             d_status,
             traj_scalars * sizeof(int8_t),
             cudaMemcpyDeviceToHost);

  // Iter 0 fval must be non-NaN.
  double fval0 = h_traj_fval[0 * N + 0];
  REQUIRE(!std::isnan(fval0));
  REQUIRE(fval0 > 0.0); // starting at (1,1), f=2

  // Find the convergence iteration (first iter with status == 1).
  int conv_iter = -1;
  for (int it = 0; it < MAX_ITER; ++it) {
    if (h_status[it * N + 0] == 1) {
      conv_iter = it;
      break;
    }
  }
  // GoodObjective must converge within MAX_ITER steps.
  REQUIRE(conv_iter >= 0);
  REQUIRE(conv_iter < MAX_ITER - 1); // must converge before the last slot

  // All trajectory slots AFTER convergence must be NaN (never written).
  for (int it = conv_iter + 1; it < MAX_ITER; ++it) {
    double v = h_traj_fval[it * N + 0];
    REQUIRE(std::isnan(v));
  }

  cudaFree(d_ctx);
  cudaFree(d_out);
  cudaFree(d_trajectory_coords);
  cudaFree(d_trajectory_fval);
  cudaFree(d_trajectory_grad);
  cudaFree(d_status);
  cudaFree(d_results);
  cudaFree(d_pso);
  cudaFree(states);
}
