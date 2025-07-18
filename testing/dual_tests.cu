#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include "duals.cuh"
#include <cuda_runtime.h>
#include <cmath>

using Catch::Approx;
using dual::DualNumber;

// Helpers
#define ALLOC_COPY(DEV, HOST, SZ)                                              \
  cudaMalloc(&DEV, SZ);                                                        \
  cudaMemcpy(DEV, HOST, SZ, cudaMemcpyHostToDevice);

#define FREE4(a, b, c, d)                                                      \
  do {                                                                         \
    cudaFree(a);                                                               \
    cudaFree(b);                                                               \
    cudaFree(c);                                                               \
    cudaFree(d);                                                               \
  } while (0)

#define FREE6(a, b, c, d, e, f)                                                \
  do {                                                                         \
    cudaFree(a);                                                               \
    cudaFree(b);                                                               \
    cudaFree(c);                                                               \
    cudaFree(d);                                                               \
    cudaFree(e);                                                               \
    cudaFree(f);                                                               \
  } while (0)

// --- KERNELS ---------------------------------------------------------------

// Kernel for addition: tests (1 + 2e) + (3 + 4e) = (4 + 6e)
__global__ void
test_add(const double* a, const double* b, double* r, double* d)
{
  DualNumber A{a[0], a[1]}; // load A = 1.0 + 2.0e
  DualNumber B{b[0], b[1]}; // load B = 3.0 + 4.0e
  DualNumber R = A + B;     // compute R = (1+3) + (2+4)e = 4.0 + 6.0e
  r[0] = R.real;            // store real part = 4.0
  d[0] = R.dual;            // store dual part = 6.0
}

// Kernel for subtraction: tests (5 + 2.5e) – (1.5 + 0.5e) = (3.5 + 2.0e)
__global__ void
test_sub(const double* a, const double* b, double* r, double* d)
{
  DualNumber A{a[0], a[1]}; // load A = 5.0 + 2.5e
  DualNumber B{b[0], b[1]}; // load B = 1.5 + 0.5e
  DualNumber R = A - B;     // compute R = (5−1.5) + (2.5−0.5)e = 3.5 + 2.0e
  r[0] = R.real;            // store real part = 3.5
  d[0] = R.dual;            // store dual part = 2.0
}

// Kernel for multiplication: tests (2 + 3e) * (4 + 5e) = (8 + 22e)
__global__ void
test_mul(const double* a, const double* b, double* r, double* d)
{
  DualNumber A{a[0], a[1]}; // load A = 2.0 + 3.0e
  DualNumber B{b[0], b[1]}; // load B = 4.0 + 5.0e
  DualNumber R = A * B; // compute R.real = 2*4 = 8.0; R.dual = 3*4 + 2*5 = 22.0
  r[0] = R.real;        // store real part = 8.0
  d[0] = R.dual;        // store dual part = 22.0
}

// Kernel for division: tests (6 + 2e) / (3 + 1e) = (2 + 0e)
__global__ void
test_div(const double* a, const double* b, double* r, double* d)
{
  DualNumber A{a[0], a[1]}; // load A = 6.0 + 2.0e
  DualNumber B{b[0], b[1]}; // load B = 3.0 + 1.0e
  DualNumber R =
    A / B;       // compute R.real = 6/3 = 2.0; R.dual = (2*3 - 6*1)/9 = 0.0
  r[0] = R.real; // store real part = 2.0
  d[0] = R.dual; // store dual part = 0.0
}

// Kernel for sin: tests sin(π/6 + 1e) = (0.5 + 0.866025e)
__global__ void
test_sin(const double* x, const double* dx, double* r, double* d)
{
  DualNumber D{x[0], dx[0]}; // load D = π/6 ~ 0.523599 + 1.0e
  DualNumber R =
    dual::sin(D); // R.real = sin(π/6) = 0.5; R.dual = 1.0 * cos(π/6) = 0.866025
  r[0] = R.real;  // store real part = 0.5
  d[0] = R.dual;  // store dual part ≈ 0.866025
}

// Kernel for cos: tests cos(π/3 + 2e) = (0.5 − 1.73205e)
__global__ void
test_cos(const double* x, const double* dx, double* r, double* d)
{
  DualNumber D{x[0], dx[0]}; // load D = π/3 ≈ 1.047198 + 2.0e
  DualNumber R = dual::cos(
    D);          // R.real = cos(π/3) = 0.5; R.dual = −2.0 * sin(π/3) = −1.73205
  r[0] = R.real; // store real part = 0.5
  d[0] = R.dual; // store dual part ≈ −1.73205
}

// Kernel for exp: tests exp(1 + 3e) = (e + 3ee)
__global__ void
test_exp(const double* x, const double* dx, double* r, double* d)
{
  DualNumber D{x[0], dx[0]}; // load D = 1.0 + 3.0e
  DualNumber R =
    dual::exp(D); // R.real = e ~ 2.71828; R.dual = 3.0 * e ~ 8.15484
  r[0] = R.real;  // store real part ~ 2.71828
  d[0] = R.dual;  // store dual part ~ 8.15484
}

// Kernel for sqrt: tests sqrt(4 + 1.5e) = (2 + 0.375e)
__global__ void
test_sqrt(const double* x, const double* dx, double* r, double* d)
{
  DualNumber D{x[0], dx[0]};    // load D = 4.0 + 1.5e
  DualNumber R = dual::sqrt(D); // R.real = 2.0; R.dual = 1.5 / (2*2) = 0.375
  r[0] = R.real;                // store real part = 2.0
  d[0] = R.dual;                // store dual part = 0.375
}

// Kernel for pow³: tests (2 + 1ε)^3 = (8 + 12ε)
__global__ void
test_pow3(const double* x, const double* dx, double* r, double* d)
{
  DualNumber D{x[0], dx[0]};        // load D = 2.0 + 1.0e
  DualNumber R = dual::pow(D, 3.0); // R.real = 8.0; R.dual = 3 * 2^2 * 1 = 12.0
  r[0] = R.real;                    // store real part = 8.0
  d[0] = R.dual;                    // store dual part = 12.0
}

// Kernel for atan2: tests atan2(1+1e, 1+0e) = (π/4 + 0.5e)
__global__ void
test_atan2(const double* y,
           const double* dy,
           const double* x,
           const double* dx,
           double* r,
           double* d)
{
  DualNumber Y{y[0], dy[0]}; // load Y = 1.0 + 1.0e
  DualNumber X{x[0], dx[0]}; // load X = 1.0 + 0.0e
  DualNumber R =
    dual::atan2(Y, X); // R.real = π/4 ~ 0.785398; R.dual = (1*1 - 1*0)/2 = 0.5
  r[0] = R.real;       // store real part ~ 0.785398
  d[0] = R.dual;       // store dual part = 0.5
}

// Kernel for unary minus: tests  –(1.5 + 2.5e) = –1.5 + –2.5e
__global__ void
test_neg(const double* a, double* r, double* d)
{
  DualNumber A{a[0], a[1]};  // load A = 1.5 + 2.5e
  DualNumber R = -A;         // compute R = –1.5 + –2.5e
  r[0] = R.real;             // store real part = –1.5
  d[0] = R.dual;             // store dual part = –2.5
}

// --- TEST CASES ----------------------------------------------------------

TEST_CASE("operator-(): -(1.5+2.5e) = -1.5-2.5e", "[dual][neg]")
{
  double ha[2] = {1.5, 2.5}, outR, outD;
  double *dA, *dR, *dD;

  // allocate & copy host → device
  ALLOC_COPY(dA, ha, sizeof(ha));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  // launch and sync
  test_neg<<<1,1>>>(dA, dR, dD);
  cudaDeviceSynchronize();

  // copy results back
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  // check
  REQUIRE(outR == Approx(-1.5).margin(1e-12));
  REQUIRE(outD == Approx(-2.5).margin(1e-12));

  // cleanup
  FREE4(dA, dA, dR, dD);  // note: FREE4(a,b,c,d) frees four pointers
}

TEST_CASE("operator+(): (1+2e)+(3+4e)=4+6e", "[dual][add]")
{
  double ha[2] = {1, 2}, hb[2] = {3, 4}, outR, outD;
  double *dA, *dB, *dR, *dD;
  ALLOC_COPY(dA, ha, sizeof(ha));
  ALLOC_COPY(dB, hb, sizeof(hb));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_add<<<1, 1>>>(dA, dB, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(4.0).margin(1e-12));
  REQUIRE(outD == Approx(6.0).margin(1e-12));

  FREE4(dA, dB, dR, dD);
}

TEST_CASE("operator-(): (5+2.5e)-(1.5+0.5e)=3.5+2e", "[dual][sub]")
{
  double ha[2] = {5.0, 2.5}, hb[2] = {1.5, 0.5}, outR, outD;
  double *dA, *dB, *dR, *dD;
  ALLOC_COPY(dA, ha, sizeof(ha));
  ALLOC_COPY(dB, hb, sizeof(hb));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_sub<<<1, 1>>>(dA, dB, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(3.5).margin(1e-12));
  REQUIRE(outD == Approx(2.0).margin(1e-12));

  FREE4(dA, dB, dR, dD);
}

TEST_CASE("operator*(): (2+3e)*(4+5e)=8+22e", "[dual][mul]")
{
  double ha[2] = {2, 3}, hb[2] = {4, 5}, outR, outD;
  double *dA, *dB, *dR, *dD;
  ALLOC_COPY(dA, ha, sizeof(ha));
  ALLOC_COPY(dB, hb, sizeof(hb));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_mul<<<1, 1>>>(dA, dB, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(8.0).margin(1e-12));
  REQUIRE(outD == Approx(22.0).margin(1e-12));

  FREE4(dA, dB, dR, dD);
}

TEST_CASE("operator/(): (6+2e)/(3+1e)=2+0e", "[dual][div]")
{
  double ha[2] = {6, 2}, hb[2] = {3, 1}, outR, outD;
  double *dA, *dB, *dR, *dD;
  ALLOC_COPY(dA, ha, sizeof(ha));
  ALLOC_COPY(dB, hb, sizeof(hb));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_div<<<1, 1>>>(dA, dB, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(2.0).margin(1e-12));
  REQUIRE(outD == Approx(0.0).margin(1e-12));

  FREE4(dA, dB, dR, dD);
}

TEST_CASE("operator sin(): sin(π/6)+1e -> 0.5+0.866e", "[dual][sin]")
{
  double x = M_PI / 6, dx = 1.0, outR, outD;
  double *dX, *dDX, *dR, *dD;
  ALLOC_COPY(dX, &x, sizeof(x));
  ALLOC_COPY(dDX, &dx, sizeof(dx));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_sin<<<1, 1>>>(dX, dDX, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(0.5).margin(1e-7));
  REQUIRE(outD == Approx(std::cos(x)).margin(1e-7));

  FREE4(dX, dDX, dR, dD);
}

TEST_CASE("operator cos(): cos(π/3)+2 -> 0.5−1.732e", "[dual][cos]")
{
  double x = M_PI / 3, dx = 2.0, outR, outD;
  double *dX, *dDX, *dR, *dD;
  ALLOC_COPY(dX, &x, sizeof(x));
  ALLOC_COPY(dDX, &dx, sizeof(dx));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_cos<<<1, 1>>>(dX, dDX, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(0.5).margin(1e-7));
  REQUIRE(outD == Approx(-dx * std::sin(x)).margin(1e-7));

  FREE4(dX, dDX, dR, dD);
}

TEST_CASE("operator exp(): exp(1)+3e→e+3ee", "[dual][exp]")
{
  double x = 1.0, dx = 3.0, outR, outD;
  double *dX, *dDX, *dR, *dD;
  ALLOC_COPY(dX, &x, sizeof(x));
  ALLOC_COPY(dDX, &dx, sizeof(dx));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_exp<<<1, 1>>>(dX, dDX, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(std::exp(x)).margin(1e-7));
  REQUIRE(outD == Approx(dx * std::exp(x)).margin(1e-7));

  FREE4(dX, dDX, dR, dD);
}

TEST_CASE("operator sqrt(): sqrt(4)+1.5e -> 2+0.375e", "[dual][sqrt]")
{
  double x = 4.0, dx = 1.5, outR, outD;
  double *dX, *dDX, *dR, *dD;
  ALLOC_COPY(dX, &x, sizeof(x));
  ALLOC_COPY(dDX, &dx, sizeof(dx));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_sqrt<<<1, 1>>>(dX, dDX, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(2.0).margin(1e-12));
  REQUIRE(outD == Approx(0.375).margin(1e-12));

  FREE4(dX, dDX, dR, dD);
}

TEST_CASE("operator pow() w 3: (2+1e)^3=8+12e", "[dual][pow]")
{
  double x = 2.0, dx = 1.0, outR, outD;
  double *dX, *dDX, *dR, *dD;
  ALLOC_COPY(dX, &x, sizeof(x));
  ALLOC_COPY(dDX, &dx, sizeof(dx));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_pow3<<<1, 1>>>(dX, dDX, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(8.0).margin(1e-12));
  REQUIRE(outD == Approx(12.0).margin(1e-12));

  FREE4(dX, dDX, dR, dD);
}

TEST_CASE("operator atan2(): atan2(1+1e,1+0e)=π/4+0.5e", "[dual][atan2]")
{
  double y = 1.0, dy = 1.0, x = 1.0, dx = 0.0, outR, outD;
  double *dY, *dDY, *dX, *dDX, *dR, *dD;
  ALLOC_COPY(dY, &y, sizeof(y));
  ALLOC_COPY(dDY, &dy, sizeof(dy));
  ALLOC_COPY(dX, &x, sizeof(x));
  ALLOC_COPY(dDX, &dx, sizeof(dx));
  cudaMalloc(&dR, sizeof(double));
  cudaMalloc(&dD, sizeof(double));

  test_atan2<<<1, 1>>>(dY, dDY, dX, dDX, dR, dD);
  cudaDeviceSynchronize();
  cudaMemcpy(&outR, dR, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&outD, dD, sizeof(double), cudaMemcpyDeviceToHost);

  REQUIRE(outR == Approx(M_PI / 4.0).margin(1e-7));
  REQUIRE(outD == Approx(0.5).margin(1e-7));

  FREE6(dY, dDY, dX, dDX, dR, dD);
}
