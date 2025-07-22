#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "matrix.cuh"
#include <cuda_runtime.h>
#include <vector>

// Square matrix host element access & dims tests
//   Create an NxN matrix, fill with values and verify rows()==cols()==N
//   and operator()(i,j) retrieving exactly what was stored
TEST_CASE("matrix:square host element access & dims", "[matrix][host][square]") {
  constexpr std::size_t N = 4;

  // squre NxN matrixi
  Matrix<double> m(N, N);

  // fill host buffer: m(i,j) = i*N + j
  // This yields the flattened sequence [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      m(i,j) = static_cast<double>(i * N + j);
    }
  }

  // check that both dimensions are equal to N
  REQUIRE(m.rows() == N);
  REQUIRE(m.cols() == N);

  // verify each element comes back exactly as stored
  for (std::size_t i = 0; i < N; ++i) {
    for (std::size_t j = 0; j < N; ++j) {
      double expect = double(i * N + j);
      REQUIRE(m(i,j) == Catch::Approx(expect));
    }
  }
}

// host element access, rows(), cols()
TEST_CASE("matrix: not square host element access & dims", "[matrix][host]") {
  constexpr std::size_t R = 2, C = 3;
  Matrix<double> m(R,C);
  

  // fill on host  m(i,j) = 1 + i*C + j 
  // so that the flattened sequence is [1,2,3,4,5,6]
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      m(i,j) = static_cast<double>(i * C + j + 1.0);

  // dim
  REQUIRE(m.rows() == R);
  REQUIRE(m.cols() == C);

  // host readback and check
  for (std::size_t i = 0; i < R; ++i) {
    for (std::size_t j = 0; j < C; ++j) {
      double expect = double(i * C + j + 1);
      REQUIRE(m(i,j) == Catch::Approx(expect));
    }
  }
}

// kernel to read back every entry with Matrix<double>::operator() from the device
template<int R,int C>
__global__ void matrix_device_access(Matrix<double> m, double* out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < R*C) {
    int i = idx / C;
    int j = idx % C;
    out[idx] = m(i,j);
  }
}

TEST_CASE("matrix: syncHostToDevice and device data()", "[matrix][device]") {
  constexpr int R = 2, C = 3, N = R*C;
  Matrix<double> m(R,C);

  // fill host buffer
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j)
      m(i,j) = double(i * C + j + 10);

  // push to GPU
  m.syncHostToDevice();

  // allocate output
  double* d_out = nullptr;
  cudaMalloc(&d_out, N * sizeof(double));
  // launch 1 block of N threads
  matrix_device_access<R,C><<<1, N>>>(m, d_out);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // copy back and check
  std::vector<double> host_out(N);
  cudaMemcpy(host_out.data(), d_out, N*sizeof(double), cudaMemcpyDeviceToHost);

  for (int idx = 0; idx < N; ++idx) {
    double expect = double(idx + 10);
    REQUIRE(host_out[idx] == Catch::Approx(expect));
  }
  cudaFree(d_out);
}

// test copy constructor - should pass with flying colours
TEST_CASE("matrix: test compilers implicitly generated shallow copy constructor", "[matrix][copy]") {
  constexpr std::size_t R = 3, C = 2;
  Matrix<double> m1(R,C);
  // fill m1
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      m1(i,j) = double(i * C + j) * 1.5;

  // copy constructor
  Matrix<double> m2(m1);

  // host readback on m2
  for (std::size_t i = 0; i < R; ++i) {
    for (std::size_t j = 0; j < C; ++j) {
      double expect = double(i * C + j) * 1.5;
      REQUIRE(m2(i,j) == Catch::Approx(expect));
    }
  }
}

// test copy constructor deep -- should fail without explicit copy constructor
TEST_CASE("matrix: copy constructor deep‑copies, no aliasing!", "[matrix][copy][deep]") {
  constexpr std::size_t R = 2, C = 2;
  Matrix<double> a(R,C);
  // fill 'a' with 0,1,2,3
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
       a(i,j) = double(i*C + j);

  // make the copy
  Matrix<double> b(a);

  // mutate the original
  a(0,0) = 999.0;
  a(1,1) = -123.0;

  // the copy must NOT see those changes
  REQUIRE(b(0,0) == Catch::Approx(0.0));
  REQUIRE(b(1,1) == Catch::Approx(3.0));
}


// move constructor (old object releases ownership)
TEST_CASE("matrix: move constructor transfers ownership and nulls source", "[matrix][move]") {
    Matrix<double> src(1,1);
    src(0,0) = 123.456;

    // sync so that host_data_ and device_data_ are consistent
    src.syncHostToDevice();

    // capture the host pointer before move
    double* old_host_ptr = src.data();

    Matrix<double> dst(std::move(src));

    // dst has the data
    REQUIRE(dst.rows() == 1);
    REQUIRE(dst.cols() == 1);
    REQUIRE(dst(0,0) == Catch::Approx(123.456));

    // src has been nulled out
    REQUIRE(src.rows() == 0);
    REQUIRE(src.cols() == 0);
    REQUIRE(src.data() == nullptr);

    // dst still uses the same host buffer
    REQUIRE(dst.data() == old_host_ptr);
}

