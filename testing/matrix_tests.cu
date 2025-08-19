#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include "matrix.hpp"
#include <cuda_runtime.h>

#include <array>
#include <vector>

// helper to read device buffer back to host
template <typename T>
static std::vector<T> read_device(const Matrix<T>& m) {
  const std::size_t n = m.rows() * m.cols();
  std::vector<T> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), m.device_data(), n * sizeof(T),
                        cudaMemcpyDeviceToHost));
  return h;
}

// host element access & dims (square)
TEST_CASE("matrix: square host element access & dims", "[matrix][host][square]") {
  constexpr std::size_t N = 4;

  Matrix<double> m(N, N);

  // m(i,j) = i*N + j
  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      m(i, j) = static_cast<double>(i * N + j);

  REQUIRE(m.rows() == N);
  REQUIRE(m.cols() == N);

  for (std::size_t i = 0; i < N; ++i)
    for (std::size_t j = 0; j < N; ++j)
      REQUIRE(m(i, j) == Catch::Approx(double(i * N + j)));
}

// host element access & dims (rectangular)
TEST_CASE("matrix: not square host element access & dims", "[matrix][host]") {
  constexpr std::size_t R = 2, C = 3;
  Matrix<double> m(R, C);

  // m(i,j) = 1 + i*C + j  -> [1,2,3,4,5,6]
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      m(i, j) = static_cast<double>(1 + i * C + j);

  REQUIRE(m.rows() == R);
  REQUIRE(m.cols() == C);

  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      REQUIRE(m(i, j) == Catch::Approx(double(1 + i * C + j)));
}

// copy constructor (values match)
TEST_CASE("matrix: copy constructor deep-copies values", "[matrix][copy]") {
  constexpr std::size_t R = 3, C = 2;
  Matrix<double> m1(R, C);

  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      m1(i, j) = double(i * C + j) * 1.5;

  Matrix<double> m2(m1); // deep copy

  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      REQUIRE(m2(i, j) == Catch::Approx(double(i * C + j) * 1.5));
}

// deep copy constructor (no aliasing)
TEST_CASE("matrix: copy constructor deep-copies, no aliasing", "[matrix][copy][deep]") {
  constexpr std::size_t R = 2, C = 2;
  Matrix<double> a(R, C);

  // fill a = [0,1,2,3]
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      a(i, j) = double(i * C + j);

  Matrix<double> b(a); // deep copy

  // mutate original
  a(0, 0) = 999.0;
  a(1, 1) = -123.0;

  // b must stay with old values
  REQUIRE(b(0, 0) == Catch::Approx(0.0));
  REQUIRE(b(1, 1) == Catch::Approx(3.0));
}

// copy assignment deep-copies + syncs device from the new host
TEST_CASE("matrix: copy assignment deep-copies and syncs device", "[matrix][assign][copy]") {
  constexpr std::size_t R = 2, C = 2;

  Matrix<double> a(R, C), b(R, C);
  // a: 1,2,3,4 ; b: 5,6,7,8
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j) {
      a(i, j) = double(1 + i * C + j);
      b(i, j) = double(5 + i * C + j);
    }

  b = a; // deep copy + H->D sync for b

  // host check
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      REQUIRE(b(i, j) == Catch::Approx(double(1 + i * C + j)));

  // device check (read back)
  auto dev = read_device(b);
  REQUIRE(dev.size() == R * C);
  for (std::size_t k = 0; k < dev.size(); ++k)
    REQUIRE(dev[k] == Catch::Approx(double(1 + k)));
}

// copy constructor from a source whose device buffer is stale
TEST_CASE("matrix: copy ctor uses source host data when device stale", "[matrix][copy][host-source]") {
  constexpr std::size_t R = 2, C = 3, N = R * C;

  Matrix<double> m1(R, C);
  // mutate via operator() only (device buffer intentionally left stale)
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      m1(i, j) = double(10 + i * C + j);

  Matrix<double> m2(m1); // copy should allocate + copy host + H->D sync for m2

  // host values match
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      REQUIRE(m2(i, j) == Catch::Approx(double(10 + i * C + j)));

  // device values match (read back from m2.device_data())
  auto dev = read_device(m2);
  REQUIRE(dev.size() == N);
  for (int k = 0; k < N; ++k)
    REQUIRE(dev[k] == Catch::Approx(double(10 + k)));
}

// move constructor: transfers ownership, nulls source
TEST_CASE("matrix: move constructor transfers ownership and nulls source", "[matrix][move][ctor]") {
  Matrix<double> src(1, 1);
  src(0, 0) = 123.456;

  double* old_host_ptr = src.host_data();

  Matrix<double> dst(std::move(src));

  REQUIRE(dst.rows() == 1);
  REQUIRE(dst.cols() == 1);
  REQUIRE(dst(0, 0) == Catch::Approx(123.456));

  REQUIRE(src.rows() == 0);
  REQUIRE(src.cols() == 0);
  REQUIRE(src.host_data() == nullptr);

  REQUIRE(dst.host_data() == old_host_ptr);
}

// move assignment: transfers ownership, nulls source
TEST_CASE("matrix: move assignment transfers ownership and nulls source", "[matrix][move][assign]") {
  constexpr std::size_t R = 2, C = 2;

  Matrix<double> m1(R, C);
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      m1(i, j) = double(1 + i * C + j); // 1,2,3,4

  double* old_ptr = m1.host_data();

  Matrix<double> m2;        // empty
  m2 = std::move(m1);       // move-assign

  REQUIRE(m2.rows() == R);
  REQUIRE(m2.cols() == C);
  for (std::size_t i = 0; i < R; ++i)
    for (std::size_t j = 0; j < C; ++j)
      REQUIRE(m2(i, j) == Catch::Approx(double(1 + i * C + j)));

  REQUIRE(m1.rows() == 0);
  REQUIRE(m1.cols() == 0);
  REQUIRE(m1.host_data() == nullptr);
  REQUIRE(m2.host_data() == old_ptr);
}

// swap correctness
TEST_CASE("swap(Matrix,Matrix) swaps buffers and dims", "[matrix][swap]") {
  Matrix<double> a(1, 2), b(2, 1);
  a(0, 0) = 11;
  a(0, 1) = 12;
  b(0, 0) = 21;
  b(1, 0) = 22;

  using std::swap;
  swap(a, b);

  // now a has the old b
  REQUIRE(a.rows() == 2);
  REQUIRE(a.cols() == 1);
  REQUIRE(a(0, 0) == Catch::Approx(21));
  REQUIRE(a(1, 0) == Catch::Approx(22));

  // and b has the old a
  REQUIRE(b.rows() == 1);
  REQUIRE(b.cols() == 2);
  REQUIRE(b(0, 0) == Catch::Approx(11));
  REQUIRE(b(0, 1) == Catch::Approx(12));
}

// copy assignment 
TEST_CASE("matrix: assignment host (copy-assign)", "[matrix][host][assign]") {
  Matrix<double> m1(2, 2);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      m1(i, j) = 2 * i + j;

  Matrix<double> m2;
  m2 = m1; //copy assignment

  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      REQUIRE(m2(i, j) == m1(i, j));
}

// setter tests 
// per-element set(i,j, x) performs immediate H->D sync of that element
TEST_CASE("matrix: set(i,j, x) updates device per element", "[matrix][device][set]") {
  Matrix<double> m(2, 2);
  m.set(0, 0, 11.0);
  m.set(0, 1, 12.0);
  m.set(1, 0, 13.0);
  m.set(1, 1, 14.0);

  // host view
  REQUIRE(m(0, 0) == Catch::Approx(11.0));
  REQUIRE(m(0, 1) == Catch::Approx(12.0));
  REQUIRE(m(1, 0) == Catch::Approx(13.0));
  REQUIRE(m(1, 1) == Catch::Approx(14.0));

  // device view
  auto dev = read_device(m);
  REQUIRE(dev.size() == 4);
  REQUIRE(dev[0] == Catch::Approx(11.0));
  REQUIRE(dev[1] == Catch::Approx(12.0));
  REQUIRE(dev[2] == Catch::Approx(13.0));
  REQUIRE(dev[3] == Catch::Approx(14.0));
}

// bulk set from std::array copies entire buffer to device
TEST_CASE("matrix: set(std::array) bulk-updates device", "[matrix][device][set][array]") {
  constexpr int R = 2, C = 3, N = R * C;
  Matrix<double> m(R, C);

  std::array<double, N> init{};
  for (int k = 0; k < N; ++k) init[k] = 100.0 + k;

  m.set(init);

  // host check
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j)
      REQUIRE(m(i, j) == Catch::Approx(100.0 + i * C + j));

  // device check
  auto dev = read_device(m);
  REQUIRE(dev.size() == N);
  for (int k = 0; k < N; ++k)
    REQUIRE(dev[k] == Catch::Approx(100.0 + k));
}

// bulk set from raw pointer
TEST_CASE("matrix: set(ptr,count) bulk-updates device", "[matrix][device]") {
  constexpr int R = 3, C = 2, N = R * C;
  Matrix<double> m(R, C);

  double buf[N];
  for (int k = 0; k < N; ++k) buf[k] = -50.0 + k;

  m.set(buf, N);

  // host check
  for (int i = 0; i < R; ++i)
    for (int j = 0; j < C; ++j)
      REQUIRE(m(i, j) == Catch::Approx(-50.0 + i * C + j));

  // device check
  auto dev = read_device(m);
  REQUIRE(dev.size() == N);
  for (int k = 0; k < N; ++k)
    REQUIRE(dev[k] == Catch::Approx(-50.0 + k));
}

// error-path tests

// constructor rejects zero dims
TEST_CASE("matrix: constructor rejects zero dims", "[matrix][errors]") {
  REQUIRE_THROWS_AS((Matrix<double>(0, 1)), std::invalid_argument);
  REQUIRE_THROWS_AS((Matrix<double>(1, 0)), std::invalid_argument);
}

// OOB access throws
TEST_CASE("matrix: operator() bounds check", "[matrix][errors][bounds]") {
  Matrix<double> m(2, 2);
  REQUIRE_THROWS_AS(m(2, 0), std::out_of_range);
  REQUIRE_THROWS_AS(m(0, 2), std::out_of_range);
}

// set(std::array) with mismatched size throws
TEST_CASE("matrix: set(std::array) rejects wrong size", "[matrix][errors][set]") {
  Matrix<double> m(2, 2);
  std::array<double, 3> wrong{{1, 2, 3}}; // expected 4
  REQUIRE_THROWS_AS(m.set(wrong), std::invalid_argument);
}

