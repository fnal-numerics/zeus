#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <stdexcept>
#include "utils.cuh"

// Test struct to verify pointer dereferencing
struct TestData {
  double value;
  int count;
};

// ============================================================================
// HOST-SIDE TESTS
// ============================================================================

TEST_CASE("NonNull: Construction from valid pointer", "[NonNull][host]")
{
  double value = 42.0;
  double* ptr = &value;
  
  REQUIRE_NOTHROW([&]() {
    util::NonNull<double*> nn(ptr);
  }());
}

TEST_CASE("NonNull: Construction from nullptr throws", "[NonNull][host]")
{
  double* null_ptr = nullptr;
  
  REQUIRE_THROWS_AS(
    [&]() { util::NonNull<double*> nn(null_ptr); }(),
    std::invalid_argument
  );
}

TEST_CASE("NonNull: Get method returns pointer", "[NonNull][host]")
{
  double value = 3.14159;
  double* ptr = &value;
  util::NonNull<double*> nn(ptr);
  
  REQUIRE(nn.get() == ptr);
}

TEST_CASE("NonNull: Implicit conversion to pointer", "[NonNull][host]")
{
  double value = 2.71828;
  double* ptr = &value;
  util::NonNull<double*> nn(ptr);
  
  double* converted = nn;
  REQUIRE(converted == ptr);
}

TEST_CASE("NonNull: Dereference operator*", "[NonNull][host]")
{
  double value = 1.41421;
  double* ptr = &value;
  util::NonNull<double*> nn(ptr);
  
  REQUIRE(*nn == 1.41421);
  
  // Test assignment through dereference
  *nn = 2.71828;
  REQUIRE(value == 2.71828);
}

TEST_CASE("NonNull: Arrow operator for struct", "[NonNull][host]")
{
  TestData data{99.99, 42};
  TestData* ptr = &data;
  util::NonNull<TestData*> nn(ptr);
  
  REQUIRE(nn->value == 99.99);
  REQUIRE(nn->count == 42);
  
  // Test modification through arrow operator
  nn->value = 55.5;
  nn->count = 100;
  REQUIRE(data.value == 55.5);
  REQUIRE(data.count == 100);
}

TEST_CASE("NonNull: Copy construction with convertible types", "[NonNull][host]")
{
  double value = 7.0;
  double* ptr = &value;
  util::NonNull<double*> nn1(ptr);
  
  // Copy construct
  util::NonNull<double*> nn2(nn1);
  REQUIRE(nn2.get() == ptr);
  REQUIRE(*nn2 == 7.0);
}

TEST_CASE("NonNull: Array pointer operations", "[NonNull][host]")
{
  double arr[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
  util::NonNull<double*> nn(arr);
  
  REQUIRE(nn.get()[0] == 1.0);
  REQUIRE(nn.get()[4] == 5.0);
  
  nn.get()[2] = 99.0;
  REQUIRE(arr[2] == 99.0);
}

// ============================================================================
// DEVICE-SIDE TESTS
// ============================================================================

// Simple kernel that uses non_null pointers for computation
__global__ void
kernel_use_non_null(util::NonNull<double*> input,
                    util::NonNull<double*> output,
                    int size)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = input[idx] * 2.0;
  }
}

// Kernel that tests dereferencing with non_null
__global__ void
kernel_dereference(util::NonNull<double*> ptr, double* result)
{
  if (threadIdx.x == 0) {
    *result = *ptr + 1.0;
  }
}

// Kernel that accesses struct members through non_null
__global__ void
kernel_struct_access(util::NonNull<TestData*> data, double* out_val, int* out_count)
{
  if (threadIdx.x == 0) {
    *out_val = data->value;
    *out_count = data->count;
  }
}

TEST_CASE("NonNull: Usage in CUDA kernel - array operations", "[NonNull][device]")
{
  const int size = 10;
  double h_input[size];
  double h_output[size] = {0};
  
  // Initialize input
  for (int i = 0; i < size; ++i) {
    h_input[i] = static_cast<double>(i + 1);
  }
  
  // Allocate device memory
  double* d_input = nullptr;
  double* d_output = nullptr;
  cudaMalloc(&d_input, size * sizeof(double));
  cudaMalloc(&d_output, size * sizeof(double));
  
  // Copy input to device
  cudaMemcpy(d_input, h_input, size * sizeof(double), cudaMemcpyHostToDevice);
  
  // Create non_null wrappers and launch kernel
  util::NonNull<double*> nn_input(d_input);
  util::NonNull<double*> nn_output(d_output);
  
  kernel_use_non_null<<<2, 8>>>(nn_input, nn_output, size);
  cudaDeviceSynchronize();
  
  // Copy output back to host
  cudaMemcpy(h_output, d_output, size * sizeof(double), cudaMemcpyDeviceToHost);
  
  // Verify results
  for (int i = 0; i < size; ++i) {
    REQUIRE(h_output[i] == Catch::Approx(h_input[i] * 2.0));
  }
  
  // Cleanup
  cudaFree(d_input);
  cudaFree(d_output);
}

TEST_CASE("NonNull: Dereference in CUDA kernel", "[NonNull][device]")
{
  double h_value = 5.5;
  double h_result = 0.0;
  
  double* d_value = nullptr;
  double* d_result = nullptr;
  cudaMalloc(&d_value, sizeof(double));
  cudaMalloc(&d_result, sizeof(double));
  
  cudaMemcpy(d_value, &h_value, sizeof(double), cudaMemcpyHostToDevice);
  
  util::NonNull<double*> nn(d_value);
  kernel_dereference<<<1, 1>>>(nn, d_result);
  cudaDeviceSynchronize();
  
  cudaMemcpy(&h_result, d_result, sizeof(double), cudaMemcpyDeviceToHost);
  
  // h_result should be h_value + 1.0 = 6.5
  REQUIRE(h_result == Catch::Approx(6.5));
  
  cudaFree(d_value);
  cudaFree(d_result);
}

TEST_CASE("NonNull: Struct member access in CUDA kernel", "[NonNull][device]")
{
  TestData h_data{42.42, 123};
  double h_out_val = 0.0;
  int h_out_count = 0;
  
  TestData* d_data = nullptr;
  double* d_out_val = nullptr;
  int* d_out_count = nullptr;
  
  cudaMalloc(&d_data, sizeof(TestData));
  cudaMalloc(&d_out_val, sizeof(double));
  cudaMalloc(&d_out_count, sizeof(int));
  
  cudaMemcpy(d_data, &h_data, sizeof(TestData), cudaMemcpyHostToDevice);
  
  util::NonNull<TestData*> nn(d_data);
  kernel_struct_access<<<1, 1>>>(nn, d_out_val, d_out_count);
  cudaDeviceSynchronize();
  
  cudaMemcpy(&h_out_val, d_out_val, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_out_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
  
  REQUIRE(h_out_val == Catch::Approx(42.42));
  REQUIRE(h_out_count == 123);
  
  cudaFree(d_data);
  cudaFree(d_out_val);
  cudaFree(d_out_count);
}

// ============================================================================
// TYPE SAFETY TESTS (Compile-time checks via REQUIRE_COMPILES)
// ============================================================================

TEST_CASE("NonNull: Type conversion between compatible pointer types", "[NonNull][types]")
{
  // Test that we can construct from a derived type through a base pointer
  struct Base { int x; };
  struct Derived : Base { int y; };
  
  Derived derived{1, 2};
  Base* base_ptr = &derived;
  
  REQUIRE_NOTHROW([&]() {
    util::NonNull<Base*> nn(base_ptr);
  }());
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

TEST_CASE("NonNull: Null pointer detection is guard-railed", "[NonNull][errors]")
{
  // This test verifies that construction from nullptr is properly rejected
  double* invalid = nullptr;
  
  SECTION("Direct nullptr detection") {
    REQUIRE_THROWS_AS(
      [&]() { util::NonNull<double*> nn(invalid); }(),
      std::invalid_argument
    );
  }
}

TEST_CASE("NonNull: Multiple successive valid constructions", "[NonNull][host]")
{
  double val1 = 1.0;
  double val2 = 2.0;
  double val3 = 3.0;
  
  REQUIRE_NOTHROW([&]() {
    util::NonNull<double*> nn1(&val1);
    util::NonNull<double*> nn2(&val2);
    util::NonNull<double*> nn3(&val3);
    
    REQUIRE(*nn1 == 1.0);
    REQUIRE(*nn2 == 2.0);
    REQUIRE(*nn3 == 3.0);
  }());
}
