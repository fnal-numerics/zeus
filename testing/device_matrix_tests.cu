#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>
#include <cuda_runtime.h>
#include <type_traits>
#include <vector>
#include <cstdint>

#include "device_matrix.cuh"

// CUDA error helper 
static inline void ck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    FAIL(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

// Kernels that use DeviceMatrix on device 
template <typename T>
__global__ void kernel_fill_and_copy(T* out, std::size_t rows, std::size_t cols, T base) {
  // Single thread constructs, fills, and copies out the matrix.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    DeviceMatrix<T> M(rows, cols);

    // Fill with base + (i * cols + j)
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        M(i, j) = static_cast<T>(base + static_cast<T>(i * cols + j));
      }
    }

    // Copy to global out buffer (row-major) so host can validate
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j) {
        out[i * cols + j] = M(i, j);
      }
    }
    // M goes out of scope here; ~DeviceMatrix frees device malloc.
  }
}

template <typename T>
static void run_fill_and_check(std::size_t rows, std::size_t cols, T base) {
  const std::size_t N = rows * cols;

  // Device buffer for results
  T* d_out = nullptr;
  ck(cudaMalloc(&d_out, N * sizeof(T)), "cudaMalloc(d_out)");

  // Launch one block, one thread (the kernel internally loops)
  kernel_fill_and_copy<T><<<1, 1>>>(d_out, rows, cols, base);
  ck(cudaGetLastError(), "kernel launch");
  ck(cudaDeviceSynchronize(), "kernel sync");

  // Copy back and verify
  std::vector<T> h(N);
  ck(cudaMemcpy(h.data(), d_out, N * sizeof(T), cudaMemcpyDeviceToHost), "memcpy D2H");
  ck(cudaFree(d_out), "cudaFree(d_out)");

  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const std::size_t idx = i * cols + j;
      const T expected = static_cast<T>(base + static_cast<T>(idx));
      REQUIRE(h[idx] == expected);
    }
  }
}

TEST_CASE("DeviceMatrix is non-copyable and non-assignable", "[traits]") {
  STATIC_REQUIRE(!std::is_copy_constructible_v<DeviceMatrix<int>>);
  STATIC_REQUIRE(!std::is_copy_assignable_v<DeviceMatrix<int>>);
  STATIC_REQUIRE(std::is_destructible_v<DeviceMatrix<int>>);
}

TEST_CASE("DeviceMatrix<int> fill and readback small", "[functional][int]") {
  // Ensure enough device heap for device-side malloc/free
  ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 20), "set heap limit");
  run_fill_and_check<int>(3, 4, 10);
}

TEST_CASE("DeviceMatrix<double> fill and readback rectangular", "[functional][double]") {
  ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 20), "set heap limit");
  run_fill_and_check<double>(5, 2, 1.5);
}

TEST_CASE("Repeated construction/destruction test", "[lifecycle]") {
  ck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1 << 20), "set heap limit");
  for (int k = 0; k < 5; ++k) {
    run_fill_and_check<std::int32_t>(2, 3, k * 100);
  }
}

