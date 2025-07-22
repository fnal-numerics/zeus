#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

template<typename T>
class Matrix {
private:
  T* host_data_   = nullptr;
  T* device_data_ = nullptr;
  std::size_t rows_ = 0, cols_ = 0;

public:
  // default constructor
  Matrix() noexcept = default;

  // allocate separate host & device buffers
  Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols)
  {
    if (!rows_ || !cols_)
      throw std::invalid_argument("Matrix dimensions must be > 0");

    host_data_ = static_cast<T*>(std::malloc(rows_*cols_*sizeof(T)));
    if (!host_data_)
      throw std::runtime_error("host malloc failed");

    cudaError_t err = cudaMalloc(&device_data_, rows_*cols_*sizeof(T));
    if (err != cudaSuccess)
      throw std::runtime_error("cudaMalloc failed");

    printf("Matrix: allocated host & device memory\n");
  }

  // picks the right pointer on host vs. device
  __host__ __device__
  T* data() {
  #ifdef __CUDA_ARCH__
    return device_data_;
  #else
    return host_data_;
  #endif
  }
  
  __host__ __device__
  T const* data() const {
  #ifdef __CUDA_ARCH__
    return device_data_;
  #else
    return host_data_;
  #endif
  }

  __host__ __device__
  T& operator()(std::size_t i, std::size_t j) {
    return data()[i*cols_ + j];
  }
  
  __host__ __device__
  T const& operator()(std::size_t i, std::size_t j) const {
    return data()[i*cols_ + j];
  }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }

  // after user fill host_data_ they need to call this function once to push host_data_ to device_data_
  void syncHostToDevice() {
    std::size_t sz = rows_*cols_*sizeof(T);
    cudaMemcpy(device_data_, host_data_, sz, cudaMemcpyHostToDevice);
  }

};
