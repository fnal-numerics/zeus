#pragma once
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <cuda_runtime.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _err = (call);                                                 \
    if (_err != cudaSuccess) {                                                 \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(_err));                      \
    }                                                                          \
  } while (0)
#endif

template <typename T>
class Matrix {
private:
  void boundsCheck(std::size_t i, std::size_t j) const {
    if (i >= rows_ || j >= cols_)
      throw std::out_of_range("Matrix index out of range");
  }

  T* host_data_   = nullptr;
  T* device_data_ = nullptr;
  std::size_t rows_ = 0;
  std::size_t cols_ = 0;

public:
  Matrix() noexcept = default;

  Matrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols)
  {
    if (!rows_ || !cols_)
      throw std::invalid_argument("Matrix dimensions must be > 0");

    const std::size_t sz = rows_*cols_*sizeof(T);

    host_data_ = static_cast<T*>(std::malloc(sz));
    if (!host_data_) throw std::runtime_error("host malloc failed");

    CUDA_CHECK(cudaMalloc(&device_data_, sz));
  }

  template <std::size_t N>
  Matrix(std::size_t rows, std::size_t cols, const std::array<T, N>& init)
    : Matrix(rows, cols)
  {
    const std::size_t expected = rows_*cols_;
    if (N != expected)
      throw std::invalid_argument("std::array size must equal rows*cols");
    std::memcpy(host_data_, init.data(), expected*sizeof(T));
    CUDA_CHECK(cudaMemcpy(device_data_, host_data_,
                          expected*sizeof(T), cudaMemcpyHostToDevice));
  }

  // copy ctor
  Matrix(const Matrix& other)
    : rows_(other.rows_), cols_(other.cols_)
  {
#ifdef __CUDA_ARCH__
    // device copy, just copy the pointers and dims
    host_data_   = other.host_data_;  
    device_data_ = other.device_data_;
#else
    if (!rows_ || !cols_) return; // allow copying an empty matrix

    const std::size_t sz = rows_*cols_*sizeof(T);

    host_data_ = static_cast<T*>(std::malloc(sz));
    if (!host_data_) throw std::runtime_error("host malloc failed");

    // copy host buffer
    std::memcpy(host_data_, other.host_data_, sz);

    // allocate device and sync from *this host* to keep new object consistent
    CUDA_CHECK(cudaMalloc(&device_data_, sz));
    CUDA_CHECK(cudaMemcpy(device_data_, host_data_,
                          sz, cudaMemcpyHostToDevice));
#endif
  }

  // copy assignment 
  Matrix& operator=(const Matrix& other) {
#ifdef __CUDA_ARCH__
    // device: shallow copy
    host_data_   = other.host_data_;
    device_data_ = other.device_data_;
    rows_ = other.rows_; cols_ = other.cols_;
    return *this;
#else
    if (this == &other) return *this;
    Matrix tmp(other);   // strong exception guarantee
    swap(tmp);           // now *this has other's deep copy
    return *this;        // tmp dtor frees old resources
#endif
  }

  // move
  Matrix(Matrix&& o) noexcept
    : host_data_(o.host_data_)
    , device_data_(o.device_data_)
    , rows_(o.rows_)
    , cols_(o.cols_)
  {
    o.host_data_   = nullptr;
    o.device_data_ = nullptr;
    o.rows_ = o.cols_ = 0;
  }

  // move assignment
  Matrix& operator=(Matrix&& o) noexcept {
    if (this != &o) {
#ifdef __CUDA_ARCH__
      // device: just steal pointers
      host_data_   = o.host_data_;
      device_data_ = o.device_data_;
      rows_ = o.rows_; cols_ = o.cols_;
      o.host_data_ = nullptr;
      o.device_data_ = nullptr;
      o.rows_ = o.cols_ = 0;
#else
      Matrix tmp(std::move(o));
      swap(tmp);
#endif
    }
    return *this;
  }

  ~Matrix() {
#ifndef __CUDA_ARCH__
    if (host_data_)   std::free(host_data_);
    if (device_data_) cudaFree(device_data_);
#endif
  }

// set one element and copy just that element to device
void set(std::size_t i, std::size_t j, const T& x) {
  boundsCheck(i, j);
  const std::size_t idx = i*cols_ + j;
  host_data_[idx] = x;
  CUDA_CHECK(cudaMemcpy(device_data_ + idx, &x, sizeof(T),cudaMemcpyHostToDevice));
}

// bulk set from std::array
// copies entire buffer to device
template <std::size_t N>
void set(const std::array<T, N>& arr) {
  const std::size_t expected = rows_*cols_;
  if (N != expected)
    throw std::invalid_argument("std::array size must equal rows*cols");
  std::memcpy(host_data_, arr.data(), expected*sizeof(T));
  CUDA_CHECK(cudaMemcpy(device_data_, host_data_,expected*sizeof(T), cudaMemcpyHostToDevice));
}

// set from raw pointer (host memory)
void set(const T* host_ptr, std::size_t count) {
  const std::size_t expected = rows_*cols_;
  if (count != expected)
    throw std::invalid_argument("count must equal rows*cols");
  std::memcpy(host_data_, host_ptr, expected*sizeof(T));
  CUDA_CHECK(cudaMemcpy(device_data_, host_data_,expected*sizeof(T), cudaMemcpyHostToDevice));
}



__host__ __device__
const T* data() const noexcept {
#ifdef __CUDA_ARCH__
  return device_data_;
#else
  return host_data_;
#endif
}

  // Accessors
  T*       host_data()       noexcept { return host_data_; }
  const T* host_data() const noexcept { return host_data_; }

  T*       device_data()       noexcept { return device_data_; }
  const T* device_data() const noexcept { return device_data_; }

  T& operator()(std::size_t i, std::size_t j) {
    boundsCheck(i, j);
    return host_data_[i*cols_ + j];
  }
  const T& operator()(std::size_t i, std::size_t j) const {
    boundsCheck(i, j);
    return host_data_[i*cols_ + j];
  }

  std::size_t rows() const noexcept { return rows_; }
  std::size_t cols() const noexcept { return cols_; }

  void swap(Matrix& other) noexcept {
    using std::swap;
    swap(host_data_, other.host_data_);
    swap(device_data_, other.device_data_);
    swap(rows_, other.rows_);
    swap(cols_, other.cols_);
  }
};
