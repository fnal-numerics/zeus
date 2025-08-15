#pragma once
#include <cuda_runtime.h>
#include <cstddef>

/*
  This class is for device allocation only from GPU heap memory
*/
template <typename T>
class DeviceMatrix {
private:
  T* data_ = nullptr;
  std::size_t cols_ = 0;
  std::size_t rows_ = 0;

public:
  __device__
  DeviceMatrix(std::size_t rows, std::size_t cols)
  : data_(nullptr), rows_(rows), cols_(cols)
  {
    const std::size_t sz = rows_*cols_*sizeof(T);
    data_ = static_cast<T*>(malloc(sz));
    if (!data_) asm("trap;");
  }

  DeviceMatrix(const DeviceMatrix&) = delete; // delete copy constructor
  DeviceMatrix& operator=(const DeviceMatrix&) = delete; // delete lvalue assign completely

  __device__
  ~DeviceMatrix() {
    if(data_) free(data_);
  }

  __device__  T* data() { return data_; }
  __device__ const T* data() const { return data_; }

  __device__ std::size_t rows() const { return rows_; }
  __device__ std::size_t cols() const { return cols_; }

  __device__ T& operator()(std::size_t i, std::size_t j) {
    return data_[i*cols_ + j];
  }
  __device__ const T& operator()(std::size_t i, std::size_t j) const {
    return data_[i*cols_ + j];
  }

  __device__ void set(std::size_t i, std::size_t j, const T& x) {
    data_[i*cols_ + j] = x;
  }

};
