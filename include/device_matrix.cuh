#pragma once
#include <cuda_runtime.h>
#include <cstddef>

/*
  This class is for device allocation only from GPU heap memory
*/
template <typename T>
class DeviceMatrix:
private:
  T* data_ = nullptr;
  std::size_t cols_ = 0;
  std::size_t rows_ = 0;

public:
  __device__
  DeviceMatrix(std::size_t rows, std::size_t cols)
  : data_(nullptr), rows_(rows), cols_(cols) 
  {
    size_t sz = rows_ * cols_ * sizeof(T);
    data_ = (T*)malloc(sz);
    if(!data_) asm("trap;");
  }

  DeviceMatrix(Matrix const&) = delete; // delete copy constructor
  DeviceMatrix& operator=(Matrix const&) = delete; // delete lvalue assign completely

  __device__
  ~DeviceMatrix() {
    if(data_) free(data_);
  }

  __device__ T const*
  data() const { return data_; }

  __device__ T*
  data() const { return data_; }

 __device__ T&
 operator()(std::size_t i, std::size_t j) {
   return data()[i * cols_ + j];
 }

 __device__ T const&
 operator()(std::size_t i, std::size_t j) const {
   return data()[i * cols_ + j];
 }


