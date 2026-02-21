#pragma once
#include <cuda_runtime.h>
#include <cstddef>

/// Matrix allocated from GPU heap memory (device-side only).
/// Uses malloc/free from device code for dynamic allocation within kernels.
/// Non-copyable and non-movable to prevent dangling pointers.
template <typename T>
class DeviceMatrix {
private:
  T* data_ = nullptr; ///< Device heap pointer
  int cols_ = 0;      ///< Number of columns
  int rows_ = 0;      ///< Number of rows

public:
  __device__ DeviceMatrix(
    int rows,
    int cols); ///< Allocate rows×cols matrix on device heap
  DeviceMatrix(const DeviceMatrix&) = delete;
  DeviceMatrix& operator=(const DeviceMatrix&) = delete;
  __device__ DeviceMatrix(DeviceMatrix&&) = delete;
  __device__ DeviceMatrix& operator=(DeviceMatrix&&) = delete;
  __device__ ~DeviceMatrix(); ///< Free device heap memory via release()

  __device__ void release(); ///< Free device heap memory and reset dimensions
  __device__ T* data();      ///< Raw pointer to device heap data
  __device__ const T* data() const; ///< Raw pointer to device heap data (const)
  __device__ int rows() const;      ///< Number of rows
  __device__ int cols() const;      ///< Number of columns
  __device__ T& operator()(
    int i,
    int j); ///< Access element at (i,j) in row-major order
  __device__ const T& operator()(int i, int j)
    const; ///< Access element at (i,j) in row-major order (const)
  __device__ void set(int i, int j,
                      const T& x); ///< Set element at (i,j)
};

// ──────────────────────────────────────────────────────────────────────────
// Implementation
// ──────────────────────────────────────────────────────────────────────────

template <typename T>
__device__
DeviceMatrix<T>::DeviceMatrix(int rows, int cols)
  : data_(nullptr), rows_(rows), cols_(cols)
{
  const std::size_t sz = static_cast<size_t>(rows_) * cols_ * sizeof(T);
  data_ = static_cast<T*>(malloc(sz));
  if (!data_)
    asm("trap;");
}

template <typename T>
__device__ DeviceMatrix<T>::~DeviceMatrix()
{
  release();
}

template <typename T>
__device__ void
DeviceMatrix<T>::release()
{
  if (data_) {
    free(data_);
    data_ = nullptr;
    rows_ = cols_ = 0;
  }
}

template <typename T>
__device__ T*
DeviceMatrix<T>::data()
{
  return data_;
}

template <typename T>
__device__ const T*
DeviceMatrix<T>::data() const
{
  return data_;
}

template <typename T>
__device__ int
DeviceMatrix<T>::rows() const
{
  return rows_;
}

template <typename T>
__device__ int
DeviceMatrix<T>::cols() const
{
  return cols_;
}

template <typename T>
__device__ T&
DeviceMatrix<T>::operator()(int i, int j)
{
  return data_[static_cast<size_t>(i) * cols_ + j];
}

template <typename T>
__device__ const T&
DeviceMatrix<T>::operator()(int i, int j) const
{
  return data_[static_cast<size_t>(i) * cols_ + j];
}

template <typename T>
__device__ void
DeviceMatrix<T>::set(int i, int j, const T& x)
{
  data_[static_cast<size_t>(i) * cols_ + j] = x;
}
