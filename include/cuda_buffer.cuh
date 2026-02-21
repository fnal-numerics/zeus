#pragma once

#include <cstddef>
#include <vector>
#include <array>
#include <cuda_runtime.h>
#include "exception.hpp" // CudaError<3>/<4>

namespace zeus {

  /// RAII wrapper for CUDA device memory allocation.
  /// Manages device memory lifetime with automatic cleanup and supports
  /// deep copy semantics for safe resource management.
  template <typename T>
  struct CudaBuffer {
    T* d = nullptr; ///< Device pointer
    size_t sz = 0;  ///< Number of elements

    CudaBuffer() noexcept = default;
    explicit CudaBuffer(std::size_t n); ///< Allocate n elements on device
    template <std::size_t N>
    explicit CudaBuffer(
      const std::array<T, N>& host); ///< Allocate and copy from host array
    CudaBuffer(CudaBuffer const& o); ///< Deep copy from device to device
    CudaBuffer& operator=(
      CudaBuffer const& o);              ///< Copy-assign via copy-and-swap
    CudaBuffer(CudaBuffer&& o) noexcept; ///< Move constructor
    CudaBuffer& operator=(CudaBuffer&& o) noexcept; ///< Move assignment
    ~CudaBuffer();                                  ///< Free device memory

    void swap(CudaBuffer& o) noexcept;   ///< Swap device pointers and sizes
    T* data() const noexcept;            ///< Raw device pointer accessor
    operator T*() const noexcept;        ///< Implicit conversion to raw pointer
    size_t size() const noexcept;        ///< Number of elements
    std::vector<T> copy_to_host() const; ///< Copy to new host vector
    int copy_to_host(
      std::vector<T>& out) const; ///< Copy to existing vector, returns status
    int copy_to_host(T* out,
                     size_t n) const; ///< Copy to raw pointer, returns status
  };

  // ──────────────────────────────────────────────────────────────────────────
  // Implementation
  // ──────────────────────────────────────────────────────────────────────────

  template <typename T>
  CudaBuffer<T>::CudaBuffer(std::size_t n) : d(nullptr), sz(n)
  {
    if (sz > 0) {
      auto st = cudaMalloc(&d, sz * sizeof(T));
      if (st != cudaSuccess)
        throw CudaError(st, "cudaMalloc failed");
    }
  }

  template <typename T>
  template <std::size_t N>
  CudaBuffer<T>::CudaBuffer(const std::array<T, N>& host) : d(nullptr), sz(N)
  {
    if (sz > 0) {
      cudaError_t st = cudaMalloc(&d, sz * sizeof(T));
      if (st != cudaSuccess)
        throw CudaError(st, "cudaMalloc failed in array ctor");
      st = cudaMemcpy(d, host.data(), sz * sizeof(T), cudaMemcpyHostToDevice);
      if (st != cudaSuccess) {
        cudaFree(d);
        throw CudaError(st, "cudaMemcpy H->D failed in array ctor");
      }
    }
  }

  template <typename T>
  CudaBuffer<T>::CudaBuffer(CudaBuffer const& o) : d(nullptr), sz(o.sz)
  {
    if (sz > 0) {
      auto st = cudaMalloc(&d, sz * sizeof(T));
      if (st != cudaSuccess)
        throw CudaError(st, "cudaMalloc failed in copy ctor");
      st = cudaMemcpy(d, o.d, sz * sizeof(T), cudaMemcpyDeviceToDevice);
      if (st != cudaSuccess) {
        cudaFree(d);
        throw CudaError(st, "cudaMemcpy D->D failed in copy ctor");
      }
    }
  }

  template <typename T>
  CudaBuffer<T>&
  CudaBuffer<T>::operator=(CudaBuffer const& o)
  {
    if (this != &o) {
      CudaBuffer temp(o);
      swap(temp);
    }
    return *this;
  }

  template <typename T>
  CudaBuffer<T>::CudaBuffer(CudaBuffer&& o) noexcept : d(o.d), sz(o.sz)
  {
    o.d = nullptr;
    o.sz = 0;
  }

  template <typename T>
  CudaBuffer<T>&
  CudaBuffer<T>::operator=(CudaBuffer&& o) noexcept
  {
    if (this != &o) {
      if (d)
        cudaFree(d);
      d = o.d;
      sz = o.sz;
      o.d = nullptr;
      o.sz = 0;
    }
    return *this;
  }

  template <typename T>
  CudaBuffer<T>::~CudaBuffer()
  {
    if (d)
      cudaFree(d);
    d = nullptr;
  }

  template <typename T>
  void
  CudaBuffer<T>::swap(CudaBuffer& o) noexcept
  {
    using std::swap;
    swap(d, o.d);
    swap(sz, o.sz);
  }

  template <typename T>
  T*
  CudaBuffer<T>::data() const noexcept
  {
    return d;
  }

  template <typename T>
  CudaBuffer<T>::operator T*() const noexcept
  {
    return d;
  }

  template <typename T>
  size_t
  CudaBuffer<T>::size() const noexcept
  {
    return sz;
  }

  template <typename T>
  std::vector<T>
  CudaBuffer<T>::copy_to_host() const
  {
    std::vector<T> out(sz);
    if (sz > 0) {
      auto st =
        cudaMemcpy(out.data(), d, sz * sizeof(T), cudaMemcpyDeviceToHost);
      if (st != cudaSuccess)
        throw CudaError(st, "cudaMemcpy D→H failed");
    }
    return out;
  }

  template <typename T>
  int
  CudaBuffer<T>::copy_to_host(std::vector<T>& out) const
  {
    out.resize(sz);
    if (sz > 0) {
      cudaError_t st =
        cudaMemcpy(out.data(), d, sz * sizeof(T), cudaMemcpyDeviceToHost);
      if (st != cudaSuccess)
        return 4;
    }
    return 0;
  }

  template <typename T>
  int
  CudaBuffer<T>::copy_to_host(T* out, size_t n) const
  {
    if (n != sz)
      return 1;
    if (sz > 0) {
      cudaError_t st =
        cudaMemcpy(out, d, sz * sizeof(T), cudaMemcpyDeviceToHost);
      if (st != cudaSuccess)
        return 4;
    }
    return 0;
  }

  template <typename T>
  bool
  operator==(CudaBuffer<T> const& a, CudaBuffer<T> const& b)
  {
    return a.d == b.d && a.sz == b.sz;
  }

  // template aliases
  using DoubleBuffer = CudaBuffer<double>;
}
