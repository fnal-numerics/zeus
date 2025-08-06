#pragma once

#include <cstddef>
#include <vector>
#include <array> 
#include <cuda_runtime.h>
#include "exception.hpp"   // cuda_exception<3>/<4>

template <std::size_t DIM>
struct Result;

namespace zeus {

template<typename T>
struct cuda_buffer {
  T*     d   = nullptr;
  size_t sz  = 0;

  // default ctor (zero‐length)
  cuda_buffer() noexcept = default;

  // allocate n elements
  explicit cuda_buffer(std::size_t n)
    : d(nullptr), sz(n)
  {
    if (sz > 0) {
      auto st = cudaMalloc(&d, sz * sizeof(T));
      if (st != cudaSuccess)
        throw cuda_exception<3>("cudaMalloc failed");
    }
  }

  // ctor from host std::array<T,N>
  template <std::size_t N>
  explicit cuda_buffer(const std::array<T, N>& host)
    : d(nullptr), sz(N)
  {
    if (sz > 0) {
      cudaError_t st = cudaMalloc(&d, sz * sizeof(T));
      if (st != cudaSuccess)
        throw cuda_exception<3>("cudaMalloc failed in array ctor");
      st = cudaMemcpy(d,
                      host.data(),
                      sz * sizeof(T),
                      cudaMemcpyHostToDevice);
      if (st != cudaSuccess) {
        cudaFree(d);
        throw cuda_exception<4>("cudaMemcpy H->D failed in array ctor");
      }
     }
  }


  // deep‐copy copy‐ctor
  cuda_buffer(cuda_buffer const& o)
    : d(nullptr), sz(o.sz)
  {
    if (sz > 0) {
      auto st = cudaMalloc(&d, sz * sizeof(T));
      if (st != cudaSuccess)
        throw cuda_exception<3>("cudaMalloc failed in copy ctor");
      st = cudaMemcpy(d, o.d, sz * sizeof(T), cudaMemcpyDeviceToDevice);
      if (st != cudaSuccess) {
        cudaFree(d);
        throw cuda_exception<4>("cudaMemcpy D->D failed in copy ctor");
      }
    }
  }

  // copy‐assign via copy‐and‐swap
  cuda_buffer& operator=(cuda_buffer const& o) {
    cuda_buffer temp(o);
    swap(temp);
    return *this;
  }

  // move‐ctor
  cuda_buffer(cuda_buffer&& o) noexcept
    : d(o.d), sz(o.sz)
  {
    o.d = nullptr;
    o.sz = 0;
  }

  // move‐assign
  cuda_buffer& operator=(cuda_buffer&& o) noexcept {
    if (this != &o) {
      if (d) cudaFree(d);
      d    = o.d;
      sz   = o.sz;
      o.d  = nullptr;
      o.sz = 0;
    }
    return *this;
  }

  // destructor
  ~cuda_buffer() {
    if (d) cudaFree(d);
    d = nullptr;
  }

  // swap utility
  void swap(cuda_buffer& o) noexcept {
    using std::swap;
    swap(d,  o.d);
    swap(sz, o.sz);
  }

  // raw pointer accessor
  T* data() const noexcept { return d; }
  operator T*() const noexcept { return d; } // implicit conversion to T*
  size_t size() const noexcept { return sz; }

  // host copy helper
  std::vector<T> copy_to_host() const {
    std::vector<T> out(sz);
    if (sz > 0) {
      auto st = cudaMemcpy(out.data(), d, sz * sizeof(T),
                           cudaMemcpyDeviceToHost);
      if (st != cudaSuccess)
        throw cuda_exception<4>("cudaMemcpy D→H failed");
    }
    return out;
  }

  // vector& overload returning int status
  int copy_to_host(std::vector<T>& out) const {
    out.resize(sz);
    if (sz > 0) {
      cudaError_t st = cudaMemcpy(
        out.data(), d, sz * sizeof(T), cudaMemcpyDeviceToHost);
      if (st != cudaSuccess) return 4;
    }
    return 0;
  }

  // raw‐pointer overload returning int status
  int copy_to_host(T* out, size_t n) const {
    if (n != sz) return 1;
    if (sz > 0) {
      cudaError_t st = cudaMemcpy(
        out, d, sz * sizeof(T), cudaMemcpyDeviceToHost);
      if (st != cudaSuccess) return 4;
    }
    return 0;
  }

};

template<typename T>
bool operator==(cuda_buffer<T> const& a, cuda_buffer<T> const& b) {
  return true;
}



// template alieses
template<typename T>
using tbuf = cuda_buffer<T>;

using dbuf = tbuf<double>;

template<std::size_t DIM>
using result_buffer = tbuf<Result<DIM>>;

}
