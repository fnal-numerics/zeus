#pragma once

#include <cstddef>
#include <array>
#include <vector>

#include "exception.hpp"
// our own exception template class that would return either status 3 or 4 depending on which failure occured
// 3 is for cudamalloc failure, and 4 is for kernel failure 

struct cuda_buffer {
  double * d = nullptr;
  size_t sz = 0;
 
  // allocate uninitialized device buffer 
  explicit cuda_buffer(std::size_t n);

  // build from host array, copying contents
  //    defined inline so the definition is visible at instantiation time
  template <std::size_t N>
  explicit cuda_buffer(const std::array<double, N>& h)
    : d(nullptr), sz(N)
  {
    if (sz > 0) {
      cudaError_t st = cudaMalloc(&d, sz * sizeof(double));
      if (st != cudaSuccess)
        throw cuda_exception<3>("cudaMalloc failed");
      st = cudaMemcpy(d, h.data(), sz * sizeof(double),
                      cudaMemcpyHostToDevice);
      if (st != cudaSuccess)
        throw cuda_exception<4>("cudaMemcpy H→D failed");
    }
  }
  // what about:
  //  1) copy ctor
  //  2) copy assignment
  //  3) move ctor
  //  4) move assignment
  ~cuda_buffer();
  // we could also have a copy_to_host member function something like:
  //int copy_to_host(double* buffer, size_t n) const;
  // and overloaded sets:
  //int copy_to_host(std::vector<double> & out) const;
  // or
  //std::vector<double> copy_to_host() const;

  // copy ctor / assignment
  cuda_buffer(cuda_buffer const& other);
  cuda_buffer& operator=(cuda_buffer const& other);

  // host-copy helpers
  std::vector<double> copy_to_host() const;
  int copy_to_host(double* out, std::size_t n) const;
  int copy_to_host(std::vector<double>& out) const;

  // move ctor
  cuda_buffer(cuda_buffer&& other) noexcept;

  // move-assign
  cuda_buffer& operator=(cuda_buffer&& other) noexcept;

  // accessors
  double* data() const noexcept { return d; }
  std::size_t size() const noexcept { return sz; }

  // swap for copy-and-swap and move
  void swap(cuda_buffer& o) noexcept {
    using std::swap;
    swap(d,  o.d);
    swap(sz, o.sz);
  }

};

