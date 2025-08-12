#pragma once

#include <array>
#include <cuda_runtime.h>

#include "matrix.cuh"

template <std::size_t N>
class Gaussian {
  Matrix<double> C;

public:
  // host constructor copy the covariance matrix into device memory
  Gaussian(std::array<std::array<double,N>,N> const& C_host) : C(N,N)
  {
    // fill the host buffer
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < N; ++j)
        C(i,j) = C_host[i][j];
    
    // push buffer to the GPU
    C.syncHost2Device();
  }

  // device & host call-operator
  template <class T>
  __host__ __device__
  T operator()(std::array<T,N> const& x) const {
    T q = T(0);
    #pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      #pragma unroll
      for (std::size_t j = 0; j < N; ++j) {
        // if cuda_arch
            auto a = C(i,j);
	    //.data()[i*N + j];
            q += x[i] * T(a * x[j]);
        // else
        //  use host matrix
      }
    }
    return T(0.5) * q;
  }
};

