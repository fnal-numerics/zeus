#pragma once

#include <array>
#include <cuda_runtime.h>

#include "matrix.hpp"

template <std::size_t N>
class Gaussian {
  Matrix<double> C;
public:
  explicit Gaussian(const std::array<std::array<double,N>,N>& C_host) : C(N,N) {
    C.set(&C_host[0][0], N*N); // one bulk copy + one cudaMemcpy
  }

  template <class T>
  __host__ __device__
  T operator()(const std::array<T,N>& x) const {
    const double* Cp = C.data();
    T q = T(0);
    #pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      #pragma unroll
      for (std::size_t j = 0; j < N; ++j) {
        q += x[i] * T(Cp[i*N + j]) * x[j];
      }
    }
    printf("end of function, q=%g\n", T(q));
    return T(0.5) * q;
  }
};

