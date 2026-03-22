#pragma once
#include "duals.cuh"

namespace util {

  template <int DIM>
  __device__ dual::DualNumber
  rastrigin(const dual::DualNumber* x)
  {
    dual::DualNumber sum(10.0 * DIM, 0.0);
    for (int i = 0; i < DIM; ++i) {
      sum = sum + (x[i] * x[i] -
                   dual::DualNumber(10.0, 0.0) *
                     dual::cos(x[i] * dual::DualNumber(2.0 * M_PI, 0.0)));
    }
    return sum;
  }

  template <int DIM>
  __host__ __device__ double
  rastrigin(const double* x)
  {
    double sum = 10.0 * DIM;
    for (int i = 0; i < DIM; ++i)
      sum += x[i] * x[i] - 10.0 * std::cos(2.0 * M_PI * x[i]);
    return sum;
  }

  template <std::size_t D>
  struct Rastrigin {
    static constexpr std::size_t arity = D;
    template <typename T>
    __host__ __device__ T
    operator()(std::array<T, D> const& x) const
    {
      return rastrigin<D>(x.data());
    }
  };

} // namespace util
