#pragma once
#include "duals.cuh"

namespace util {

  template <int DIM>
  __device__ dual::DualNumber
  rosenbrock(const dual::DualNumber* x)
  {
    dual::DualNumber sum(0.0, 0.0);
    for (int i = 0; i < DIM - 1; ++i) {
      dual::DualNumber t1 = dual::DualNumber(1.0, 0.0) - x[i];
      dual::DualNumber t2 = x[i + 1] - x[i] * x[i];
      sum = sum + t1 * t1 + dual::DualNumber(100.0, 0.0) * t2 * t2;
    }
    return sum;
  }

  template <int DIM>
  __host__ __device__ double
  rosenbrock(const double* x)
  {
    double sum = 0.0;
    for (int i = 0; i < DIM - 1; ++i) {
      double t1 = 1.0 - x[i];
      double t2 = x[i + 1] - x[i] * x[i];
      sum += t1 * t1 + 100.0 * t2 * t2;
    }
    return sum;
  }

  template <std::size_t D>
  struct Rosenbrock {
    static constexpr std::size_t arity = D;
    template <typename T>
    __host__ __device__ T
    operator()(const std::array<T, D>& x) const
    {
      return rosenbrock<D>(x.data());
    }
  };

} // namespace util
