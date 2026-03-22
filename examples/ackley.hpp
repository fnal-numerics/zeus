#pragma once
#include "duals.cuh"

namespace util {

  // Ackley Function (general d-dimensions)
  //   f(x) = -20 exp\Bigl(-0.2\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2}\Bigr)
  //          - exp\Bigl(\frac{1}{d}\sum_{i=1}^{d}\cos(2\pi x_i)\Bigr)
  //          + 20 + e
  template <int DIM>
  __device__ dual::DualNumber
  ackley(const dual::DualNumber* x)
  {
    dual::DualNumber sum_sq = 0.0;
    dual::DualNumber sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
      sum_sq += dual::pow(x[i], 2);
      sum_cos += dual::cos(2.0 * M_PI * x[i]);
    }
    dual::DualNumber term1 =
      dual::DualNumber(-20.0) * dual::exp(-0.2 * dual::sqrt(sum_sq / DIM));
    dual::DualNumber term2 = dual::DualNumber(0.0) - dual::exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + dual::exp(1.0);
  }

  template <int DIM>
  __host__ __device__ double
  ackley(const double* x)
  {
    double sum_sq = 0.0;
    double sum_cos = 0.0;
    for (int i = 0; i < DIM; ++i) {
      sum_sq += x[i] * x[i];
      sum_cos += cos(2.0 * M_PI * x[i]);
    }
    double term1 = -20.0 * exp(-0.2 * sqrt(sum_sq / DIM));
    double term2 = -exp(sum_cos / DIM);
    return term1 + term2 + 20.0 + exp(1.0);
  }

  template <std::size_t D>
  struct Ackley {
    static constexpr std::size_t arity = D;
    template <typename T>
    __host__ __device__ T
    operator()(std::array<T, D> const& x) const
    {
      return ackley<D>(x.data());
    }
  };

} // namespace util
