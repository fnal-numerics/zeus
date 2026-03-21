#pragma once
#include "duals.cuh"

namespace util {

  // Himmelblau's Function (2D only)
  template <int DIM>
  __device__ dual::DualNumber
  himmelblau(const dual::DualNumber* x)
  {
    static_assert(DIM == 2,
                  "Himmelblau's function is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0], x2 = x[1];
    dual::DualNumber term1 =
      dual::pow(x1 * x1 + x2 - dual::DualNumber(11.0), 2);
    dual::DualNumber term2 = dual::pow(x1 + x2 * x2 - dual::DualNumber(7.0), 2);
    return term1 + term2;
  }

  template <int DIM>
  __host__ __device__ double
  himmelblau(const double* x)
  {
    static_assert(DIM == 2,
                  "Himmelblau's function is defined for 2 dimensions only.");
    double x1 = x[0], x2 = x[1];
    double term1 = pow(x1 * x1 + x2 - 11.0, 2);
    double term2 = pow(x1 + x2 * x2 - 7.0, 2);
    return term1 + term2;
  }

  template <std::size_t D>
  struct Himmelblau {
    static constexpr std::size_t arity = D;
    template<typename T>
    __host__ __device__ 
    T operator()(std::array<T,D> const& x) const {   
      return himmelblau<D>(x.data());
    }
  };

} // namespace util
