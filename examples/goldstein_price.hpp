#pragma once
#include "duals.cuh"

namespace util {

  // Goldstein-Price Function
  //   f(x,y) = [1+(x+y+1)^2 (19-14x+3x^2-14y+6xy+3y^2)]
  //            [30+(2x-3y)^2 (18-32x+12x^2+48y-36xy+27y^2)]
  template <int DIM>
  __device__ dual::DualNumber
  goldstein_price(const dual::DualNumber* x)
  {
    static_assert(DIM == 2,
                  "Goldstein-Price is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0];
    dual::DualNumber x2 = x[1];
    dual::DualNumber term1 =
      dual::DualNumber(1.0) +
      dual::pow(x1 + x2 + 1.0, 2) *
        (19.0 - 14.0 * x1 + 3.0 * dual::pow(x1, 2) - 14.0 * x2 + 6.0 * x1 * x2 +
         3.0 * dual::pow(x2, 2));
    dual::DualNumber term2 =
      dual::DualNumber(30.0) +
      dual::pow(2.0 * x1 - 3.0 * x2, 2) *
        (18.0 - 32.0 * x1 + 12.0 * dual::pow(x1, 2) + 48.0 * x2 -
         36.0 * x1 * x2 + 27.0 * dual::pow(x2, 2));
    return term1 * term2;
  }

  template <int DIM>
  __host__ __device__ double
  goldstein_price(const double* x)
  {
    static_assert(DIM == 2,
                  "Goldstein-Price is defined for 2 dimensions only.");
    double x1 = x[0];
    double x2 = x[1];
    double term1 = 1.0 + pow(x1 + x2 + 1.0, 2) *
                           (19.0 - 14.0 * x1 + 3.0 * pow(x1, 2) - 14.0 * x2 +
                             6.0 * x1 * x2 + 3.0 * pow(x2, 2));
    double term2 = 30.0 + pow(2.0 * x1 - 3.0 * x2, 2) *
                           (18.0 - 32.0 * x1 + 12.0 * pow(x1, 2) + 48.0 * x2 -
                            36.0 * x1 * x2 + 27.0 * pow(x2, 2));
    return term1 * term2;
  }

  template <std::size_t D>
  struct GoldsteinPrice {
    static constexpr std::size_t arity = D;
    template<typename T>
    __host__ __device__
    T operator()(std::array<T,D> const& x) const
    {
      return goldstein_price<D>(x.data());
    }
  };

} // namespace util
