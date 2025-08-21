#pragma once

#include <cassert>
#include <array>

namespace dual {

  class DualNumber {
  public:
    double real;
    double dual;

    __host__ __device__
    DualNumber(double real = 0.0, double dual = 0.0)
      : real(real), dual(dual)
    {}

    // unary minus
    __host__ __device__ DualNumber
    operator-() const
    {
      return DualNumber(-real, -dual);
    }
    __host__ __device__ DualNumber&
    operator+=(const DualNumber& rhs)
    {
      real += rhs.real;
      dual += rhs.dual;
      return *this;
    }

    __host__ __device__ DualNumber
    operator+(const DualNumber& rhs) const
    {
      return DualNumber(real + rhs.real, dual + rhs.dual);
    }

    __host__ __device__ DualNumber
    operator-(const DualNumber& rhs) const
    {
      return DualNumber(real - rhs.real, dual - rhs.dual);
    }

    __host__ __device__ DualNumber
    operator*(const DualNumber& rhs) const
    {
      return DualNumber(real * rhs.real, dual * rhs.real + real * rhs.dual);
    }

    __host__ __device__ DualNumber
    operator/(const DualNumber& rhs) const
    {
      double denom = rhs.real * rhs.real;
      return DualNumber(real / rhs.real,
                        (dual * rhs.real - real * rhs.dual) / denom);
    }
    // operator for double - DualNumber
    __host__ __device__ friend DualNumber
    operator-(double lhs, const DualNumber& rhs)
    {
      return DualNumber(lhs - rhs.real, -rhs.dual);
    }

    // operator for double * DualNumber
    __host__ __device__ friend DualNumber
    operator*(double lhs, const DualNumber& rhs)
    {
      return DualNumber(lhs * rhs.real, lhs * rhs.dual);
    }
  };

  __host__ __device__ inline dual::DualNumber
  abs(const dual::DualNumber& a)
  {
    return (a.real < 0.0) ? dual::DualNumber(-a.real, -a.dual) : a;
  }

  static __inline__ __host__ __device__ DualNumber
  sin(const DualNumber& x)
  {
    return DualNumber(sinf(x.real), x.dual * cosf(x.real));
  }

  static __inline__ __host__ __device__ DualNumber
  cos(const DualNumber& x)
  {
    return DualNumber(cosf(x.real), -x.dual * sinf(x.real));
  }

  static __inline__ __host__ __device__ DualNumber
  exp(const DualNumber& x)
  {
    double ex = expf(x.real);
    return DualNumber(ex, x.dual * ex);
  }

  static __inline__ __host__ __device__ DualNumber
  sqrt(const DualNumber& x)
  {
    double sr = sqrtf(x.real);
    return DualNumber(sr, x.dual / (2.0 * sr));
  }

  static __inline__ __host__ __device__ DualNumber
  atan2(const DualNumber& y, const DualNumber& x)
  {
    double denom = x.real * x.real + y.real * y.real;
    return DualNumber(atan2f(y.real, x.real),
                      (x.real * y.dual - y.real * x.dual) / denom);
  }

  template <typename T>
  static __inline__ __host__ __device__ T
  pow(const T& base, double exponent)
  {
    return T(powf(base.real, exponent),
             exponent * powf(base.real, exponent - 1) * base.dual);
  }

  // can we call F with std::array<Scalar,DIM>?

  // only enabled if f takes std::array<dual,DIM> -> dual
  template <class Function,
            std::size_t DIM,
            class = std::enable_if_t<std::is_same_v<
              decltype(std::declval<Function>()(
                std::declval<std::array<dual::DualNumber, DIM>>())),
              dual::DualNumber>>>

  __device__ void
  calculateGradientUsingAD(
    Function const& f,
    const std::array<double, DIM>& x_arr, // input point
    std::array<double, DIM>& grad)        // output derivative vector
  {
    // build dual‐array on stack
    std::array<dual::DualNumber, DIM> xDual;
#pragma unroll
    for (std::size_t i = 0; i < DIM; ++i) {
      xDual[i].real = x_arr[i];
      xDual[i].dual = 0.0;
    }

    // partials
#pragma unroll
    for (std::size_t i = 0; i < DIM; ++i) {
      xDual[i].dual = 1.0;                // derivative w.r.t. dimension i
      dual::DualNumber result = f(xDual); // evaluate the function using AD
      grad[i] = result.dual;              // store derivative
      xDual[i].dual = 0.0;
    }
  }

} // end of dual
