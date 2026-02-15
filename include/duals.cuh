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

    __host__ __device__ DualNumber&
    operator-=(const DualNumber& rhs)
    {
      real -= rhs.real;
      dual -= rhs.dual;
      return *this;
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
    return DualNumber(::sin(x.real), x.dual * ::cos(x.real));
  }

  static __inline__ __host__ __device__ DualNumber
  cos(const DualNumber& x)
  {
    return DualNumber(::cos(x.real), -x.dual * ::sin(x.real));
  }

  static __inline__ __host__ __device__ DualNumber
  exp(const DualNumber& x)
  {
    double ex = ::exp(x.real);
    return DualNumber(ex, x.dual * ex);
  }

  static __inline__ __host__ __device__ DualNumber
  sqrt(const DualNumber& x)
  {
    double sr = ::sqrt(x.real);
    return DualNumber(sr, x.dual / (2.0 * sr));
  }

  static __inline__ __host__ __device__ DualNumber
  atan2(const DualNumber& y, const DualNumber& x)
  {
    double denom = x.real * x.real + y.real * y.real;
    return DualNumber(::atan2(y.real, x.real),
                      (x.real * y.dual - y.real * x.dual) / denom);
  }
  // log for DualNumber: (ln r, r'/r)
  static __inline__ __host__ __device__ DualNumber
  log(const DualNumber& x)
  {
    // assume x.real > 0
    return DualNumber(::log(x.real), x.dual / x.real);
  }

  // pow for DualNumber ^ DualNumber
  static __inline__ __host__ __device__ DualNumber
  pow(const DualNumber& base, const DualNumber& exponent)
  {
    // base.real must be > 0
    const double br = base.real;
    const double er = exponent.real;
    const double pr = ::pow(br, er);
    // d(b^e) = b^e * (e' * ln b + e * b'/b)
    const double pd = pr * (exponent.dual * ::log(br) + er * base.dual / br);
    return DualNumber(pr, pd);
  }

  template <typename T>
  static __inline__ __host__ __device__ T
  pow(const T& base, double exponent)
  {
    return T(::pow(base.real, exponent),
             exponent * ::pow(base.real, exponent - 1) * base.dual);
  }

  // pi via std::acos(-1)
  static __inline__ __host__ __device__ double
  pi()
  {
    return ::acos(-1.0);
  }

  // digamma(double) helper for lgamma
  // Reflection for x<0.5, then recur to x>=8, then asymptotic series.
  static __inline__ __host__ __device__ double
  digamma(double x)
  {
    if (x < 0.5) {
      // gamma(x) = gamma(1-x) - pi cot(pi x)
      return digamma(1.0 - x) - pi() / ::tan(pi() * x);
    }
    double acc = 0.0;
    while (x < 8.0) {
      acc -= 1.0 / x;
      x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    // gamma(x) ~ ln x − 1/(2x) − 1/(12x^2) + 1/(120x^4) − 1/(252x^6) +
    // 1/(240x^8) − 1/(132x^10)
    const double s =
      -1.0 / 12.0 +
      inv2 *
        (1.0 / 120.0 +
         inv2 * (-1.0 / 252.0 + inv2 * (1.0 / 240.0 + inv2 * (-1.0 / 132.0))));
    return acc + ::log(x) - 0.5 * inv + s;
  }

  // lgamma for DualNumber: (lgamma(r), r' * digamma(r))
  static __inline__ __host__ __device__ DualNumber
  lgamma(const DualNumber& x)
  {
    const double lr = ::lgamma(x.real);
    const double dg = digamma(x.real);
    return DualNumber(lr, x.dual * dg);
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
