#pragma once

#include <cassert>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

namespace dual {

  /// Representation of a dual number for automatic differentiation.
  struct DualNumber {
    double real; ///< Real part of the dual number
    double dual; ///< Dual part (derivative) of the dual number

    __host__ __device__ DualNumber(
      double real = 0.0,
      double dual =
        0.0); ///< Construct a dual number with given real and dual parts.

    __host__ __device__ DualNumber operator-() const; ///< Unary minus operator.

    __host__ __device__ DualNumber& operator+=(
      const DualNumber& rhs); ///< Compound addition assignment.

    __host__ __device__ DualNumber
    operator+(const DualNumber& rhs) const; ///< Binary addition operator.

    __host__ __device__ DualNumber
    operator-(const DualNumber& rhs) const; ///< Binary subtraction operator.

    __host__ __device__ DualNumber
    operator*(const DualNumber& rhs) const; ///< Binary multiplication operator.

    __host__ __device__ DualNumber
    operator/(const DualNumber& rhs) const; ///< Binary division operator.

    __host__ __device__ friend DualNumber operator-(
      double lhs,
      const DualNumber& rhs); ///< Subtraction of a dual number from a double.

    __host__ __device__ friend DualNumber operator*(
      double lhs,
      const DualNumber& rhs); ///< Multiplication of a double and a dual number.

    __host__ __device__ DualNumber& operator-=(
      const DualNumber& rhs); ///< Compound subtraction assignment.
  };

  // ──────────────────────────────────────────────────────────────────────────
  // Implementation
  // ──────────────────────────────────────────────────────────────────────────

  __host__ __device__ inline DualNumber::DualNumber(double real, double dual)
    : real(real), dual(dual)
  {}

  __host__ __device__ inline DualNumber
  DualNumber::operator-() const
  {
    return DualNumber(-real, -dual);
  }

  __host__ __device__ inline DualNumber&
  DualNumber::operator+=(const DualNumber& rhs)
  {
    real += rhs.real;
    dual += rhs.dual;
    return *this;
  }

  __host__ __device__ inline DualNumber
  DualNumber::operator+(const DualNumber& rhs) const
  {
    return DualNumber(real + rhs.real, dual + rhs.dual);
  }

  __host__ __device__ inline DualNumber
  DualNumber::operator-(const DualNumber& rhs) const
  {
    return DualNumber(real - rhs.real, dual - rhs.dual);
  }

  __host__ __device__ inline DualNumber
  DualNumber::operator*(const DualNumber& rhs) const
  {
    return DualNumber(real * rhs.real, dual * rhs.real + real * rhs.dual);
  }

  __host__ __device__ inline DualNumber
  DualNumber::operator/(const DualNumber& rhs) const
  {
    double denom = rhs.real * rhs.real;
    return DualNumber(real / rhs.real,
                      (dual * rhs.real - real * rhs.dual) / denom);
  }

  __host__ __device__ inline DualNumber
  operator-(double lhs, const DualNumber& rhs)
  {
    return DualNumber(lhs - rhs.real, -rhs.dual);
  }

  __host__ __device__ inline DualNumber
  operator*(double lhs, const DualNumber& rhs)
  {
    return DualNumber(lhs * rhs.real, lhs * rhs.dual);
  }

  __host__ __device__ inline DualNumber&
  DualNumber::operator-=(const DualNumber& rhs)
  {
    real -= rhs.real;
    dual -= rhs.dual;
    return *this;
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Mathematical functions
  // ──────────────────────────────────────────────────────────────────────────

  /// Absolute value of a dual number.
  __host__ __device__ inline dual::DualNumber
  abs(const dual::DualNumber& a)
  {
    return (a.real < 0.0) ? dual::DualNumber(-a.real, -a.dual) : a;
  }

  /// Sine of a dual number.
  static __inline__ __host__ __device__ DualNumber
  sin(const DualNumber& x)
  {
    return DualNumber(::sin(x.real), x.dual * ::cos(x.real));
  }

  /// Cosine of a dual number.
  static __inline__ __host__ __device__ DualNumber
  cos(const DualNumber& x)
  {
    return DualNumber(::cos(x.real), -x.dual * ::sin(x.real));
  }

  /// Exponential of a dual number.
  static __inline__ __host__ __device__ DualNumber
  exp(const DualNumber& x)
  {
    double ex = ::exp(x.real);
    return DualNumber(ex, x.dual * ex);
  }

  /// Square root of a dual number.
  static __inline__ __host__ __device__ DualNumber
  sqrt(const DualNumber& x)
  {
    double sr = ::sqrt(x.real);
    return DualNumber(sr, x.dual / (2.0 * sr));
  }

  /// Two-argument arctangent of dual numbers.
  static __inline__ __host__ __device__ DualNumber
  atan2(const DualNumber& y, const DualNumber& x)
  {
    double denom = x.real * x.real + y.real * y.real;
    return DualNumber(::atan2(y.real, x.real),
                      (x.real * y.dual - y.real * x.dual) / denom);
  }

  /// Natural logarithm of a dual number.
  static __inline__ __host__ __device__ DualNumber
  log(const DualNumber& x)
  {
    // assume x.real > 0
    return DualNumber(::log(x.real), x.dual / x.real);
  }

  /// Power of a dual number raised to another dual number.
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

  /// Power of a dual number raised to a double exponent.
  static __inline__ __host__ __device__ DualNumber
  pow(const DualNumber& base, double exponent)
  {
    const double br = base.real;
    const double pr = ::pow(br, exponent);
    if (br == 0.0) {
      return DualNumber(pr, exponent * ::pow(br, exponent - 1.0) * base.dual);
    }
    return DualNumber(pr, exponent * pr / br * base.dual);
  }

  /// Returns the value of pi.
  static constexpr __host__ __device__ double
  pi()
  {
    return 3.14159265358979323846;
  }

  /// Digamma function for a double.
  static __inline__ __host__ __device__ double
  digamma(double x)
  {
    // Digamma has simple poles at x = 0, -1, -2, ...
    // Handle these exact values explicitly so behavior is deterministic and
    // does not depend on tan(pi*x) roundoff in the reflection path.
    if (x <= 0.0 && x == ::floor(x)) {
      return std::numeric_limits<double>::infinity();
    }

    if (x < 0.5) {
      // digamma(x) = digamma(1-x) - pi cot(pi x)
      return digamma(1.0 - x) - pi() / ::tan(pi() * x);
    }
    double acc = 0.0;
    while (x < 8.0) {
      acc -= 1.0 / x;
      x += 1.0;
    }
    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    // digamma(x) ~ ln x − 1/(2x) − 1/(12x^2) + 1/(120x^4) − 1/(252x^6) +
    // 1/(240x^8) − 1/(132x^10) + 691/(32760x^12)
    const double s =
      inv2 *
      (-1.0 / 12.0 +
       inv2 *
         (1.0 / 120.0 +
          inv2 * (-1.0 / 252.0 +
                  inv2 * (1.0 / 240.0 +
                          inv2 * (-1.0 / 132.0 + inv2 * (691.0 / 32760.0))))));
    return acc + ::log(x) - 0.5 * inv + s;
  }

  /// Log-gamma function for a dual number.
  static __inline__ __host__ __device__ DualNumber
  lgamma(const DualNumber& x)
  {
    const double lr = ::lgamma(x.real);
    const double dg = digamma(x.real);
    return DualNumber(lr, x.dual * dg);
  }

  /// Calculate the gradient of a function using automatic differentiation.
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
