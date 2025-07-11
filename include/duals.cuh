#pragma once

#include <type_traits>

namespace dual {

  class DualNumber {
  public:
    double real;
    double dual;

    __host__ __device__
    DualNumber(double real = 0.0, double dual = 0.0)
      : real(real), dual(dual)
    {}

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
  dual_abs(const dual::DualNumber& a)
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

  // compile-time trait: can we call F with Scalar*? double or DualNumber
  template <class F, class Scalar, class = void>
  struct is_callable_with : std::false_type {}; // default to false

  //
  template <class F, class Scalar>
  struct is_callable_with<
    F,
    Scalar, //
    std::void_t<decltype(std::declval<F>()(std::declval<const Scalar*>()))>>
    : std::true_type {};
  // void if success, else SFINAE

  template <class F, class Scalar>
  inline constexpr bool is_callable_with_v = is_callable_with<F, Scalar>::value;

  // enabled only if Function can be instantiated with DualNumber
  template <class Function,
            int DIM,
            class = std::enable_if_t<is_callable_with_v<Function, DualNumber>>>
  __device__ void
  calculateGradientUsingAD(Function f, double const* x, double* gradient)
  {
    dual::DualNumber xDual[DIM];

    for (int i = 0; i < DIM;
         ++i) { // // iterate through each dimension (vairbale)
      xDual[i] = dual::DualNumber(x[i], 0.0);
    }

    // calculate the partial derivative of  each dimension
    for (int i = 0; i < DIM; ++i) {
      xDual[i].dual = 1.0;                // derivative w.r.t. dimension i
      dual::DualNumber result = f(xDual); // evaluate the function using AD
      gradient[i] = result.dual;          // store derivative
      // printf("\nxDual[%d]: %f, grad[%d]: %f ",i,xDual[i].real,i,gradient[i]);
      xDual[i].dual = 0.0;
    }
  } // end calculate gradientusingAD

} // end of dual
