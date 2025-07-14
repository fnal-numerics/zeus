#pragma once
#include <cassert>
#include "duals.cuh"

namespace util {

  extern "C" {
  __device__ __noinline__ void vector_add(const double*,
                                          const double*,
                                          double*,
                                          int);
  __device__ __noinline__ void vector_scale(const double*,
                                            double,
                                            double*,
                                            int);
  }

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

  template <int DIM>
  struct GoldsteinPrice {
    __device__ static dual::DualNumber
    evaluate(const dual::DualNumber* x)
    {
      return goldstein_price<DIM>(x);
    }
    __host__ __device__ static double
    evaluate(const double* x)
    {
      return goldstein_price<DIM>(x);
    }
  };

  // Eggholder Function
  //   f(x,y) = -(y+47) sin\Bigl(\sqrt{\Bigl|x/2+y+47\Bigr|}\Bigr)
  //            - x sin\Bigl(\sqrt{\Bigl|x-(y+47)\Bigr|}\Bigr)

  template <int DIM>
  __device__ dual::DualNumber
  eggholder(const dual::DualNumber* x)
  {
    static_assert(DIM == 2, "Eggholder is defined for 2 dimensions only.");
    dual::DualNumber x1 = x[0], x2 = x[1];
    // Use (0 - value) in place of unary minus
    dual::DualNumber term1 =
      (dual::DualNumber(0.0) - (x2 + dual::DualNumber(47.0))) *
      dual::sin(dual::sqrt(
        dual_abs(x1 / dual::DualNumber(2.0) + x2 + dual::DualNumber(47.0))));
    dual::DualNumber term2 =
      (dual::DualNumber(0.0) - x1) *
      dual::sin(dual::sqrt(dual_abs(x1 - (x2 + dual::DualNumber(47.0)))));
    return term1 + term2;
  }

  template <int DIM>
  __device__ double
  eggholder(const double* x)
  {
    static_assert(DIM == 2, "Eggholder is defined for 2 dimensions only.");
    double x1 = x[0];
    double x2 = x[1];
    double term1 = -(x2 + 47.0) * sin(sqrt(fabs(x1 / 2.0 + x2 + 47.0)));
    double term2 = -x1 * sin(sqrt(fabs(x1 - (x2 + 47.0))));
    return term1 + term2;
  }

  template <int DIM>
  struct Eggholder {
    __host__ __device__ static dual::DualNumber
    evaluate(const dual::DualNumber* x)
    {
      return eggholder<DIM>(x);
    }
    __host__ __device__ static double
    evaluate(const double* x)
    {
      return eggholder<DIM>(x);
    }
  };

template<int D>
struct Rosenbrock {
  template<class T, std::size_t N, class = std::enable_if_t<N == D>>
  __host__ __device__
  T operator()(const std::array<T,N>& x) const {
    return rosenbrock<D>(x.data());
  }
};


template<int D>
struct Rastrigin {
  template<class T, std::size_t N, class = std::enable_if_t<N == D>>
  __host__ __device__
  T operator()(const std::array<T,N>& x) const {
    return rastrigin<D>(x.data());
  }
};

template<int D>
struct Ackley {
  template<class T, std::size_t N,
           class = std::enable_if_t<N == D>>
  __host__ __device__
  T operator()(const std::array<T,N>& x) const {
    return ackley<D>(x.data());
  }
};


  
  template <int DIM>
  struct Himmelblau {
    __host__ __device__ static dual::DualNumber
    evaluate(const dual::DualNumber* x)
    {
      return himmelblau<DIM>(x);
    }
    __host__ __device__ static double
    evaluate(const double* x)
    {
      return himmelblau<DIM>(x);
    }
  };

} // namespace util
