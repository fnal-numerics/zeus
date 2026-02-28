#include "duals.cuh"
#include <cmath>
#include <limits>

namespace dual {

  /// Digamma function for a double.
  /// Uses the Stirling asymptotic series with reflection and argument
  /// reduction.
  __host__ __device__ double
  digamma(double x)
  {
    // Digamma has simple poles at x = 0, -1, -2, ...
    if (x <= 0.0 && x == ::floor(x))
      return std::numeric_limits<double>::infinity();

    if (x < 0.5)
      // Reflection: digamma(x) = digamma(1-x) - pi*cot(pi*x)
      return digamma(1.0 - x) - pi() / ::tan(pi() * x);

    double acc = 0.0;
    while (x < 8.0) {
      acc -= 1.0 / x;
      x += 1.0;
    }

    const double inv = 1.0 / x;
    const double inv2 = inv * inv;
    // Asymptotic series: digamma(x) ~ ln x - 1/(2x) - sum B_{2k}/(2k * x^{2k})
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

} // namespace dual
