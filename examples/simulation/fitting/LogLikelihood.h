#pragma once
#include <cmath>
#include <vector>

// LogLikelihood calculates the product of Poisson probabilities for a vector of
// bins. The constructor takes:
//   - bin_counts: vector<int> with the count of elements in each bin
//   - bin_centers: vector<double> with the value of x at the center of each bin
// The Poisson mean in bin i is: a * (1-x)^b / (x^(c + d*ln(x)))
class LogLikelihood {
public:
  static constexpr std::size_t arity = 4;
  LogLikelihood(std::vector<int> const& bin_counts,
                std::vector<double> const& bin_centers)
    : counts(bin_counts), centers(bin_centers),
      n_(counts.size()), K_(n_,1), X_(n_,1)
  {
    // one bulk host->device copy each (same pattern as Gaussian)
    K_.set(counts.data(),  n_);
    X_.set(centers.data(), n_);
  }

  // Zeus-compatible overload
  template <typename T>
  __host__ __device__
  T operator()(std::array<T,4> const& theta) const {
    using ::pow;     // device intrinsics for double; ADL picks dual::pow for DualNumber
    using ::log;     // same
    using ::lgamma;  // device lgamma(double); ADL picks dual::lgamma for DualNumber

    const T a = theta[0];
    const T b = theta[1];
    const T c = theta[2];
    const T d = theta[3];

    // read from device-visible mirrors (no std::vector in device code)
    const int*    Kp = K_.data();
    const double* Xp = X_.data();

    T nll = T(0);
    for (std::size_t i = 0; i < n_; ++i) {
      T x = T(Xp[i]);         // promote to T
      int ki = Kp[i];

      T mean   = a * pow(T(1.0) - x, b) / pow(x, c + d * log(x));
      T log_p  = T(ki) * log(mean) - mean - lgamma(T(ki) + T(1));
      nll -= log_p;
    }
    return nll;
  }

  template <typename T>
  __host__ __device__
  T operator()(T a, T b, T c, T d) const {
    return (*this)(std::array<T,4>{a,b,c,d});
  }

private:
  std::vector<int>    counts;
  std::vector<double> centers;

  // device-visible mirrors 
  std::size_t    n_;
  Matrix<int>    K_;
  Matrix<double> X_;
};

namespace zeus {
template<> struct FnTraits<LogLikelihood> {
  static constexpr std::size_t arity = 4;
};
}
