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
  LogLikelihood(std::vector<int> const& bin_counts,
                std::vector<double> const& bin_centers)
    : counts(bin_counts), centers(bin_centers)
  {}

  // Returns the sum of the negative log likelihood over all bins.
  template <typename T>
  T
  operator()(T a, T b, T c, T d) const
  {
    using std::lgamma; // may need to write this for CUDA
    using std::pow;    // may need to write this for CUDA
    double nll = 0.0;
    for (size_t i = 0; i < counts.size(); ++i) {
      double x = centers[i];
      double mean = a * pow(1.0 - x, b) / pow(x, c + d * log(x));
      int k = counts[i];
      // log Poisson probability: k*log(mean) - mean - log(k!)
      // log(k!) = lgamma(k+1)
      double log_p = k * log(mean) - mean - lgamma(k + 1);
      nll -= log_p;
    }
    return nll;
  }

private:
  std::vector<int> counts;
  std::vector<double> centers;
};
