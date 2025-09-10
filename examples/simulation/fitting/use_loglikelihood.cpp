#include "LogLikelihood.h"
#include <iostream>
#include <vector>

int
main()
{
  std::vector<int> counts = {3, 2, 5};
  std::vector<double> centers = {0.2, 0.5, 0.8};
  LogLikelihood ll(counts, centers);
  double nll = ll(1.0, 2.0, 3.0, 4.0);
  std::cout << "Negative log likelihood: " << nll << std::endl;
}
