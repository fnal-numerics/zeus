#include <vector>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "zeus.cuh"
#include "simulation/fitting/LogLikelihood.h"

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Load a three-column TSV (bin_low  bin_high  counts) and convert each bin
/// centre to the dimensionless variable x = m_centre / sqrt_s_units.
static void
load_dijet_spectrum_as_x(const std::string& path,
                         std::vector<double>& bin_low,
                         std::vector<double>& bin_high,
                         std::vector<int>& true_counts,
                         std::vector<double>& x_centers,
                         double sqrt_s_units)
{
  bin_low.clear();
  bin_high.clear();
  true_counts.clear();
  x_centers.clear();

  std::ifstream in(path);
  if (!in)
    throw std::runtime_error("couldn't open: " + path);

  std::string header;
  std::getline(in, header); // skip header line

  double low, high;
  int cnt;
  while (in >> low >> high >> cnt) {
    const double m_center = 0.5 * (low + high);
    double x = m_center / sqrt_s_units;
    constexpr double eps = 1e-12;
    if (x <= eps)
      x = eps;
    if (x >= 1.0 - eps)
      x = 1.0 - eps;

    bin_low.push_back(low);
    bin_high.push_back(high);
    true_counts.push_back(cnt);
    x_centers.push_back(x);
  }

  if (true_counts.empty() || true_counts.size() != x_centers.size())
    throw std::runtime_error("empty or mismatched data in: " + path);

  auto mn = *std::min_element(x_centers.begin(), x_centers.end());
  auto mx = *std::max_element(x_centers.begin(), x_centers.end());
  std::cout << "Loaded " << true_counts.size() << " bins; x in [" << mn << ", "
            << mx << "]\n";
}

/// Write per-bin predicted vs. observed counts to a TSV file.
static void
write_pred_vs_true_tsv(const std::string& path,
                       const std::vector<double>& bin_low,
                       const std::vector<double>& bin_high,
                       const std::vector<int>& true_counts,
                       const std::vector<double>& x_centers,
                       const std::vector<double>& pred_mu,
                       const std::vector<double>& pred_residual,
                       const std::vector<double>& pred_pull)
{
  std::ofstream out(path);
  if (!out)
    throw std::runtime_error("could not open: " + path);
  out << std::setprecision(17);
  out << "bin_low\tbin_high\tx_center\ttrue_counts\tpred_mu\t"
         "pred_residual\tpred_pull\n";
  for (size_t i = 0; i < true_counts.size(); ++i) {
    out << bin_low[i] << '\t' << bin_high[i] << '\t' << x_centers[i] << '\t'
        << true_counts[i] << '\t' << pred_mu[i] << '\t' << pred_residual[i]
        << '\t' << pred_pull[i] << '\n';
  }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
  if (argc < 4) {
    std::cerr
      << "Usage: " << argv[0]
      << " <num_optimizations> <max_bfgs_iter> <run_id> "
         "[--parallel] [--save-trajectories <filename>] [--prng <xorwow|philox|sobol>]"
         " [--nzerosteps <n>]\n";
    return 1;
  }
  const size_t N = std::stoul(argv[1]);
  const int bfgs = std::stoi(argv[2]);
  const int run = std::stoi(argv[3]);

  std::string trajectory_file;
  zeus::PRNGType prng_type = zeus::PRNGType::XORWOW;
  bool parallel = false;
  int nzerosteps = 0;
  for (int i = 4; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--parallel") {
      parallel = true;
    } else if (arg == "--save-trajectories" && i + 1 < argc) {
      trajectory_file = argv[++i];
    } else if (arg == "--prng" && i + 1 < argc) {
      std::string val = argv[++i];
      if (val == "xorwow")
        prng_type = zeus::PRNGType::XORWOW;
      else if (val == "philox")
        prng_type = zeus::PRNGType::PHILOX;
      else if (val == "sobol")
        prng_type = zeus::PRNGType::SOBOL;
      else {
        std::cerr << "Unknown PRNG type: " << val
                  << ". Using default xorwow.\n";
      }
    } else if (arg == "--nzerosteps" && i + 1 < argc) {
      nzerosteps = std::stoi(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << "\n";
      return 1;
    }
  }

  util::setStackSize();

  // --- load data -----------------------------------------------------------
  std::vector<double> bin_low, bin_high;
  std::vector<int> counts;
  std::vector<double> centers;

  load_dijet_spectrum_as_x("../examples/simulation/dijet_spectrum.tsv",
                           bin_low,
                           bin_high,
                           counts,
                           centers,
                           /*sqrt_s in TeV*/ 13.6);

  std::cout << "\n#bins = " << counts.size() << "\n";
  long long total = 0;
  for (int k : counts)
    total += k;
  std::cout << "total events in the spectrum = " << total << "\n\n";

  // --- fit -----------------------------------------------------------------
  // Model: mu_i = a * (1 - x_i)^b / x_i^(c + d*ln(x_i))
  // Parameters: a (normalisation), b, c, d (shape)
  // Search range [0, 10] covers all physically reasonable values.
  LogLikelihood ll(counts, centers);
  auto res = zeus::Zeus(ll,
                        0.00,
                        10.00,
                        N,
                        bfgs,
                        10,
                        100,
                        "poisson",
                        1e-8,
                        42,
                        run,
                        parallel,
                        prng_type,
                        trajectory_file,
                        nzerosteps);

  std::cout << "best NLL: " << res.fval << "\n";

  const double a = res.coordinates[0];
  const double b = res.coordinates[1];
  const double c = res.coordinates[2];
  const double d = res.coordinates[3];
  std::cout << "best-fit parameters: a=" << a << "  b=" << b << "  c=" << c
            << "  d=" << d << "\n\n";

  // --- post-fit diagnostics ------------------------------------------------
  constexpr double eps = 1e-12;
  std::vector<double> pred_mu(centers.size());
  std::vector<double> pred_residual(centers.size());
  std::vector<double> pred_pull(centers.size());

  for (size_t i = 0; i < centers.size(); ++i) {
    double x = std::min(std::max(centers[i], eps), 1.0 - eps);
    double mu = a * std::pow(1.0 - x, b) / std::pow(x, c + d * std::log(x));
    pred_mu[i] = mu;
    pred_residual[i] = static_cast<double>(counts[i]) - mu;
    pred_pull[i] = pred_residual[i] / std::sqrt(std::max(mu, eps));
  }

  double pred_chi2 = 0.0, pred_dev = 0.0;
  for (size_t i = 0; i < pred_mu.size(); ++i) {
    const double k = static_cast<double>(counts[i]);
    const double m = std::max(pred_mu[i], eps);
    pred_chi2 += (k - m) * (k - m) / m;
    pred_dev += (k > 0) ? 2.0 * (k * std::log(k / m) - (k - m)) : 2.0 * m;
  }
  const int ndf = std::max<int>(1, static_cast<int>(pred_mu.size()) - 4);
  std::cout << "chi2/ndf=" << pred_chi2 / ndf << "  dev/ndf=" << pred_dev / ndf
            << "\n";

  write_pred_vs_true_tsv("dijet_fit_vs_data.tsv",
                         bin_low,
                         bin_high,
                         counts,
                         centers,
                         pred_mu,
                         pred_residual,
                         pred_pull);
  std::cout << "predictions written to dijet_fit_vs_data.tsv\n";

  return 0;
}
