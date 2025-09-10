import numpy as np
from scipy.integrate import quad
from scipy.stats import poisson


def cms_dijet_mass(mjj, p0, p1, p2, p3, sqrt_s=13600):
    """
    CMS dijet mass spectrum functional form: dsigma/dmjj = p0 * (1 - x)^p1 / x^(p2 + p3 * ln(x))
    From https://arxiv.org/pdf/1911.03947
    mjj: dijet invariant mass (GeV)
    p0, p1, p2, p3: fit parameters
    sqrt_s: center-of-mass energy (GeV, default 13.6 TeV)
    Returns: differential cross-section (arbitrary units)
    """
    x = mjj / sqrt_s
    # Avoid numerical issues with x near 0 or 1
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return p0 * np.power(1 - x, p1) / np.power(x, p2 + p3 * np.log(x))


def integrate_cross_section(mjj_low, mjj_high, p0, p1, p2, p3, sqrt_s):
    """
    Integrate the differential cross-section over a bin [mjj_low, mjj_high].
    Returns: integrated cross-section (arbitrary units)
    """
    result, _ = quad(cms_dijet_mass, mjj_low, mjj_high, args=(p0, p1, p2, p3, sqrt_s))
    return result


def simulate_dijet_spectrum(
    p0=1e4,
    p1=7.0,
    p2=4.5,
    p3=0.05,
    sqrt_s=13.600,
    mjj_min=2.000,
    mjj_max=10.000,
    bin_width=0.200,
    luminosity=137,
    seed=42,
):
    """
    Simulate a binned dijet mass spectrum with Poisson-distributed counts.
    Parameters:
        p0, p1, p2, p3: CMS fit parameters
        sqrt_s: center-of-mass energy (GeV)
        mjj_min, mjj_max: mass range for the spectrum (GeV)
        bin_width: width of bins (GeV)
        luminosity: integrated luminosity (fb^-1)
        seed: random seed for reproducibility
    Returns:
        bins: bin edges
        counts: Poisson-distributed event counts per bin
    """
    # Set random seed
    np.random.seed(seed)

    # Define bins
    bins = np.arange(mjj_min, mjj_max + bin_width, bin_width)

    # Integrate cross-section over each bin to get expected events
    expected_counts = np.zeros(len(bins) - 1)
    for i, (mjj_low, mjj_high) in enumerate(zip(bins[:-1], bins[1:])):
        # Integrate dsigma/dmjj over the bin
        sigma_bin = integrate_cross_section(mjj_low, mjj_high, p0, p1, p2, p3, sqrt_s)
        # Convert to expected events: N = sigma * luminosity (fb^-1 to pb^-1)
        expected_counts[i] = sigma_bin * luminosity * 1e3
        # Warn if expected counts are too large
        if expected_counts[i] > 1e9:
            print(
                f"Warning: Expected counts in bin {i} ({mjj_low}-{mjj_high} GeV) = {expected_counts[i]:.2e}, may be too large for poisson.rvs"
            )

    # Generate Poisson-distributed counts
    counts = poisson.rvs(expected_counts)

    return bins, counts


if __name__ == "__main__":

    params = {
        "p0": 0.1,  # Reduced normalization for lower counts
        "p1": 7.0,  # High-mass shape
        "p2": 4.5,  # Power-law fall-off
        "p3": 0.05,  # Logarithmic correction
        "sqrt_s": 13.600,  # Center-of-mass energy (GeV, 13.6 TeV)
        "mjj_min": 2.000,  # Minimum dijet mass (GeV)
        "mjj_max": 10.000,  # Maximum dijet mass (GeV)
        "bin_width": 0.200,  # Bin width (GeV)
        "luminosity": 137,  # Integrated luminosity (fb^-1)
        "seed": 42,  # Random seed
    }

    # Simulate the spectrum
    bins, counts = simulate_dijet_spectrum(**params)

    # Write to TSV file
    output_file = "dijet_spectrum.tsv"
    with open(output_file, "w") as f:
        f.write("bin_low\tbin_high\tcounts\n")
        for low, high, count in zip(bins[:-1], bins[1:], counts):
            f.write(f"{low}\t{high}\t{count}\n")
