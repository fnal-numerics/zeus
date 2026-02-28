# Zeus Benchmark Examples

This directory contains five stand-alone benchmark programs that exercise the
`zeus::Zeus` hybrid PSO+BFGS optimizer on standard mathematical test functions.
They serve as both integration tests and performance characterization harnesses:
run them with different hyperparameter combinations (swarm size, iteration
budget, tolerance, etc.) to measure convergence rate and solution quality.

---

## Common Command-Line Interface

Every program accepts exactly the same nine positional arguments:

```
<executable> <lower_bound> <upper_bound> <max_iter> <pso_iters> \
             <required_converged> <num_optimizations> <tolerance> <seed> <run_id>
```

| Position | Argument | Type | Description |
|----------|----------|------|-------------|
| 1 | `lower_bound` | `double` | Lower bound applied to every dimension of the search space |
| 2 | `upper_bound` | `double` | Upper bound applied to every dimension of the search space |
| 3 | `max_iter` | `int` | Maximum number of BFGS iterations per optimization thread |
| 4 | `pso_iters` | `int` | Number of PSO swarm iterations used to seed the BFGS starts |
| 5 | `required_converged` | `int` | Number of independent BFGS runs that must agree before Zeus declares convergence |
| 6 | `num_optimizations` | `int` | Total number of parallel optimization threads launched on the GPU |
| 7 | `tolerance` | `double` | Gradient-norm convergence threshold |
| 8 | `seed` | `int` | RNG seed for reproducible swarm initialization |
| 9 | `run_id` | `int` | Integer label written to output, useful when sweeping over multiple runs |

**Example:**

```bash
./optimize_rosenbrock -5.0 5.0 10000 20 100 1024 1e-8 42 0
```

---

## Common Output Fields

All programs print the same result block:

| Field | Description |
|-------|-------------|
| `Status` | Integer convergence code (see table below) |
| `Function value` | Best objective value found |
| `Gradient norm` | Euclidean norm of the gradient at the best point |
| `Iterations` | Number of BFGS iterations consumed |
| `Optimization time` | Wall-clock time in milliseconds |
| `Best coordinates` | The coordinates of the best-found minimum |

### Status Codes

| Code | Name | Meaning |
|------|------|---------|
| 0 | Surrendered | Maximum iterations reached without satisfying the convergence criterion |
| 1 | Converged | Gradient norm is below tolerance and `required_converged` runs agree |
| 2 | Stopped by flag | Run halted early because enough other threads already converged |
| 3 | CUDA malloc failure | GPU memory allocation failed |
| 4 | CUDA runtime error | A CUDA kernel launch or runtime call failed |

---

## Test Functions

Each function is chosen to stress a different aspect of the optimizer.

### Rosenbrock (`optimize_rosenbrock`)

$$f(\mathbf{x}) = \sum_{i=1}^{D-1} \left[ (1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2 \right]$$

- **Global minimum:** $f(\mathbf{1}) = 0$
- **Typical bounds:** $[-5, 5]^D$

The Rosenbrock "banana" function has a single global minimum that sits inside a
long, narrow, curved valley. The gradient along the valley floor is very
shallow, so gradient-based methods must take many small steps and are prone to
slow convergence. It is the canonical test for a BFGS line-search
implementation's ability to follow a curved ridge without stalling.

---

### Rastrigin (`optimize_rastrigin`)

$$f(\mathbf{x}) = 10D + \sum_{i=1}^{D} \left[ x_i^2 - 10 \cos(2\pi x_i) \right]$$

- **Global minimum:** $f(\mathbf{0}) = 0$
- **Typical bounds:** $[-5.12, 5.12]^D$

Rastrigin is highly multimodal: it adds a cosine lattice of local minima on top
of a parabolic bowl. The density of local traps grows with dimension,
making it a demanding test for the PSO phase's ability to locate a good basin
of attraction before the BFGS refinement begins. Strong performance here
indicates that the swarm component is surveying the landscape broadly enough to
avoid premature convergence.

---

### Ackley (`optimize_ackley`)

$$f(\mathbf{x}) = -20 \exp\!\left(-0.2\sqrt{\tfrac{1}{D}\sum x_i^2}\right) - \exp\!\left(\tfrac{1}{D}\sum \cos(2\pi x_i)\right) + 20 + e$$

- **Global minimum:** $f(\mathbf{0}) = 0$
- **Typical bounds:** $[-5, 5]^D$

Ackley combines a smooth exponential bowl with a high-frequency cosine
modulation. Near the origin the landscape is nearly flat, which can confuse
gradient estimators; far from it, misleading local minima abound. It tests
whether automatic differentiation through the `exp` and `cos` branches
produces numerically stable gradients across the full domain, and whether
PSO can survive the flat near-zero region.

---

### Himmelblau (`optimize_himmelblau`)

$$f(x_1, x_2) = (x_1^2 + x_2 - 11)^2 + (x_1 + x_2^2 - 7)^2$$

- **Four global minima**, all at $f = 0$:
  $(3,\,2)$, $(-2.805,\,3.131)$, $(-3.779,\,-3.283)$, $(3.584,\,-1.848)$
- **Typical bounds:** $[-5, 5]^2$

Himmelblau's function is 2D and has four symmetric global minima of equal
value. It tests whether the optimizer can reliably find *any* of them when
started from a random swarm, and whether the `required_converged` consensus
mechanism handles the case where different threads converge to different
(but equally valid) basins.

---

### Goldstein-Price (`optimize_goldstein_price`)

$$f(x,y) = \bigl[1 + (x+y+1)^2(19 - 14x + 3x^2 - 14y + 6xy + 3y^2)\bigr]$$
$$\times \bigl[30 + (2x-3y)^2(18 - 32x + 12x^2 + 48y - 36xy + 27y^2)\bigr]$$

- **Global minimum:** $f(0, -1) = 3$
- **Typical bounds:** $[-2, 2]^2$

Goldstein-Price is a high-degree polynomial with several local minima and a
dynamic range that spans many orders of magnitude across the search space. The
steep walls surrounding the basin at $(0, -1)$ make it easy for gradient steps
to overshoot. It stresses the BFGS line-search's step-size control and tests
whether automatic differentiation correctly handles the nested polynomial
products.

---

## Shared Source Files

| File | Purpose |
|------|---------|
| [sample_functions.hpp](sample_functions.hpp) | `__device__`/`__host__` implementations of all five functions plus their Zeus-compatible callable wrappers (`util::Rosenbrock<D>`, etc.) |
| [optimization_utils.hpp](optimization_utils.hpp) | `OptimizationParams` struct, `parse_args()`, and `print_params()` — the shared CLI plumbing used by every driver |
| [gaussian.hpp](gaussian.hpp) | Example of a user-defined callable using a `Matrix` covariance structure (not used by the benchmark drivers) |
| [tutorial.md](tutorial.md) | Step-by-step guide for converting a free function into a Zeus-compatible templated callable |
