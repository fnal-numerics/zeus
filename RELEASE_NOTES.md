# Release Notes

## v2.1

### 🔧 Improvements

- **Stronger BFGS line search prevents optimizer stalling**: The BFGS
  optimizer's line search now uses a modified Strong Wolfe-like acceptance
  criterion. Previously, only the Armijo sufficient-decrease condition was
  checked, which could allow steps that technically satisfied the criterion
  but did not actually reduce the objective function. This caused the
  optimizer to stall — sometimes for 100 or more consecutive iterations —
  with coordinates remaining unchanged while the algorithm believed it was
  making progress. The line search now requires both the Armijo condition
  and a strict decrease in the objective value before accepting a step.
  This improvement is particularly beneficial for ill-conditioned problems
  such as the Rosenbrock function, where the BFGS Hessian approximation
  can degrade and produce poor search directions. No changes to function
  signatures or calling code are required.

---

## v2.0

### 🎉 New Features

- **Save optimizer trajectories to disk**: Pass `--save-trajectories <filename>`
  to any example program to record the full path the optimizer took through
  parameter space. Output files include the objective function value, gradient
  norm, and a convergence status code at every step, making it straightforward
  to diagnose why a run converged (or did not). Both TSV and NetCDF4 (`.nc`)
  output formats are supported; the format is chosen automatically from the
  file extension.

- **Selectable random number generator**: You can now choose the cuRAND
  generator used to initialize the particle swarm and BFGS starting points.
  The options are XORWOW (default), Philox, and Sobol. Different generators
  can affect the diversity of the initial population and, in turn, the quality
  of the solution found.

- **New example: dijet spectrum fit**: A new `fit_dijet_spectrum` example
  demonstrates fitting a parametric model to a high-energy physics dijet
  spectrum using a Poisson negative log-likelihood objective.

- **New example: correlated Gaussian and neural-network fitting**: The
  `optimize_gaussian_nn` example shows how to optimize over the weights of a
  small neural network and over a high-dimensional correlated Gaussian, both
  of which stress-test the optimizer in ways the standard benchmark functions
  do not.

### 🔧 Improvements

- **Clearer errors for invalid objective functions**: Zeus now uses C++20
  concepts to validate that your objective function is callable with both
  `double` and dual-number types. If your function does not satisfy the
  requirements, the compiler will tell you exactly what is wrong rather than
  emitting a long template error.

- **Step-failure diagnostics in trajectory output**: When the BFGS line search
  produces a zero step size, that event is now recorded in the trajectory file.
  This makes it much easier to identify runs where the optimizer stalled and to
  understand at which iteration the problem occurred.

---

## v1.1 (Maintenance Patch)

**Branch:** `maintenance_v_1`

### Build System Fixes

- **Static-library enforcement.**
  Zeus now rejects `BUILD_SHARED_LIBS=ON` with a `FATAL_ERROR`.
  CUDA separable compilation (`-rdc=true`) requires static object files
  at device-link time; shared libraries silently produce broken executables.
  Previous releases defaulted to shared libraries.

- **Fixed `digamma` link error.**
  Removed `__inline__` from the `digamma` forward declaration in `duals.cuh`.
  The qualifier told NVCC call sites the body was available for inlining,
  but the definition lives in a separately-compiled translation unit (`duals.cu`),
  causing undefined-reference errors at device-link time.

- **Created `include/fun.h` forwarder header.**
  Test files (`fun_tests.cu`, `pso_tests.cu`, `bfgs_tests.cu`) include `fun.h`,
  but this header was never tracked in version control.
  Added a thin forwarder that delegates to `examples/sample_functions.hpp`.

- **Added `examples/` to unit-test include path.**
  The `unit_test` CMake target now includes the `examples/` directory,
  making `sample_functions.hpp` (and thus `fun.h`) resolvable.

### Remote Development Improvements

- **`remote_env.sh`: unversioned module loads.**
  Replaced pinned `cmake/3.24.3` and `gcc/12.2.0` with generic `cmake` and `gcc`
  module loads, preventing breakage when Perlmutter retires old module versions.

- **`Makefile` `remote-build`: proper error propagation.**
  Replaced the `&&`-chained shell command with `set -e` and an explicit `if` guard,
  so that build failures on the remote host are correctly reported to `make`.

- **`Makefile` `remote-sync`: mirror deletions.**
  Added `--delete` with `--filter=':- .gitignore'` to `rsync`,
  ensuring locally-deleted files are also removed on the remote host.

---

## v1.0

**Tag:** `v1_0`

### Features

- **GPU-accelerated global optimization** combining Particle Swarm Optimization (PSO)
  with L-BFGS on NVIDIA GPUs using CUDA.
- **Automatic differentiation** via dual numbers for exact gradient computation,
  eliminating the need for user-supplied gradients.
- **Templated objective functions** — supply any callable that accepts
  `std::array<T, N>` and returns `T`, where `T` is `double` or `DualNumber`.
- **Scalable to high dimensions** — demonstrated on problems up to 150 dimensions
  (correlated Gaussian optimization).

### Dual Number Library

- `DualNumber` type with `constexpr` arithmetic operators.
- Mathematical functions: `sin`, `cos`, `exp`, `sqrt`, `log`, `pow`, `abs`,
  `atan2`, `lgamma` (via `digamma`).
- Equality and inequality operators.

### GPU Infrastructure

- `cuda_buffer<T>` — RAII wrapper for GPU memory with copy/move semantics
  and host↔device transfer.
- `DeviceMatrix` — GPU-resident dense matrix with host-side construction
  and device-side access.
- `CudaError` exception hierarchy for structured CUDA error handling.
- `NonNull<T>` — a non-nullable pointer wrapper for safer kernel argument passing.

### Build System

- CMake-based build with `FetchContent` and `find_package(Zeus)` support.
- Installable library with exported CMake targets (`Zeus::zeus`).
- Catch2 v3 test suite for dual numbers, matrices, buffers, PSO, and BFGS.
- `Makefile` with remote development targets for Perlmutter (sync, build, test).

### Included Examples

- Optimizer benchmarks: Rosenbrock, Rastrigin, Ackley, Goldstein-Price, Himmelblau.
- Full tutorial with `Matrix` class integration for structured objectives.
