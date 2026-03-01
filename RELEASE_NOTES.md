# Release Notes

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
