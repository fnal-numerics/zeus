# Zeus: An Efficient GPU Optimization method integrating PSO, BFGS, and Automatic Differentiation 

Please read the [tutorial](https://github.com/fnal-numerics/global-optimizer-gpu/tree/main/benchmarks/tutorial.md) on how to convert free functions to a templated callable that is acceptable by Zeus.

This document shows you how to:

- Use templated callable with the `Zeus` global optimization framework.
- Integrate our custom `Matrix` class with GPU support.

---

## Step 1: Use With Zeus

Now you can use the templated callable with `Zeus`:

```cpp
#include "zeus.cuh"
using namespace zeus;

Foo<2> foo;
auto result = Zeus(foo,/*lower_bound=*/-5.0,/*upper_bound=*/5.0,/*optimization=*/1024,
              /*bfgs_iterations=*/10000,/*pso_iterations=*/10,/*required_convergences=*/100,
             /*function_name=*/"foo",/*tolerance=*/1e-8,/*seed=*/42,/*index_of_run=*/run);
std::cout<< "best result: " << result.fval <<  " with status: " << result.status << std::endl;
```

Zeus returns a status code that has a following meaning:
| Status code | Meaning    | Description      |
| ------------- | ------------- |
| 0 | surrendered | reached maximum iterations without satisfying the convergence criterion |
| 1 | converged  | satisfied both norm of the gradient & number of required convergences |
| 2 | stopped early | the best result was stopped early, because other optimizations hit the norm of the gradient flag |
| 3 | malloc | an error occurred while allocating memory on the GPU |
| 4 | kernel | an error occurred while launching the kernel |
| 5 | non-finite | the function value or the gradient norm has non-finite value that is either NaN, inf, -inf, etc.. |

---

## Step 2: Use `Matrix` in Objective Function

If your objective requires structured data like a covariance matrix, use our custom `Matrix` class with GPU support.

### Gaussian Class Using `Matrix`

```cpp
#pragma once

#include <array>
#include <cuda_runtime.h>
#include "matrix.cuh"
#include "zeus.cuh"

template <std::size_t N>
class Gaussian {
  Matrix<double> C;  // Covariance matrix

public:
  // Constructor: Upload covariance matrix to device
  Gaussian(std::array<std::array<double,N>,N> const& C_host) : C(N,N) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < N; ++j)
        C(i,j) = C_host[i][j];
    C.syncHost2Device();
  }

  // Objective function: xᵀ C x
  template <class T>
  __host__ __device__
  T operator()(std::array<T,N> const& x) const {
    T q = T(0);
    #pragma unroll
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j < N; ++j) {
        auto a = C.data()[i*N + j];  // CUDA device access
        q += x[i] * T(a * x[j]);
      }
    return T(0.5) * q;
  }
};
```

---

## Full Example: 150D Gaussian Optimization with Zeus

```cpp
#include <array>
#include <iostream>
#include "gaussian.hpp" 

constexpr std::size_t D = 150;
using T = double;
T off = T(0.5);

// Create covariance matrix with structure
std::array<std::array<T, D>, D> C;
for (std::size_t i = 0; i < D; ++i)
  for (std::size_t j = 0; j < D; ++j)
    C[i][j] = (i == j ? T(1) + (D - 1) * off : off);

// Construct the Gaussian objective
Gaussian<D> g{C};

std::cout << "Running " << D << "D Gaussian minimization" << std::endl;

auto res = zeus::Zeus(g, -5.00, 5.00, D, 10000, 10, 100, "gaussian", 1e-8, 42, run);
std::cout << "Global minimum for " << D << "D Gaussian: " << res.fval << std::endl;
```

---

## Notes

- To use our `Matrix` class, your objective must support:
  - `.data()` for raw pointer access (device/host)
  - `syncHost2Device()` for copying data to the GPU
  - Indexing via `C(i,j)`
- You can add conditional compilation based on `__CUDA_ARCH__` to handle host/device behavior if needed.
- Zeus expects callables to accept `T` or `std::array<T, N>`.



🇭🇺
