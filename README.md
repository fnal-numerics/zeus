# Zeus: An Efficient GPU Optimization method integrating PSO, BFGS, and Automatic Differentiation 

This tutorial helps users converting a free function to a templated callable and using Zeus

You'll learn how to:

- Convert a `double`-only free function into a templated callable.
- Use it with the `Zeus` global optimization framework.
- Integrate our custom `Matrix` class with GPU support.

---

## Step 1: Start with a Free Function

Here's a basic `double`-only function:

```cpp
double foo(double x) {
    return 0.5 * x * x;
}
```

This is simple, but inflexible. You can't easily run it on the GPU or with autodiff types.

---

## Step 2: Convert to a Templated Callable

To make it compatible with `Zeus`, convert it into a class with a templated call operator:

```cpp
struct Foo {
    template <typename T>
    __host__ __device__
    T operator()(T x) const {
        return T(0.5) * x * x;
    }
};
```

This works with any `T` that supports basic arithmetic: `float`, `double`, `DualNumber` types.

---

## Step 3: Use With Zeus

Now you can use the templated callable with `Zeus`:

```cpp
#include "zeus.cuh"
using namespace zeus;

Foo f;
auto result = Zeus(f,/*lower_bound=*/-5.0,/*upper_bound=*/5.0,/*optimization=*/1024,
              /*bfgs_iterations=*/10000,/*pso_iterations=*/10,/*required_convergences=*/100,
             /*function_name=*/"foo",/*tolerance=*/1e-8,/*seed=*/42,/*index_of_run=*/run);
std::cout<< "best result: " result.fval <<  
```

---

## Step 4: Use `Matrix` in Objective Function

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
#include "gaussian.cuh"  // Assuming Gaussian is in this header

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

