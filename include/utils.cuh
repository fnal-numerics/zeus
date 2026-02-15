#pragma once

#include <curand_kernel.h> // for curandState
#include <cassert>
#include <stdexcept>

#include "util.hpp"
#include "device_matrix.cuh"

inline constexpr double malloc_error = 3.0; // value 3.0
inline constexpr double kernel_error = 4.0; // value 4.0

inline double* const MALLOC_ERROR = const_cast<double*>(&malloc_error);
inline double* const KERNEL_ERROR = const_cast<double*>(&kernel_error);

namespace util {

  template <typename T>
  struct non_null {
    T ptr;

    __host__ __device__ explicit non_null(const T p) : ptr(p)
    {
#ifndef __CUDA_ARCH__
      if (p == nullptr) {
        throw std::invalid_argument(
          "util::non_null: construction from nullptr");
      }
#endif
    }

    template <typename U,
              typename = std::enable_if_t<std::is_convertible_v<U, T>>>
    __host__ __device__
    non_null(const non_null<U>& other)
      : ptr(other.get())
    {}

    // Disable construction from nullptr literal
    non_null(std::nullptr_t) = delete;
    non_null& operator=(std::nullptr_t) = delete;

    __host__ __device__ T
    get() const
    {
      return ptr;
    }
    __host__ __device__ T
    operator->() const
    {
      return ptr;
    }
    __host__ __device__ auto&
    operator*() const
    {
      return *ptr;
    }

    // Implicit conversion to the underlying pointer type
    __host__ __device__
    operator T() const
    {
      return ptr;
    }
  };

  template <class... Ptrs>
  inline void
  freeCudaPtrs(Ptrs... ptrs)
  {
    (cudaFree(ptrs), ...);
  }

  void set_stack_size();

  // The following utility functions are implemented in the header (and marked
  // inline) to ensure they are available to user code that instantiates Zeus
  // templates. This avoids 'undefined reference' or CUDA registration errors
  // when Zeus is built as a shared library.

  // https://xorshift.di.unimi.it/splitmix64.c
  // Very fast 64-bit mixer — returns a new 64-bit value each time.
  __device__ inline uint64_t
  splitmix64(uint64_t& x)
  {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }

  // return a random double in [minVal, maxVal)
  __device__ inline double
  random_double(uint64_t& state, double minVal, double maxVal)
  {
    uint64_t z = splitmix64(state);
    double u = (z >> 11) * (1.0 / 9007199254740992.0);
    return minVal + u * (maxVal - minVal);
  }

  __device__ inline double
  dot_product_device(const double* a, const double* b, int size)
  {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  __device__ inline void
  outer_product_device(const double* v1,
                       const double* v2,
                       double* result,
                       int size)
  {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        int idx = i * size + j;
        if (idx < size * size) {
          result[idx] = v1[i] * v2[j];
        }
      }
    }
  }

  template <int DIM>
  __device__ double
  calculate_gradient_norm(const double* g)
  {
    double grad_norm = 0.0;
    for (int i = 0; i < DIM; ++i) {
      grad_norm += g[i] * g[i];
    }
    return sqrt(grad_norm);
  }

  template <int DIM>
  __device__ void
  compute_search_direction(double* p,
                           const DeviceMatrix<double>* H,
                           const double* g)
  {
    for (int i = 0; i < DIM; i++) {
      double sum = 0.0;
      for (int j = 0; j < DIM; j++) {
        sum += (*H)(i, j) * g[j]; // i * dim + j since H is flattened arr[]
      }
      p[i] = -sum;
    }
  }

  // overload to take arrays
  template <int DIM>
  __device__ double
  calculate_gradient_norm(const std::array<double, DIM>& g_arr)
  {
    return calculate_gradient_norm<DIM>(g_arr.data());
  }

  template <int DIM>
  __device__ void
  compute_search_direction(std::array<double, DIM>& p_arr,
                           const DeviceMatrix<double>* H,
                           const std::array<double, DIM>& g_arr)
  {
    compute_search_direction<DIM>(p_arr.data(), H, g_arr.data());
  }

  // wrap kernel definitions extern "C" block so that their symbols are exported
  // with C linkage
  __device__ inline void
  vector_add(const double* a, const double* b, double* result, int size)
  {
    for (int i = 0; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
  }

  __device__ inline void
  vector_scale(const double* a, double scalar, double* result, int dim)
  {
    for (int i = 0; i < dim; ++i) {
      result[i] = a[i] * scalar;
    }
  }

  __device__ inline void
  initialize_identity_matrix(DeviceMatrix<double>* H, int dim)
  {
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        (*H)(i, j) = (i == j ? 1.0 : 0.0);
      }
    }
  }

  __device__ inline bool
  valid(double x)
  {
    return !(isinf(x) || isnan(x));
  }

  __device__ inline double
  pow2(double x)
  {
    return x * x;
  }

  __device__ inline void
  initialize_identity_matrix_device(double* H, int n)
  {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        H[i * n + j] = (i == j) ? 1.0 : 0.0;
      }
    }
  }

  template <int DIM>
  __device__ inline void
  matrix_multiply_device(const double* A, const double* B, double* C)
  {
    for (int i = 0; i < DIM; ++i) {
      for (int j = 0; j < DIM; ++j) {
        double sum = 0.0;
        for (int k = 0; k < DIM; ++k) {
          sum += A[i * DIM + k] * B[k * DIM + j];
        }
        C[i * DIM + j] = sum;
      }
    }
  }

  // BFGS update with compile-time dimension
  template <int DIM>
  __device__ void
  bfgs_update(DeviceMatrix<double>* H,
              const double* s,
              const double* y,
              double sTy,
              DeviceMatrix<double>* Htmp)
  {
    if (::fabs(sTy) < 1e-14)
      return;
    double rho = 1.0 / sTy;

    initialize_identity_matrix(Htmp, DIM);
    // Compute H_new element-wise without allocating large temporary matrices.
    // H_new = (I - rho * s * y^T) * H * (I - rho * y * s^T) + rho * s * s^T

    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        double sum = 0.0;
        for (int k = 0; k < DIM; k++) {
          // Compute element (i,k) of (I - rho * s * y^T)
          double A_ik = ((i == k) ? 1.0 : 0.0) - rho * s[i] * y[k];
          double inner = 0.0;
          for (int m = 0; m < DIM; m++) {
            // Compute element (m,j) of (I - rho * y * s^T)
            double B_mj = ((m == j) ? 1.0 : 0.0) - rho * y[m] * s[j];
            inner += (*H)(k, m) * B_mj;
          }
          sum += A_ik * inner;
        }
        // Add the rho * s * s^T term
        (*Htmp)(i, j) = sum + rho * s[i] * s[j];
      }
    }

    // Copy H_new back into H
    for (int i = 0; i < DIM; i++) {
      for (int j = 0; j < DIM; j++) {
        (*H)(i, j) = (*Htmp)(i, j);
      }
    }
  }

  // overloaded identity fill
  __device__ inline void
  initialize_identity_matrix(double* H, int dim)
  {
    for (int i = 0; i < dim; ++i)
      for (int j = 0; j < dim; ++j)
        H[i * dim + j] = (i == j ? 1.0 : 0.0);
  }

  // overloaded search direction
  template <int DIM>
  __device__ inline void
  compute_search_direction(double* p, const double* H, const double* g)
  {
    for (int i = 0; i < DIM; ++i) {
      double sum = 0.0;
      for (int j = 0; j < DIM; ++j) {
        sum += H[i * DIM + j] * g[j];
      }
      p[i] = -sum;
    }
  }

  // overloaded BFGS‐update
  template <int DIM>
  __device__ inline void
  bfgs_update(double* H, const double* s, const double* y, double sTy)
  {
    if (fabs(sTy) < 1e-14)
      return;
    double rho = 1.0 / sTy;
    double H_new[DIM * DIM];

    for (int i = 0; i < DIM; ++i) {
      for (int j = 0; j < DIM; ++j) {
        double sum = 0.0;
        for (int k = 0; k < DIM; ++k) {
          double Aik = ((i == k) ? 1.0 : 0.0) - rho * s[i] * y[k];
          double inner = 0.0;
          for (int m = 0; m < DIM; ++m) {
            double Bmj = ((m == j) ? 1.0 : 0.0) - rho * y[m] * s[j];
            inner += H[k * DIM + m] * Bmj;
          }
          sum += Aik * inner;
        }
        H_new[i * DIM + j] = sum + rho * s[i] * s[j];
      }
    }
    // copy back
    for (int idx = 0; idx < DIM * DIM; ++idx)
      H[idx] = H_new[idx];
  }

  // function to calculate scalar directional direvative d = g * p
  __device__ inline double
  directional_derivative(const double* grad, const double* p, int dim)
  {
    double d = 0.0;
    for (int i = 0; i < dim; ++i) {
      d += grad[i] * p[i];
    }
    return d;
  }

  __device__ inline double
  generate_random_double(curandState* state, double lower, double upper)
  {
    return lower + (upper + (-lower)) * curand_uniform_double(state);
  }

  __global__ void setup_curand_states(non_null<curandState*> states,
                                      uint64_t seed,
                                      int N);

  template <typename Function, int DIM>
  __device__ double
  line_search(double f0,
              const double* x,
              const double* p,
              const double* g,
              Function const& f)
  {
    const double c1 = 0.3;
    double alpha = 1.0;
    double ddir = dot_product_device(g, p, DIM);
    std::array<double, DIM> xTemp;
    for (int i = 0; i < 20; i++) {
      for (int j = 0; j < DIM; j++) {
        xTemp[j] = x[j] + alpha * p[j];
      }
      double f1 = f(xTemp);
      if (f1 <= f0 + c1 * alpha * ddir)
        break;
      alpha *= 0.5;
    }
    return alpha;
  }

  // overload that takes arrays
  template <typename Function, std::size_t DIM>
  __host__ __device__ double
  line_search(double current_best,
              const std::array<double, DIM>& x_arr,
              const std::array<double, DIM>& p_arr,
              const std::array<double, DIM>& g_arr,
              Function const& f)
  {
    // forward to your existing pointer‐based routine
    return line_search<Function, DIM>(
      current_best, x_arr.data(), p_arr.data(), g_arr.data(), f);
  }

  __device__ inline double
  atomicMinDouble(double* addr, double val)
  {
    // reinterpret the address as 64‑bit unsigned
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old_bits = *ptr, assumed_bits;

    do {
      assumed_bits = old_bits;
      double old_val = __longlong_as_double(assumed_bits);
      // if the current value is already <= our candidate, nothing to do
      if (old_val <= val)
        break;
      // else try to swap in the new min value’s bit‐pattern
      unsigned long long new_bits = __double_as_longlong(val);
      old_bits = atomicCAS(ptr, assumed_bits, new_bits);
    } while (assumed_bits != old_bits);

    // return the previous minimum
    return __longlong_as_double(old_bits);
  }

} // end namespace util
