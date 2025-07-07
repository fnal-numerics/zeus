#pragma once
#include <curand_kernel.h>

namespace pso {
  template<typename Func, int DIM>
  __global__ void initKernel(Func f,
                             double lo, double hi,
                             double* X, double* V,
                             double* pBestX, double* pBestVal,
                             double* gBestX, double* gBestVal,
                             int N, uint64_t seed,
                             curandState* states);

  template<typename Func, int DIM>
  __global__ void iterKernel(Func f,
                             double lo, double hi,
                             double w, double c1, double c2,
                             double* X, double* V,
                             double* pBestX, double* pBestVal,
                             double* gBestX, double* gBestVal,
                             int N, int iter,
                             curandState* states);

  template<typename Func, int DIM>
  double* launch(int N, int iters,
                 double lo, double hi,
                 curandState* states,
                 float& ms_init, float& ms_loop);
}

