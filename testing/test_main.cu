#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

#include "fun.h"

// Forward declarations for device functions from main.cu
extern "C" {
    __device__ void vector_add(const double* a, const double* b, double* result, int size);
    __device__ void vector_scale(const double* a, double scalar, double* result, int dim);
}


// Define __global__ kernels in the test file that call the device functions.
__global__ void testVectorAddKernel(const double* a, const double* b, double* result, int size) {
    // Call device function (all threads do the same work here)
    vector_add(a, b, result, size);
}

__global__ void testVectorScaleKernel(const double* a, double scalar, double* result, int dim) {
    vector_scale(a, scalar, result, dim);
}

// Test function for vector_add
void test_vector_add() {
    const int size = 5;
    double host_a[size]     = {1,  2,  3,  4,  5};
    double host_b[size]     = {5,  4,  3,  2,  1};
    double host_result[size] = {0};

    double *dev_a, *dev_b, *dev_result;
    cudaMalloc(&dev_a, size * sizeof(double));
    cudaMalloc(&dev_b, size * sizeof(double));
    cudaMalloc(&dev_result, size * sizeof(double));

    cudaMemcpy(dev_a, host_a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, host_b, size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the test kernel (one block, one thread)
    testVectorAddKernel<<<1,1>>>(dev_a, dev_b, dev_result, size);
    cudaMemcpy(host_result, dev_result, size * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; ++i) {
        double expected = host_a[i] + host_b[i];
        assert(host_result[i] == expected);
    }
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_result);
    printf("test_vector_add passed.\n");
}

// Test function for vector_scale
void test_vector_scale() {
    const int dim = 5;
    double scalar = 2.0;
    double host_a[dim]       = {1, 2, 3, 4, 5};
    double host_result[dim]  = {0};

    double *dev_a, *dev_result;
    cudaMalloc(&dev_a, dim * sizeof(double));
    cudaMalloc(&dev_result, dim * sizeof(double));

    cudaMemcpy(dev_a, host_a, dim * sizeof(double), cudaMemcpyHostToDevice);
    testVectorScaleKernel<<<1,1>>>(dev_a, scalar, dev_result, dim);
    cudaMemcpy(host_result, dev_result, dim * sizeof(double), cudaMemcpyDeviceToHost);

    for (int i = 0; i < dim; ++i) {
        assert(host_result[i] == host_a[i] * scalar);
    }
    
    cudaFree(dev_a);
    cudaFree(dev_result);
    printf("test_vector_scale passed.\n");
}

template<template<int> class Func, int DIM>
__global__ void testValueKernel(const double* x, double* out) {
  out[0] = Func<DIM>::evaluate(x);
}

template<template<int> class Func, int DIM>
void run_value_test(const double x[DIM], double expect, const char* name) {
  double *dX, *dOut, hOut;
  cudaMalloc(&dX, DIM*sizeof(double));
  cudaMalloc(&dOut,sizeof(double));
  cudaMemcpy(dX,x,DIM*sizeof(double),cudaMemcpyHostToDevice);
  testValueKernel<Func,DIM><<<1,1>>>(dX,dOut);
  cudaMemcpy(&hOut, dOut,sizeof(double),cudaMemcpyDeviceToHost);
  assert(fabs(hOut - expect) < 1e-6);
  cudaFree(dX);
  cudaFree(dOut);
  printf("%s (value) passed\n", name);
}

//------------------------------------------------------------------------------
// Gradient‐test kernel
//------------------------------------------------------------------------------
template<template<int> class Func, int DIM>
__global__ void testGradKernel(const double* x, double* g) {
  dual::calculateGradientUsingAD<Func<DIM>, DIM>(
    const_cast<double*>(x), g
  );
}

template<template<int> class Func, int DIM>
void run_grad_test(const double x[DIM], const double expect[DIM], const char* name) {
  double *dX, *dG, hG[DIM];
  cudaMalloc(&dX, DIM*sizeof(double));
  cudaMalloc(&dG, DIM*sizeof(double));
  cudaMemcpy(dX,x,DIM*sizeof(double),cudaMemcpyHostToDevice);
  testGradKernel<Func,DIM><<<1,1>>>(dX,dG);
  cudaMemcpy(hG, dG, DIM*sizeof(double),cudaMemcpyDeviceToHost);
  for(int i=0;i<DIM;++i) {
    std::cout << "expected: "<<expect[i] << "\tactual: "<< hG[i]<< std::endl;
    assert(fabs(hG[i] - expect[i]) < 1e-6);
  }
  cudaFree(dX);
  cudaFree(dG);
  printf("%s (grad) passed\n", name);
}

int main() {
    test_vector_add();
    test_vector_scale();

    {// rastirign
      const int D = 2;
      double x0[D] = {0.0,0.0};
      // at min: f=10*2 + 0 = 20
      run_value_test<util::Rastrigin,D>(x0, 0.0, "rastrigin@origin");
      double x1[D] = {0.1,-0.3};
      double expected1 = 10.0*D
        + (0.1*0.1 - 10*cos(2*M_PI*0.1))
        + ((-0.3)*(-0.3) - 10*cos(2*M_PI*-0.3));
      run_value_test<util::Rastrigin,D>(x1, expected1, "rastrigin@generic");
    }
        // --- Ackley value tests (2D) ---
    {
      const int D = 2;
      double x0[D] = {0.0,0.0};
      // at origin: f = 0
      run_value_test<util::Ackley,D>(x0, 0.0, "ackley@origin");
      double x1[D] = {0.5,-0.5};
      // compute on host:
      double sumsq = 0.5*0.5 + (-0.5)*(-0.5);
      double sumcos= cos(2*M_PI*0.5)+cos(2*M_PI*-0.5);
      double expected1 = -20*exp(-0.2*sqrt(sumsq/D))
                       - exp(sumcos/D)
                       + 20 + M_E;
      run_value_test<util::Ackley,D>(x1, expected1, "ackley@generic");
    }

    // --- Rosenbrock value tests (2D) ---
    {
      const int D = 2;
      double x0[D] = {1.0,1.0};
      // at min: f=0
      run_value_test<util::Rosenbrock,D>(x0, 0.0, "rosenbrock@min");
      double x1[D] = {1.2,1.2};
      double expected1 = 100*(1.2 - 1.2*1.2)*(1.2 - 1.2*1.2)
                       + (1 - 1.2)*(1 - 1.2);
      run_value_test<util::Rosenbrock,D>(x1, expected1, "rosenbrock@generic");
    }

    // --- AD gradient tests (2D) ---
    {
      const int D = 2;
      //--- Rastrigin grad @ origin = [0,0]
      {
        double x[D] = {0.0,0.0}, expg[D]={0,0};
        run_grad_test<util::Rastrigin,D>(x, expg, "grad_rastrigin@origin");
      }
      //--- Ackley grad @ origin = [0,0]
      {
        double x[D] = {-1e-17,1e-17}, expg[D]={-2.0,2.0};
        run_grad_test<util::Ackley,D>(x, expg, "grad_ackley@origin");
      }
      //--- Rosenbrock grad @ min = [0,0]
      {
        double x[D] = {1.0,1.0}, expg[D]={0,0};
        run_grad_test<util::Rosenbrock,D>(x, expg, "grad_rosenbrock@min");
      }
    }

    printf("All tests passed!\n");


    return 0;
}

