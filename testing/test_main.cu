#include <assert.h>
#include <stdio.h>
#include <cuda_runtime.h>

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

int main() {
    test_vector_add();
    test_vector_scale();
    printf("All tests passed!\n");
    return 0;
}

