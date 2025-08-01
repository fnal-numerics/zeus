#include "cuda_buffer.cuh"

// allocate uninitialized buffer
cuda_buffer::cuda_buffer(std::size_t n)
  : d(nullptr), sz(n)
{
  cudaError_t status = cudaMalloc(&d, sz * sizeof(double));
  if (status != cudaSuccess) {
    throw cuda_exception<3>("cudaMalloc failed");
  }
}

// destructor
cuda_buffer::~cuda_buffer() {
  if (d) cudaFree(d);
}

