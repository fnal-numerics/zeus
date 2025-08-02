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

// copy ctor
cuda_buffer::cuda_buffer(cuda_buffer const& u)
  : d(nullptr), sz(u.sz)
{
  cudaError_t st = cudaMalloc(&d, sz * sizeof(double));
  if (st != cudaSuccess) {
    throw cuda_exception<3>("cudaMalloc failed in copy ctor");
  }
  st = cudaMemcpy(d, u.d, sz * sizeof(double),
                  cudaMemcpyDeviceToDevice);
  if (st != cudaSuccess) {
    throw cuda_exception<4>("cudaMemcpy D→D failed in copy ctor");
  }
}

// copy-assign (copy-and-swap)
cuda_buffer& cuda_buffer::operator=(cuda_buffer const& u) {
  if (this != &u) {
    cuda_buffer tmp(u);
    swap(tmp);
  }
  return *this;
}

// move ctor
cuda_buffer::cuda_buffer(cuda_buffer&& u) noexcept
  : d(u.d), sz(u.sz)
{
  u.d = nullptr;
  u.sz = 0;
}

//  move-assign
cuda_buffer& cuda_buffer::operator=(cuda_buffer&& u) noexcept {
  if (this != &u) {
    if (d) cudaFree(d);
    d    = u.d;
    sz   = u.sz;
    u.d  = nullptr;
    u.sz = 0;
  }
  return *this;
}


// vector-return overload (throws on failure)
std::vector<double> cuda_buffer::copy_to_host() const {
  std::vector<double> out;
  int status = copy_to_host(out);
  if (status != 0)
    throw cuda_exception<4>("cudaMemcpy D→H failed");
  return out;
}

// raw-pointer overload (returns 0 on success, <0 on error)
int cuda_buffer::copy_to_host(double* out, std::size_t n) const {
  if (n != sz) return -1;
  if (n > 0 && d) {
    auto st = cudaMemcpy(out, d, n * sizeof(double),
                         cudaMemcpyDeviceToHost);
    if (st != cudaSuccess) return -2;
  }
  return 0;
}

// vector-by-ref overload
int cuda_buffer::copy_to_host(std::vector<double>& out) const {
  out.resize(sz);
  return copy_to_host(out.data(), sz);
}

