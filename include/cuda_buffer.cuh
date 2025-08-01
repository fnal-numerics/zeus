#pragma once

// our own exception template class that would return either status 3 or 4 depending on which failure occured
// 3 is for cudamalloc failure, and 4 is for kernel failure 

struct cuda_buffer {
  double * d = nullptr;
  size_t sz;
  
  cuda_buffer(size_t n);
  cuda_buffer(std::array<double> const& h);
  ~cuda_buffer();

  // what about:
  //  1) copy ctor
  //  2) copy assignment
  //  3) move ctor
  //  4) move assignment

  cuda_buffer::cuda_buffer(size_t n) : sz(n) {
    auto status cudaMalloc(&d, sz * sizeof(double));
    if(status != cudaSuccess) throw std::badalloc();

  // we could also have a copy_to_host member function something like:
  int copy_to_host(double* buffer, size_t n) const;
  // and overloaded sets:
  int copy_to_host(std::vector<double> & out) const;
  // or
  std::vector<double> copy_to_host() const;
};


