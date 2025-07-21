#pragma once
#include <cstddef>
#include <utility>    // std::swap
#include <new>        // std::bad_alloc

extern "C" void*  cuda_malloc   (std::size_t bytes);
extern "C" void   cuda_free     (void* ptr);
extern "C" void   cuda_copy_to_device(void* dst, void const* src, std::size_t bytes);

template<std::size_t N>
class ManagedMatrix {
public:
  ManagedMatrix();                     
  ManagedMatrix(ManagedMatrix const&); 
  ManagedMatrix(ManagedMatrix&&) noexcept;
  ManagedMatrix& operator=(ManagedMatrix) noexcept;
  ~ManagedMatrix();

  // Host accessors:
  __host__ double&      at (std::size_t i, std::size_t j);
  __host__ double       atc(std::size_t i, std::size_t j) const;
  double*       host_data()   noexcept;
  double const* host_data()   const noexcept;

  // Device accessor:
  __device__ double     dev(std::size_t i, std::size_t j) const;
  double*       device_data() noexcept;
  double const* device_data() const noexcept;

private:
  void swap(ManagedMatrix& o) noexcept {
    std::swap(host_data_,  o.host_data_);
    std::swap(device_data_, o.device_data_);
  }

  double* host_data_{nullptr};
  double* device_data_{nullptr};
};

