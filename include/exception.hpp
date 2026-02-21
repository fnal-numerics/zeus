#pragma once
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

namespace zeus {

/// Exception class for CUDA runtime errors.
class CudaError : public std::runtime_error {
public:
  explicit CudaError(
    cudaError_t error,
    const std::string& msg = ""); ///< Construct with CUDA error and message

  cudaError_t code() const noexcept; ///< Get the CUDA error code

private:
  cudaError_t code_; ///< The CUDA error code

  static std::string format_message(
    cudaError_t error,
    const std::string& msg); ///< Format the exception message
};

} // namespace zeus





