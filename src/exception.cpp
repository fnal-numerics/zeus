#include "exception.hpp"

namespace zeus {

CudaError::CudaError(cudaError_t error, const std::string& msg)
  : std::runtime_error(format_message(error, msg)), code_(error)
{}

cudaError_t
CudaError::code() const noexcept
{
  return code_;
}

std::string
CudaError::format_message(cudaError_t error, const std::string& msg)
{
  std::string full_msg = "CUDA Error: ";
  full_msg += cudaGetErrorString(error);
  if (!msg.empty()) {
    full_msg += " (" + msg + ")";
  }
  return full_msg;
}

} // namespace zeus
