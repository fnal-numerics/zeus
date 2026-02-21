#pragma once
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

class cuda_exception : public std::runtime_error {
public:
    explicit cuda_exception(cudaError_t error, const std::string& msg = "")
        : std::runtime_error(format_message(error, msg)), code_(error) {}

    cudaError_t code() const noexcept { return code_; }

private:
    cudaError_t code_;

    static std::string format_message(cudaError_t error, const std::string& msg) {
        std::string full_msg = "CUDA Error: ";
        full_msg += cudaGetErrorString(error);
        if (!msg.empty()) {
            full_msg += " (" + msg + ")";
        }
        return full_msg;
    }
};


