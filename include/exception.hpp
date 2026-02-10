#pragma once
#include <stdexcept>

template<int Status>
class cuda_exception : public std::runtime_error {
public:
    explicit cuda_exception(const char* msg)
      : std::runtime_error(msg) {}
    int code() const noexcept { return Status; }
};

