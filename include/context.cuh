#pragma once

#include <cuda_runtime.h>

namespace util {
  struct BFGSContext {
    int stopFlag;
    int convergedCount;
  };
}
