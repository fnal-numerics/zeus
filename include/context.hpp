#pragma once

namespace util {
  /// Shared state for coordinating parallel BFGS optimizations.
  /// Tracks global convergence status and signals early termination
  /// when required number of optimizations have converged.
  struct BFGSContext {
    int stopFlag;        ///< Global flag to signal all threads to stop
    int convergedCount;  ///< Number of optimizations that have converged
  };
}
