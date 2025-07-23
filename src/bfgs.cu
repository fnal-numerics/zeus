// definition for global variables
namespace bfgs {
  __device__ int d_stopFlag       = 0;
  __device__ int d_convergedCount = 0;
  __device__ int d_threadsRemaining = 0;
}
