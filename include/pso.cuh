#pragma once
#include <curand_kernel.h>
#include "utils.cuh"
#include "cuda_buffer.cuh"
#include "traits.hpp"

using namespace zeus;

namespace bfgs {
  extern __device__ int d_stopFlag; // 0 = keep going; 1 = stop immediately
  extern __device__ int d_convergedCount; // how many threads have converged?
  extern __device__ int d_threadsRemaining;
}

namespace pso {

  // pso initialization kernel: initialize X, V, pBest;
  //                            atomically seed gBestVal/gBestX
  template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
  __global__ void
  initKernel(Function func,
             double lower,
             double upper,
             double* X,
             double* V,
             double* pBestX,
             double* pBestVal,
             double* gBestX,
             double* gBestVal,
             int N,
             uint64_t seed,
             curandState* states)
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
      return;

    curandState localState = states[i];
    const double vel_range = (upper - lower) * 0.1;
    // init position & velocity
    for (int d = 0; d < DIM; ++d) {
      double rx = util::generate_random_double(&localState, lower, upper);
      double rv =
        util::generate_random_double(&localState, -vel_range, vel_range);

      X[i * DIM + d] = rx;
      V[i * DIM + d] = rv;
      pBestX[i * DIM + d] = rx;
    }

    std::array<double, DIM> xarr;
    double* x = xarr.data();
#pragma unroll
    for (int d = 0; d < DIM; ++d)
      x[d] = X[i * DIM + d]; // pointer indexing is allowed in __device__

    // eval personal best
    double fval = func(xarr);
    pBestVal[i] = fval;

    // atomic update of global best
    double oldGB = util::atomicMinDouble(gBestVal, fval);
    if (fval < oldGB) {
      // we’re the new champion: copy pBestX into gBestX
      for (int d = 0; d < DIM; ++d)
        gBestX[d] = pBestX[i * DIM + d];
    }
    states[i] = localState; // next time we draw, we continue where we left off
  }

  // one PSO iteration kernel
  template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
  __global__ void
  iterKernel(Function func,
             double lower,
             double upper,
             double w,  // weight inertia
             double c1, // cognitive coefficient
             double c2, // social coefficient
             double* X,
             double* V,
             double* pBestX,
             double* pBestVal,
             double* gBestX,
             double* gBestVal,
             double* traj, // pass nullptr if not saving
             bool saveTraj,
             int N,
             int iter,
             uint64_t seed,
             curandState* states)
  // iteration
  {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
      return;

    curandState localState = states[i];
    // update velocity & position
    for (int d = 0; d < DIM; ++d) {
      double r1 = util::generate_random_double(&localState, 0.0, 1.0);
      double r2 = util::generate_random_double(&localState, 0.0, 1.0);

      double xi = X[i * DIM + d];
      double vi = V[i * DIM + d];
      double pb = pBestX[i * DIM + d];
      double gb = gBestX[d];

      double nv = w * vi +
                  c1 * r1 * (pb - xi) // “cognitive” pull toward personal best
                  + c2 * r2 * (gb - xi); // “social” pull toward global best
      double nx = xi + nv;

      V[i * DIM + d] = nv;
      X[i * DIM + d] = nx;

      if (saveTraj) {
        // traj is laid out [iter][i][d]
        size_t idx = size_t(iter) * N * DIM + i * DIM + d;
        traj[idx] = nx;
      }
    }

    std::array<double, DIM> xarr;
    double* x = xarr.data();
#pragma unroll
    for (int d = 0; d < DIM; ++d)
      x[d] = X[i * DIM + d];
    // evaluate at new position
    double fval = func(xarr);

    // personal best? no atomic needed, it's a private best position
    if (fval < pBestVal[i]) {
      pBestVal[i] = fval;
      for (int d = 0; d < DIM; ++d)
        pBestX[i * DIM + d] = X[i * DIM + d];
    }

    // global best?
    double oldGB = util::atomicMinDouble(gBestVal, fval);
    if (fval < oldGB) {
      for (int d = 0; d < DIM; ++d)
        gBestX[d] = X[i * DIM + d];
    }
    states[i] = localState; // next time we draw, we continue where we left off
  }

  // A simple logger function you can call from your launch()
  template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
  void
  saveIteration(int iter,
                int N,
                const double* dX,
                const double* dV,
                Function const& f,
                std::ofstream& out,
                std::vector<double>& hX,
                std::vector<double>& hV)
  {
    // copy device → host
    cudaMemcpy(hX.data(), dX, sizeof(double) * N * DIM, cudaMemcpyDeviceToHost);
    cudaMemcpy(hV.data(), dV, sizeof(double) * N * DIM, cudaMemcpyDeviceToHost);

    // write one iteration’s rows
    for (int i = 0; i < N; ++i) {
      out << iter << '\t' << i;
      // x coords
      for (int d = 0; d < DIM; ++d)
        out << std::scientific << '\t' << hX[i * DIM + d];
      // v coords
      for (int d = 0; d < DIM; ++d)
        out << '\t' << hV[i * DIM + d];

      // compute fval on host
      std::array<double, DIM> xx;
      for (int d = 0; d < DIM; ++d)
        xx[d] = hX[i * DIM + d];
      double fval = f(xx);
      out << '\t' << fval << '\n';
    }
    out.flush();
  }

  template <typename Function, std::size_t DIM = fn_traits<Function>::arity>
  dbuf
  launch(const int PSO_ITER,
         const int N,
         const double lower,
         const double upper,
         float& ms_init,
         float& ms_pso,
         const int seed,
         curandState* states,
         Function fun)
  { //, Result<DIM>& best) {
    static_assert(!std::is_reference<decltype(fun)>::value,"ZEUS: function must be passed by VALUE to kernels (no &).");

    // allocate PSO buffers on device
    size_t freeBytes = 0, total = 0;
    cudaMemGetInfo(&freeBytes, &total);
    size_t sizeX        = size_t(N) * DIM;
    size_t sizePBestVal = size_t(N);
    size_t sizeF        = sizePBestVal;                    // same size as N doubles
    size_t sizeGBestX   = DIM;
 
   size_t need = sizeof(double) * (sizeX * 3   // X, V, PBestX
            + sizePBestVal * 2         // PBestVal, F
            + sizeGBestX + 1 );//sizeof(double);              // GBestX + Val
    printf("GPU reporting %.2f GB free of %.2f GB total\n",freeBytes/1e9, total/1e9);
    printf("Need %.2f GB for PSO buffers\n", need/1e9);
    if (need > freeBytes) {
      
      throw cuda_exception<3>("Not enough device memory for PSO buffers");
   } 
   // once we know we have enough memory, we can allocate it
    try { // resource acquisition is initialization
      dbuf dX (sizeX);
      dbuf dV (sizeX);
      dbuf dPBestX (sizeX);
      dbuf dPBestVal (sizePBestVal);
      dbuf dGBestX (sizeGBestX);
      dbuf dGBestVal (1);
      dbuf dF (sizeF); 
      int zero = 0;
      cudaMemcpyToSymbol(bfgs::d_stopFlag, &zero, sizeof(int));
      cudaMemcpyToSymbol(bfgs::d_threadsRemaining, &N, sizeof(int));
      cudaMemcpyToSymbol(bfgs::d_convergedCount, &zero, sizeof(int));
      // set seed to infinity
      {
        double inf = std::numeric_limits<double>::infinity();
        cudaMemcpy(dGBestVal.data(), &inf, sizeof(inf), cudaMemcpyHostToDevice);
      }

      dim3 psoBlock(256);
      dim3 psoGrid((N + psoBlock.x - 1) / psoBlock.x);

      // host-side buffers for printing
      //double hostGBestVal;
      //std::vector<double> hostGBestX(DIM);

    // PSO‐init Kernel
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    initKernel<Function, DIM><<<psoGrid, psoBlock>>>(fun,
                                                     lower,
                                                     upper,
                                                     dX.data(),
                                                     dV.data(),
                                                     dPBestX.data(),
                                                     dPBestVal.data(),
                                                     dGBestX.data(),
                                                     dGBestVal.data(),
                                                     N,
                                                     seed,
                                                     states);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    if (cudaGetLastError() != cudaSuccess)
      throw cuda_exception<4>("PSO initKernel failed");
    cudaEventElapsedTime(&ms_init, t0, t1);
#if (0)
    std::vector<double> hX(N * DIM), hV(N * DIM);
    std::ofstream out("pso_log.tsv");
    assert(out && "failed to open pso_log.tsv");
    out << "iter\tpid";
    for (int d = 0; d < DIM; ++d)
      out << "\tx" << d;
    for (int d = 0; d < DIM; ++d)
      out << "\tv" << d;
    out << "\tfval\n";
    saveIteration<Function, DIM>(0, N, dX.data(), dV.data(), fun, out, hX, hV);
#endif
    // copy back and print initial global best
    auto hostGBestVal = dGBestVal.copy_to_host();
    auto hostGBestX = dGBestX.copy_to_host();
    printf("Initial PSO gBestVal = %.6e at gBestX = [", hostGBestVal);
    for (int d = 0; d < DIM; ++d)
      printf(" %.4f", hostGBestX[d]);
    printf(" ]\n\n");

    // PSO iterations
    // const double w  = 0.7298, c1 = 1.4962, c2 = 1.4962;
    const double w = 0.5, c1 = 1.2, c2 = 1.5;
    for (int iter = 1; iter < PSO_ITER + 1; ++iter) {
      cudaEventRecord(t0);
      iterKernel<Function, DIM><<<psoGrid, psoBlock>>>(fun,
                                                       lower,
                                                       upper,
                                                       w,
                                                       c1,
                                                       c2,
                                                       dX,
                                                       dV,
                                                       dPBestX.data(),
                                                       dPBestVal.data(),
                                                       dGBestX.data(),
                                                       dGBestVal.data(),
                                                       nullptr, // traj
                                                       false,   // saveTraj
                                                       N,
                                                       iter,
                                                       seed,
                                                       states); //, best);
      cudaEventRecord(t1);
      cudaEventSynchronize(t1);
      if (cudaGetLastError() != cudaSuccess)
        throw cuda_exception<4>("PSO iterKernel failed");
      float ms_iter = 0;
      cudaEventElapsedTime(&ms_iter, t0, t1);
      hostGBestVal = dGBestVal.copy_to_host();
      hostGBestX = dGBestX.copy_to_host();
      //saveIteration<Function, DIM>(iter, N, dX, dV, f, out, hX, hV);
      ms_pso += ms_iter;
    } // end pso loop
    // printf("total pso time = %.3f\n", ms_pso+ms_init);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return dPBestX; // return the personal best for each particle
 
   } catch (cuda_exception<3>&) {
     // allocation failure 
      throw;
    } 
  }

} // end namespace pso
