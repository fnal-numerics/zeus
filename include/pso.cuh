#pragma once
#include <curand_kernel.h>
#include "utils.cuh"

namespace bfgs {
  extern __device__ int d_stopFlag; // 0 = keep going; 1 = stop immediately
  extern __device__ int d_convergedCount; // how many threads have converged?
  extern __device__ int d_threadsRemaining;
}

namespace pso {

  // pso initialization kernel: initialize X, V, pBest;
  //                            atomically seed gBestVal/gBestX
  template <typename Function, int DIM>
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
  template <typename Function, int DIM>
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
  template <typename Function, int DIM>
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

  template <typename Function, int DIM>
  double const*
  launch(const int PSO_ITER,
         const int N,
         const double lower,
         const double upper,
         float& ms_init,
         float& ms_pso,
         const int seed,
         curandState* states,
         Function const& f)
  { //, Result<DIM>& best) {
    // allocate PSO buffers on device
    double *dX=nullptr, *dV=nullptr, *dPBestVal=nullptr, *dGBestX=nullptr, *dGBestVal=nullptr, *dPBestX=nullptr;
    double* dF;
    size_t freeBytes = 0, total = 0;
    cudaMemGetInfo(&freeBytes, &total);
    size_t sizeX        = static_cast<size_t>(N) * DIM * sizeof(double);
    size_t sizePBestVal = static_cast<size_t>(N) * sizeof(double);
    size_t sizeF        = sizePBestVal;                    // same size as N doubles
    size_t sizeGBestX   = static_cast<size_t>(DIM) * sizeof(double);
 
   size_t need = sizeX * 3   // X, V, PBestX
            + sizePBestVal * 2         // PBestVal, F
            + sizeGBestX + sizeof(double);              // GBestX + Val
   if (need > freeBytes) return MALLOC_ERROR;
    // once we know we have enough memory, we can allocate it
    if (cudaMalloc(&dX,        sizeX)        != cudaSuccess ||
        cudaMalloc(&dV,        sizeX)        != cudaSuccess ||
        cudaMalloc(&dPBestX,   sizeX)        != cudaSuccess ||
        cudaMalloc(&dPBestVal, sizePBestVal) != cudaSuccess ||
        cudaMalloc(&dGBestX,   sizeGBestX)   != cudaSuccess ||
        cudaMalloc(&dGBestVal, sizeof(double)) != cudaSuccess ||
        cudaMalloc(&dF,        sizeF)        != cudaSuccess)
    {
        // Free anything that might have succeeded before the failure
        util::freeCudaPtrs(dX, dV, dPBestX, dPBestVal,
                           dGBestX, dGBestVal, dF);
        return MALLOC_ERROR;                 // ← sentinel value (3)
    }
    int zero = 0;
    cudaMemcpyToSymbol(bfgs::d_stopFlag, &zero, sizeof(int));
    cudaMemcpyToSymbol(bfgs::d_threadsRemaining, &N, sizeof(int));
    cudaMemcpyToSymbol(bfgs::d_convergedCount, &zero, sizeof(int));
    // set seed to infinity
    {
      double inf = std::numeric_limits<double>::infinity();
      cudaMemcpy(dGBestVal, &inf, sizeof(inf), cudaMemcpyHostToDevice);
    }

    dim3 psoBlock(256);
    dim3 psoGrid((N + psoBlock.x - 1) / psoBlock.x);

    // host-side buffers for printing
    double hostGBestVal;
    std::vector<double> hostGBestX(DIM);

    // PSO‐init Kernel
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    initKernel<Function, DIM><<<psoGrid, psoBlock>>>(f,
                                                     lower,
                                                     upper,
                                                     dX,
                                                     dV,
                                                     dPBestX,
                                                     dPBestVal,
                                                     dGBestX,
                                                     dGBestVal,
                                                     N,
                                                     seed,
                                                     states);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      std::fprintf(
        stderr, "pso::initKernel launch error: %s\n", cudaGetErrorString(err));
        util::freeCudaPtrs(dX, dV, dPBestVal, dGBestX, dGBestVal,dF);
        return KERNEL_ERROR;
    }  
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::fprintf(
        stderr, "pso::initKernel runtime error: %s\n", cudaGetErrorString(err));
        util::freeCudaPtrs(dX, dV, dPBestVal, dGBestX, dGBestVal, dF);       
        return KERNEL_ERROR;
    }
    cudaEventElapsedTime(&ms_init, t0, t1);
    // printf("PSO‑Init Kernel execution time = %.4f ms\n", ms_init);
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
    saveIteration<Function, DIM>(0, N, dX, dV, f, out, hX, hV);
#endif
    // copy back and print initial global best
    cudaMemcpy(&hostGBestVal, dGBestVal, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostGBestX.data(), dGBestX, DIM * sizeof(double), cudaMemcpyDeviceToHost);
    err = cudaGetLastError();
    if(err != cudaSuccess) { 
     util::freeCudaPtrs(dX, dV, dPBestVal, dGBestX, dGBestVal, dF);
     return MALLOC_ERROR;
    }
    printf("Initial PSO gBestVal = %.6e at gBestX = [", hostGBestVal);
    for (int d = 0; d < DIM; ++d)
      printf(" %.4f", hostGBestX[d]);
    printf(" ]\n\n");

    // PSO iterations
    // const double w  = 0.7298, c1 = 1.4962, c2 = 1.4962;
    const double w = 0.5, c1 = 1.2, c2 = 1.5;
    for (int iter = 1; iter < PSO_ITER + 1; ++iter) {
      cudaEventRecord(t0);
      iterKernel<Function, DIM><<<psoGrid, psoBlock>>>(f,
                                                       lower,
                                                       upper,
                                                       w,
                                                       c1,
                                                       c2,
                                                       dX,
                                                       dV,
                                                       dPBestX,
                                                       dPBestVal,
                                                       dGBestX,
                                                       dGBestVal,
                                                       nullptr, // traj
                                                       false,   // saveTraj
                                                       N,
                                                       iter,
                                                       seed,
                                                       states); //, best);
      cudaEventRecord(t1);
      cudaEventSynchronize(t1);
      err = cudaGetLastError();
      if (err != cudaSuccess) { 
        std::fprintf(stderr,
                     "pso::iterKernel launch error: %s\n",
                     cudaGetErrorString(err));
        util::freeCudaPtrs(dX, dV, dPBestVal, dGBestX, dGBestVal, dF);
        return KERNEL_ERROR;
      } 
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        std::fprintf(stderr,
                     "pso::iterKernel runtime error: %s\n",
                     cudaGetErrorString(err));
         util::freeCudaPtrs(dX, dV, dPBestVal, dGBestX, dGBestVal, dF);
         return KERNEL_ERROR;
      }
      float ms_iter = 0;
      cudaEventElapsedTime(&ms_iter, t0, t1);
      cudaMemcpy(
        &hostGBestVal, dGBestVal, sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy(hostGBestX.data(),
                 dGBestX,
                 DIM * sizeof(double),
                 cudaMemcpyDeviceToHost);
      err = cudaGetLastError();
      if(err != cudaSuccess) {
         util::freeCudaPtrs(dX, dV, dPBestVal, dGBestX, dGBestVal, dF);
         return MALLOC_ERROR;
      }
      //saveIteration<Function, DIM>(iter, N, dX, dV, f, out, hX, hV);
      ms_pso += ms_iter;
    } // end pso loop
    // printf("total pso time = %.3f\n", ms_pso+ms_init);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    util::freeCudaPtrs(dX, dV, dPBestVal, dGBestX, dGBestVal, dF);    
    return dPBestX; // return the personal best for each particle
  }

} // end namespace pso
