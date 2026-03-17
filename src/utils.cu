#include "utils.cuh"
#include <string>
#include <fstream>
#include <string_view>
#include <iomanip>
#include <cmath>
#include <iostream>

#ifdef ZEUS_HAS_NETCDF4
#include <netcdf>
#endif

namespace bfgs {}

namespace util {

  __global__ void
  setupXorwowStates(util::NonNull<curandStateXORWOW_t*> states,
                    uint64_t seed,
                    int N)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
      curand_init(
        seed, (unsigned long long)thread_id, 0ULL, &states[thread_id]);
    }
  }

  __global__ void
  setupPhiloxStates(util::NonNull<curandStatePhilox4_32_10_t*> states,
                    uint64_t seed,
                    int N)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
      curand_init(
        seed, (unsigned long long)thread_id, 0ULL, &states[thread_id]);
    }
  }

  __global__ void
  setupSobolStates(util::NonNull<curandStateSobol32_t*> states,
                   unsigned int* vectors,
                   int N)
  {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N) {
      curand_init(vectors, (unsigned int)thread_id, &states[thread_id]);
    }
  }

  void
  setStackSize()
  {
    // logic to set the stact size limit to 65 kB per thread
    size_t currentStackSize = 0;
    cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
    // printf("Current stack size: %zu bytes\n", currentStackSize);
    size_t newStackSize = 64 * 1024; // 65 kB
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (err != cudaSuccess) {
      printf("cudaDeviceSetLimit Stack error: %s\n", cudaGetErrorString(err));
    }

    // get current device ID
    int device;
    err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
      printf("cudaGetDevice error: %s\n", cudaGetErrorString(err));
      return;
    }

    // Get device properties (including total global memory)
    cudaDeviceProp deviceProp;
    err = cudaGetDeviceProperties(&deviceProp, device);
    if (err != cudaSuccess) {
      printf("cudaGetDeviceProperties error: %s\n", cudaGetErrorString(err));
      return;
    }

    // Use total free memory as heap size
    size_t freeBytes = 0, total = 0;
    cudaMemGetInfo(&freeBytes, &total);
    printf("GPU reporting %.2f GB free of %.2f GB total\n",
           freeBytes / (1024.0 * 1024.0 * 1024.0),
           total / (1024.0 * 1024.0 * 1024.0));
    // Use a reasonable heap size (up to 1GB or 50% of free memory, whichever is
    // smaller)
    size_t newHeap = std::min(freeBytes / 2, (size_t)1024 * 1024 * 1024);
    err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, newHeap);
    if (err != cudaSuccess) {
      printf("Failed to set heap to %zu bytes: %s\n",
             newHeap,
             cudaGetErrorString(err));
    } else {
      printf("Successfully set heap size to %.2f GB\n",
             newHeap / (1024.0 * 1024.0 * 1024.0));
    }
  }

  __global__ void
  fillWithNaN_kernel(double* d_ptr, size_t n)
  {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
      d_ptr[idx] = NAN;
    }
  }

  cudaError_t
  fillWithNaN(double* d_ptr, size_t n)
  {
    if (n == 0)
      return cudaSuccess;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    fillWithNaN_kernel<<<gridSize, blockSize>>>(d_ptr, n);
    return cudaGetLastError();
  }

  static bool
  endsWith(std::string_view str, std::string_view suffix)
  {
    if (str.size() < suffix.size())
      return false;
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
  }

  cudaError_t
  writeTrajectoryDataTSV(double* hostTrajectoryCoords,
                         double* hostTrajectoryFval,
                         double* hostTrajectoryGrad,
                         int8_t* hostStatus,
                         double* hostAlpha,
                         const OptimizationParams& params,
                         int DIM,
                         std::string_view filename)
  {
    std::ofstream stepOut(std::string{filename});
    stepOut << "traj\tstep\tfval\tgrad\tstatus\talpha";
    for (int d = 0; d < DIM; d++)
      stepOut << "\tx" << d;
    stepOut << "\n";
    stepOut << std::scientific << std::setprecision(17);
    int N = params.N;
    int MAX_ITER = params.MAX_ITER;
    for (int i = 0; i < N; i++) {
      for (int it = 0; it < MAX_ITER; it++) {
        stepOut << i << "\t" << it;

        // Write fval and grad first
        double fval = hostTrajectoryFval[it * N + i];
        double gnorm = hostTrajectoryGrad[it * N + i];
        if (std::isnan(fval)) {
          stepOut << "\tNaN";
        } else {
          stepOut << "\t" << fval;
        }
        if (std::isnan(gnorm)) {
          stepOut << "\tNaN";
        } else {
          stepOut << "\t" << gnorm;
        }
        stepOut << "\t" << (int)hostStatus[it * N + i];
        stepOut << "\t" << hostAlpha[it * N + i];

        // Then write coordinates
        for (int d = 0; d < DIM; d++) {
          double v = hostTrajectoryCoords[it * DIM * N + d * N + i];
          if (std::isnan(v)) {
            stepOut << "\tNaN";
          } else {
            stepOut << "\t" << v;
          }
        }
        stepOut << "\n";
      }
    }
    stepOut.close();
    return cudaSuccess;
  }

#ifdef ZEUS_HAS_NETCDF4
  cudaError_t
  writeTrajectoryDataNetCDF4(double* hostTrajectoryCoords,
                             double* hostTrajectoryFval,
                             double* hostTrajectoryGrad,
                             int8_t* hostStatus,
                             double* hostAlpha,
                             const OptimizationParams& params,
                             int DIM,
                             std::string_view filename)
  {
    using namespace netCDF;
    try {
      NcFile dataFile(std::string{filename}, NcFile::replace);

      NcDim trajDim = dataFile.addDim("trajectory", params.N);
      NcDim stepDim = dataFile.addDim("step", params.MAX_ITER);
      NcDim spatialDim = dataFile.addDim("spatial_dim", DIM);

      // Coords: (step, spatial_dim, trajectory)
      std::vector<NcDim> coordsDims = {stepDim, spatialDim, trajDim};
      NcVar coordsVar =
        dataFile.addVar("trajectory_coords", ncDouble, coordsDims);
      coordsVar.putVar(hostTrajectoryCoords);

      // fval, grad, status, alpha: (step, trajectory)
      std::vector<NcDim> valDims = {stepDim, trajDim};
      NcVar fvalVar = dataFile.addVar("fval", ncDouble, valDims);
      fvalVar.putVar(hostTrajectoryFval);

      NcVar gradVar = dataFile.addVar("grad", ncDouble, valDims);
      gradVar.putVar(hostTrajectoryGrad);

      NcVar statusVar = dataFile.addVar("status", ncByte, valDims);
      statusVar.putVar(hostStatus);

      NcVar alphaVar = dataFile.addVar("alpha", ncDouble, valDims);
      alphaVar.putVar(hostAlpha);

      // Attributes
      dataFile.putAtt("ZEUS_TRAJECTORY_FORMAT_VERSION", ncInt, 1);
      dataFile.putAtt("fun_name", params.fun_name);
      dataFile.putAtt("lower_bound", ncDouble, params.lower_bound);
      dataFile.putAtt("upper_bound", ncDouble, params.upper_bound);
      dataFile.putAtt("N", ncInt, params.N);
      dataFile.putAtt("MAX_ITER", ncInt, params.MAX_ITER);
      dataFile.putAtt("PSO_ITER", ncInt, params.PSO_ITER);
      dataFile.putAtt("requiredConverged", ncInt, params.requiredConverged);
      dataFile.putAtt("tolerance", ncDouble, params.tolerance);
      dataFile.putAtt("seed", ncInt, params.seed);
      dataFile.putAtt("run", ncInt, params.run);
      dataFile.putAtt("parallel", ncByte, (signed char)params.parallel);

      return cudaSuccess;
    }
    catch (const exceptions::NcException& e) {
      std::cerr << "NetCDF Error in writeTrajectoryDataNetCDF4: " << e.what()
                << std::endl;
      return cudaErrorUnknown;
    }
  }
#endif

  cudaError_t
  writeTrajectoryData(double* hostTrajectoryCoords,
                      double* hostTrajectoryFval,
                      double* hostTrajectoryGrad,
                      int8_t* hostStatus,
                      double* hostAlpha,
                      const OptimizationParams& params,
                      int DIM,
                      std::string_view filename)
  {
    if (endsWith(filename, ".tsv")) {
      return writeTrajectoryDataTSV(hostTrajectoryCoords,
                                    hostTrajectoryFval,
                                    hostTrajectoryGrad,
                                    hostStatus,
                                    hostAlpha,
                                    params,
                                    DIM,
                                    filename);
    } else if (endsWith(filename, ".nc")) {
#ifdef ZEUS_HAS_NETCDF4
      return writeTrajectoryDataNetCDF4(hostTrajectoryCoords,
                                        hostTrajectoryFval,
                                        hostTrajectoryGrad,
                                        hostStatus,
                                        hostAlpha,
                                        params,
                                        DIM,
                                        filename);
#else
      std::cerr
        << "Error: NetCDF4 support was disabled at build time. Cannot write "
        << filename << std::endl;
      return cudaErrorUnknown;
#endif
    } else {
      std::cerr << "Error: unsupported trajectory file extension for "
                << filename << ". Use .tsv or .nc" << std::endl;
      return cudaErrorUnknown;
    }
  }

} // end namespace util
