#include "utils.cuh"
#include <string>
namespace util {

  bool
  askUser2saveTrajectories()
  {
    std::cout << "Save optimization trajectories? (y/n): ";
    char ans;
    std::cin >> ans;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    return (ans == 'y' || ans == 'Y');
  }

  void
  createOutputDirs(const std::string& path)
  {
    std::filesystem::create_directories(path);
  }

  void
  set_stack_size()
  {
    // logic to set the stact size limit to 65 kB per thread
    size_t currentStackSize = 0;
    cudaDeviceGetLimit(&currentStackSize, cudaLimitStackSize);
    // printf("Current stack size: %zu bytes\n", currentStackSize);
    size_t newStackSize = 64 * 1024; // 65 kB
    cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, newStackSize);
    if (err != cudaSuccess) {
      printf("cudaDeviceSetLimit error: %s\n", cudaGetErrorString(err));
      // return 1;
    }
  }

  cudaError_t
  writeTrajectoryData(double* hostTrajectory,
                      int N,
                      int MAX_ITER,
                      int DIM,
                      const std::string& fun_name,
                      const std::string& basePath)
  {
    // construct the directory path and create it.
    std::string dirPath = basePath + "/" + fun_name + "/" +
                          std::to_string(DIM) + "d/" +
                          std::to_string(MAX_ITER * N) + "/trajectories";
    std::filesystem::create_directories(dirPath);
    // createOutputDirs(dirPath);

    // the final filename.
    std::string filename = dirPath + "/" + std::to_string(MAX_ITER) + "it_" +
                           std::to_string(N) + ".tsv";

    std::ofstream stepOut(filename);
    stepOut << "OptIndex\tStep";
    for (int d = 0; d < DIM; d++)
      stepOut << "\tX_" << d;
    stepOut << "\n";
    stepOut << std::scientific << std::setprecision(17);
    for (int i = 0; i < N; i++) {
      for (int it = 0; it < MAX_ITER; it++) {
        stepOut << i << "\t" << it;
        for (int d = 0; d < DIM; d++) {
          stepOut << "\t"
                  << hostTrajectory[i * (MAX_ITER * DIM) + it * DIM + d];
        }
        stepOut << "\n";
      }
    }
    stepOut.close();
    return cudaSuccess;
  }

  double
  calculate_euclidean_error(const std::string fun_name,
                            const double* coordinates,
                            const int dim)
  {
    double sum_sq = 0.0;

    if (fun_name == "rosenbrock") {
      for (int i = 0; i < dim; i++) {
        double diff = coordinates[i] - 1.0;
        sum_sq += diff * diff;
      }
    } else if (fun_name == "goldstein_price") {
      // Goldstein–Price is only defined in 2D (minimum at (0, -1))
      if (dim != 2) {
        fprintf(stderr, "Error: goldstein_price only defined for dim = 2\n");
        return NAN;
      }
      double dx = coordinates[0] - 0.0;
      double dy = coordinates[1] - (-1.0);
      sum_sq = dx * dx + dy * dy;
    } else if (fun_name == "rastrigin" ||
               fun_name ==
                 "ackley") { // both rastrigin and ackley have the same
                             // coordinates for the global minimum
      for (int i = 0; i < dim; ++i) {
        sum_sq += coordinates[i] * coordinates[i];
      }
    }
    return std::sqrt(sum_sq);
  } // end calculate_euclidean_error

  void
  append_results_2_tsv(const int dim,
                       const int N,
                       const std::string fun_name,
                       float ms_init,
                       float ms_pso,
                       float ms_opt,
                       float ms_rand,
                       const int max_iter,
                       const int pso_iter,
                       const double error,
                       const double globalMin,
                       double* hostCoordinates,
                       const int idx,
                       const int status,
                       const double norm,
                       const int run,
                       const int claimed,
                       const int actual,
                       const int surrendered,
                       const int stopped)
  {
    std::string filename = "zeus_" + std::to_string(dim) + "d_results.tsv";
    std::ofstream outfile(filename, std::ios::app);

    bool file_exists = std::filesystem::exists(filename);
    bool file_empty =
      file_exists ? (std::filesystem::file_size(filename) == 0) : true;
    // std::ofstream outfile(filename, std::ios::app);
    if (!outfile.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
    }
    // if file is new or empty, let us write the header
    if (file_empty) {
      outfile
        << "fun\trun\tN\tclaimed\tactual\tsurrender\tstopped\tidx\tstatus\t"
           "bfgs_iter\tpso_iter\ttime\terror\tfval\tnorm";
      for (int i = 0; i < dim; i++)
        outfile << "\tcoord_" << i;
      outfile << std::endl;
    } // end if file is empty

    double time_seconds = std::numeric_limits<double>::infinity();
    if (pso_iter > 0) {
      time_seconds = (ms_init + ms_pso + ms_opt + ms_rand);
      // printf("total time = pso + bfgs = total time = %0.4f ms\n",
      // time_seconds);
    } else {
      time_seconds = (ms_opt + ms_rand);
      // printf("bfgs time = total time = %.4f ms\n", time_seconds);
    }
    outfile << fun_name << "\t" << run << "\t" << N << "\t" << claimed << "\t"
            << actual << "\t" << surrendered << "\t" << stopped << "\t" << idx
            << "\t" << status << "\t" << max_iter << "\t" << pso_iter << "\t"
            << time_seconds << "\t" << std::scientific << error << "\t"
            << globalMin << "\t" << norm << "\t";
    for (int i = 0; i < dim; i++) {
      outfile << hostCoordinates[i];
      if (i < dim - 1)
        outfile << "\t";
    }
    outfile << "\n";
    outfile.close();
    // printf("results are saved to %s", filename.c_str());
  } // end append_results_2_tsv

  // https://xorshift.di.unimi.it/splitmix64.c
  // Very fast 64-bit mixer — returns a new 64-bit value each time.
  __device__ inline uint64_t
  splitmix64(uint64_t& x)
  {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL); // 1 add
    z = (z ^ (z >> 30)) *
        0xbf58476d1ce4e5b9ULL; // 1 shift, 1 xor, 1 64x64 multiplier
    z = (z ^ (z >> 27)) *
        0x94d049bb133111ebULL; // 1 shift, 1, xor, 1 64x64 multiplier
    // printf("split");
    return z ^ (z >> 31); // 1 shift, 1 xor
  }

  // return a random double in [minVal, maxVal)
  __device__ inline double
  random_double(uint64_t& state, double minVal, double maxVal)
  {
    // get 64‐bit random int
    uint64_t z = splitmix64(state);
    // map high 53 bits into [0,1)
    double u =
      (z >> 11) *
      (1.0 /
       9007199254740992.0); // discard lower 11 bits, leaving mantissa width of
                            // IEEE double, then normalize integer into [0,1)
    // scale into [minVal, maxVal)
    return minVal + u * (maxVal - minVal);
  }

  __device__ double
  dot_product_device(const double* a, const double* b, int size)
  {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  __device__ void
  outer_product_device(const double* v1,
                       const double* v2,
                       double* result,
                       int size)
  {
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        int idx = i * size + j;
        if (idx < size * size) {
          result[idx] = v1[i] * v2[j];
        } else {
          printf("outer product out of bounds..\ndim=%d i*size+j=%d\n",
                 size,
                 i * size + j);
        }
      }
    }
  }

  extern "C" {
  __device__ void
  vector_add(const double* a, const double* b, double* result, int size)
  {
    for (int i = 0; i < size; ++i) {
      result[i] = a[i] + b[i];
    }
  }

  __device__ void
  vector_scale(const double* a, double scalar, double* result, int dim)
  {
    for (int i = 0; i < dim; ++i) {
      result[i] = a[i] * scalar;
    }
  }

  } // end extern C

  __device__ void
  initialize_identity_matrix(double* H, int dim)
  {
    for (int i = 0; i < dim; ++i) {
      for (int j = 0; j < dim; ++j) {
        H[i * dim + j] = (i == j) ? 1.0 : 0.0;
      }
    }
  }

  __device__ bool
  valid(double x)
  {
    if (isinf(x)) {
      return false;
    } else if (isnan(x)) {
      return false;
    } else {
      return true;
    }
  }

  __device__ double
  pow2(double x)
  {
    return x * x;
  }

  __device__ void
  initialize_identity_matrix_device(double* H, int n)
  {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        H[i * n + j] = (i == j) ? 1.0 : 0.0;
      }
    }
  }

  template <int DIM>
  __device__ void
  matrix_multiply_device(const double* A, const double* B, double* C)
  {
    for (int i = 0; i < DIM; ++i) {
      for (int j = 0; j < DIM; ++j) {
        double sum = 0.0;
        for (int k = 0; k < DIM; ++k) {
          sum += A[i * DIM + k] * B[k * DIM + j];
        }
        C[i * DIM + j] = sum;
      }
    }
  }

  // function to calculate scalar directional direvative d = g * p
  __device__ double
  directional_derivative(const double* grad, const double* p, int dim)
  {
    double d = 0.0;
    for (int i = 0; i < dim; ++i) {
      d += grad[i] * p[i];
    }
    return d;
  }

  __device__ double
  generate_random_double(curandState* state, double lower, double upper)
  {
    return lower + (upper + (-lower)) * curand_uniform_double(state);
  }

  __global__ void
  setup_curand_states(curandState* states, uint64_t seed, int N)
  {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N)
      return;
    curand_init(seed, idx, 0, &states[idx]);
  }

  __device__ double
  atomicMinDouble(double* addr, double val)
  {
    // reinterpret the address as 64‑bit unsigned
    unsigned long long* ptr = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old_bits = *ptr, assumed_bits;

    do {
      assumed_bits = old_bits;
      double old_val = __longlong_as_double(assumed_bits);
      // if the current value is already <= our candidate, nothing to do
      if (old_val <= val)
        break;
      // else try to swap in the new min value’s bit‐pattern
      unsigned long long new_bits = __double_as_longlong(val);
      old_bits = atomicCAS(ptr, assumed_bits, new_bits);
    } while (assumed_bits != old_bits);

    // return the previous minimum
    return __longlong_as_double(old_bits);
  }

} // end namespace util
