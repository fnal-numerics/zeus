#include "bfgs.cuh"  // bfgs::optimizeKernel
#include "duals.cuh" // dual::DualNumber
#include <array>

constexpr int DIM = 2;

// not callable on std::array<double,DIM>
struct BadObjective1 {
  __device__ double
  operator()(const std::array<int, DIM>& x) const
  {
    return 0.0;
  }
};

// callable on double, but not templated for DualNumber
struct BadObjective2 {
  __device__ double
  operator()(const std::array<double, DIM>& x) const
  {
    return x[0] + x[1];
  }
};

void
instantiate_bad1()
{
  curandState* states = nullptr;
  double* d_pso = nullptr;
  double* d_results = nullptr;
  Result<DIM>* d_out = nullptr;

  // should trigger static_assert -- wrong signature for double)
  bfgs::optimizeKernel<BadObjective1, DIM, 128><<<1, 128>>>(BadObjective1{},
                                                            0.0,
                                                            1.0,
                                                            d_pso,
                                                            d_results,
                                                            nullptr,
                                                            /*N=*/1,
                                                            /*MAX_ITER=*/1,
                                                            /*reqConv=*/1,
                                                            /*tol=*/1e-6,
                                                            d_out,
                                                            states);
}

void
instantiate_bad2()
{
  curandState* states = nullptr;
  double* d_pso = nullptr;
  double* d_results = nullptr;
  Result<DIM>* d_out = nullptr;

  // this should pass the first check but trigger the static_assert (no
  // DualNumber overload).
  bfgs::optimizeKernel<BadObjective2, DIM, 128><<<1, 128>>>(BadObjective2{},
                                                            0.0,
                                                            1.0,
                                                            d_pso,
                                                            d_results,
                                                            nullptr,
                                                            /*N=*/1,
                                                            /*MAX_ITER=*/1,
                                                            /*reqConv=*/1,
                                                            /*tol=*/1e-6,
                                                            d_out,
                                                            states);
}

int
main()
{
  instantiate_bad1();
  instantiate_bad2();
  return 0;
}
