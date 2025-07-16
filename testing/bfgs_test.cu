#include <catch2/catch_all.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
using Catch::Approx;

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <string>

#include "bfgs.cuh"
#include "fun.h"
#include "utils.cuh"


TEST_CASE("bfgs::launch converges immediately for util::Rastrigin<2>", "[bfgs][optimize]") {
    constexpr int N = 1, DIM = 2;
    const double lower = -5.0, upper = 5.0;
    const int MAX_ITER = 10, requiredConverged = 1;
    const double tolerance = 1e-6;
    const uint64_t seed = 123;

    // 1) prepare PSO‐init at exact minimizer {0,0}
    double  hInit[DIM] = {0.0, 0.0};
    double* dPSOInit   = nullptr;
    cudaMalloc(&dPSOInit, N * DIM * sizeof(double));
    cudaMemcpy(dPSOInit, hInit, DIM * sizeof(double), cudaMemcpyHostToDevice);

    // 2) hostResults placeholder
    double* hostResults = new double[N];

    // 3) curand states
    float ms_rand = 0.0f;
    curandState* d_states = bfgs::initialize_states(N, int(seed), ms_rand);

    // 4) correctly typed args
    double*     deviceTrajectory = nullptr;
    float       ms_opt           = 0.0f;
    std::string fun_name         = "rastrigin-bfgs-test";

    // 5) invoke bfgs::launch<Function,DIM>
    auto best = bfgs::launch<util::Rastrigin<DIM>, DIM>(
        /*N*/                  N,
        /*pso_iter*/           0,
        /*MAX_ITER*/           MAX_ITER,
        /*upper*/              upper,
        /*lower*/              lower,
        /*pso_results_device*/ dPSOInit,
        /*hostResults*/        hostResults,
        /*deviceTrajectory*/   deviceTrajectory,
        /*requiredConverged*/  requiredConverged,
        /*tolerance*/          tolerance,
        /*save_trajectories*/  false,
        /*ms_opt*/             ms_opt,
        /*fun_name*/           fun_name,
        /*states*/             d_states,
        /*run*/                0,
        /*f*/                  util::Rastrigin<DIM>()
    );

    REQUIRE(best.status == 1);
    REQUIRE(best.iter   == 0);
    REQUIRE(best.fval   == Approx(0.0).margin(tolerance));
    for (int d = 0; d < DIM; ++d)
        REQUIRE(best.coordinates[d] == Approx(hInit[d]).margin(1e-6));

    // cleanup
    delete[] hostResults;
    cudaFree(dPSOInit);
    cudaFree(d_states);
}

