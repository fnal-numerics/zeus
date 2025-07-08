#pragma once
#include <curand_kernel.h>

namespace pso {

// kernel #1: initialize X, V, pBest; atomically seed gBestVal/gBestX
template<typename Function, int DIM>
__global__ void psoInitKernel(
    Function           func,
    double             lower,
    double             upper,
    double*            X,
    double*            V,
    double*            pBestX,
    double*            pBestVal,
    double*            gBestX,
    double*            gBestVal,
    int                N,
    uint64_t           seed,
    curandState*       states)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState localState = states[i];
    const double vel_range = (upper - lower) * 0.1;
    // const unsigned int seed = 1234u;
    //uint64_t counter = seed ^ (uint64_t)i;
    //uint64_t state = seed * 0x9e3779b97f4a7c15ULL + (uint64_t)i;
    //if (i==0) {
    //  printf(">> initKernel sees seed = %llu\n", (unsigned long long)seed);
    //}
    // init position & velocity
    for (int d = 0; d < DIM; ++d) {
        double rx = util::generate_random_double(&localState,lower, upper);
        double rv = util::generate_random_double(&localState,-vel_range, vel_range);

        X[i*DIM + d]      = rx;
        V[i*DIM + d]      = rv;
        pBestX[i*DIM + d] = rx;
    }

    // eval personal best
    double fval = Function::evaluate(&X[i*DIM]);
    pBestVal[i] = fval;

    // atomic update of global best
    double oldGB = util::atomicMinDouble(gBestVal, fval);
    if (fval < oldGB) {
        // we’re the new champion: copy pBestX into gBestX
        for (int d = 0; d < DIM; ++d)
            gBestX[d] = pBestX[i*DIM + d];
    }
    states[i] = localState; // next time we draw, we continue where we left off
}


// kernel #2: one PSO iteration (velocity+position update, personal & global best)
template<typename Function, int DIM>
__global__ void psoIterKernel(
    Function           func,
    double             lower,
    double             upper,
    double             w, // weight inertia
    double             c1, // cognitive coefficient
    double             c2, // social coefficient
    double*            X,
    double*            V,
    double*            pBestX,
    double*            pBestVal,
    double*            gBestX,
    double*            gBestVal,
    double*            traj,        // pass nullptr if not saving
    bool               saveTraj,
    int                N,
    int                iter,
    uint64_t           seed,
    curandState*       states)//,
    //Result<DIM>&        best) // store the best particle's results at each iteration
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    curandState localState = states[i];
    /*uint64_t state = seed;
    state = state * 6364136223846793005ULL + iter;   // mix in iteration
    state = state * 6364136223846793005ULL + (uint64_t)i;  // mix in thread idx
    */
    // update velocity & position
    for (int d = 0; d < DIM; ++d) {
        //uint64_t z1 = splitmix64(state);
        //uint64_t z2 = splitmix64(state);
        //double r1 = random_double(state,0.0,1.0 ); //util::generate_random_double(seed1, 0.0, 1.0);
        //double r2 = random_double(state, 0.0, 1.0); //util::generate_random_double(seed2, 0.0, 1.0);
        double r1 = util::generate_random_double(&localState, 0.0, 1.0);
        double r2 = util::generate_random_double(&localState, 0.0, 1.0);

        double xi = X[i*DIM + d];
        double vi = V[i*DIM + d];
        double pb = pBestX[i*DIM + d];
        double gb = gBestX[d];

        double nv = w*vi
                  + c1*r1*(pb - xi) // “cognitive” pull toward personal best
                  + c2*r2*(gb - xi); // “social” pull toward global best
        double nx = xi + nv;

        V[i*DIM + d] = nv;
        X[i*DIM + d] = nx;

        if (saveTraj) {
            // traj is laid out [iter][i][d]
            size_t idx = size_t(iter)*N*DIM + i*DIM + d;
            traj[idx] = nx;
        }
    }



    // evaluate at new position
    double fval = Function::evaluate(&X[i*DIM]);

    // personal best? no atomic needed, it's a private best position
    if (fval < pBestVal[i]) {
        pBestVal[i] = fval;
        for (int d = 0; d < DIM; ++d)
            pBestX[i*DIM + d] = X[i*DIM + d];
    }

    // global best?
    double oldGB = util::atomicMinDouble(gBestVal, fval);
    if (fval < oldGB) {
        for (int d = 0; d < DIM; ++d)
            gBestX[d] = X[i*DIM + d];
    }
    // best.coordinates = gBestX;
    // best.fval = gBestVal;
    /*printf("it %d gBestVal = %.6f  at gBestX = [",i,fval);
    for (int d = 0; d < DIM; ++d)
        printf(" %8.4f", gBestX[d]);
    printf(" ]\n");*/
    states[i] = localState; // next time we draw, we continue where we left off
}






}

