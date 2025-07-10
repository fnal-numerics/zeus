#pragma once

#include <cstdint>           // for uint64_t
#include <curand_kernel.h>   // for curandState
#include <cstdio> 	    // for printf
#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>

struct Convergence {
    int actual;
    int claimed;
    int surrendered;
    int stopped;
};

template<int DIM>
struct Result {
    int idx;
    int status; // 1 if converged, else if stopped_bc_someone_flipped_the_flag: 2, else 0
    double fval; // function value
    double gradientNorm;
    double coordinates[DIM];
    int iter;
    Convergence c;
};

namespace util {

bool
askUser2saveTrajectories();

void set_stack_size();

double
calculate_euclidean_error(const std::string fun_name,
                          const double* coordinates,
                          const int dim);



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
                     const int stopped);

template <int DIM>
Convergence
dump_data_2_file(const Result<DIM>* h_results,
                 const std::string fun_name,
                 const int N,
                 const int PSO_ITER,
                 const int run)
{
    Convergence result;

    std::string tab = "\t";
    int actually_converged = 0;
    int countConverged = 0, surrender = 0, stopped = 0;
    for (int i = 0; i < N; ++i) {
        //outfile << fun_name << tab << run << tab << i << tab << std::scientific; 
        if (h_results[i].status == 1) {
            countConverged++;
            //outfile << 1 << tab;
            double error = calculate_euclidean_error(fun_name,  h_results[i].coordinates, DIM);
            if(error < 0.5) {actually_converged++;}
            //outfile << h_results[i].iter << tab << h_results[i].fval << tab << h_results[i].gradientNorm;
            //for(int d = 0; d < DIM; ++d) { outfile << "\t"<< h_results[i].coordinates[d]; }
            //outfile << std::endl;
        } else if(h_results[i].status == 2) { // particle was stopped early
            stopped++;
            //outfile << 2 << tab;
            //printf("Thread %d was stopped early (iter=%d)\n", i, h_results[i].iter);
        } else {
            surrender++;
            //outfile << 0 << tab;
        }
        //outfile << h_results[i].iter << tab << h_results[i].fval << tab << h_results[i].gradientNorm;
        //for(int d = 0; d < DIM; ++d) { outfile << "\t"<< h_results[i].coordinates[d]; }
        //outfile << std::endl;
    }
    result.actual = actually_converged;
    result.claimed = countConverged;
    result.surrendered = surrender;
    result.stopped = stopped;
    return result;
    //std::cout << "\ndumped data 2 "<< filename << "\n"<<countConverged <<" converged, "<<stopped << " stopped early, "<<surrender<<" surrendered\n"; 
    //printf("\ndumped data 2 %s\n%d converged, %d stopped early, %d surrendered\n",filename.c_str(),countConverged, stopped, surrender);
}

// https://xorshift.di.unimi.it/splitmix64.c
// Very fast 64-bit mixer — returns a new 64-bit value each time.
__device__ inline uint64_t splitmix64(uint64_t &x);


// return a random double in [minVal, maxVal)
__device__ inline double random_double(uint64_t &state,
                                       double minVal,
                                       double maxVal);

__device__ double dot_product_device(const double* a, const double* b, int size);


__device__ void outer_product_device(const double* v1, const double* v2, double* result, int size);

template<int DIM>
__device__ double calculate_gradient_norm(const double* g) {
    double grad_norm = 0.0;
    for (int i = 0; i < DIM; ++i) {
        grad_norm += g[i] * g[i];
    }
    return sqrt(grad_norm);
}



template<int DIM>
__device__ void compute_search_direction(double* p,const double*  H,const double* g) {
    for (int i = 0; i < DIM; i++) {
        double sum=0.0;
        for (int j = 0; j < DIM; j++) {
           sum += H[i * DIM + j] * g[j]; // i * dim + j since H is flattened arr[]
        }
    p[i] = -sum;
    }
}

// wrap kernel definitions extern "C" block so that their symbols are exported with C linkage
extern "C" {
__device__  void vector_add(const double* a, const double* b, double* result, int size);


__device__  void vector_scale(const double* a, double scalar, double* result, int dim);
}// end extern C

__device__ void initialize_identity_matrix(double* H, int dim);


__device__ bool valid(double x);


__device__ double pow2(double x);


__device__ void initialize_identity_matrix_device(double* H, int n);

template<int DIM>
__device__ void matrix_multiply_device(const double* A, const double* B, double* C);

// BFGS update with compile-time dimension
template<int DIM>
__device__ void bfgs_update(double* H, const double* s, const double* y, double sTy) {
    if (::fabs(sTy) < 1e-14) return;
    double rho = 1.0 / sTy;

    // Compute H_new element-wise without allocating large temporary matrices.
    // H_new = (I - rho * s * y^T) * H * (I - rho * y * s^T) + rho * s * s^T
    double H_new[DIM * DIM];  // Temporary array (DIM^2 elements)

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            double sum = 0.0;
            for (int k = 0; k < DIM; k++) {
                // Compute element (i,k) of (I - rho * s * y^T)
                double A_ik = ((i == k) ? 1.0 : 0.0) - rho * s[i] * y[k];
                double inner = 0.0;
                for (int m = 0; m < DIM; m++) {
                    // Compute element (m,j) of (I - rho * y * s^T)
                    double B_mj = ((m == j) ? 1.0 : 0.0) - rho * y[m] * s[j];
                    inner += H[k * DIM + m] * B_mj;
                }
                sum += A_ik * inner;
            }
            // Add the rho * s * s^T term
            H_new[i * DIM + j] = sum + rho * s[i] * s[j];
        }
    }

    // Copy H_new back into H
    for (int i = 0; i < DIM * DIM; i++) {
        H[i] = H_new[i];
    }
}

// function to calculate scalar directional direvative d = g * p
__device__ double directional_derivative(const double *grad, const double *p, int dim);

__device__ double generate_random_double(curandState* state, double lower,double upper);


__global__ void setup_curand_states(curandState* states, uint64_t seed, int N);

template<typename Function, int DIM>
__device__ double line_search(double f0, const double* x, const double* p, const double* g, Function f){
    const double c1=0.3;
    double alpha=1.0;
    double ddir = dot_product_device(g,p,DIM);
    double xTemp[DIM];
    for(int i=0;i<20;i++){
        for(int j=0;j<DIM;j++){
            xTemp[j] = x[j] + alpha*p[j];
        }
        double f1 = f(xTemp);
        if(f1 <= f0 + c1*alpha*ddir) break;
        alpha *= 0.5;
    }
    return alpha;
}

__device__
double atomicMinDouble(double* addr, double val);


}// end namespace util
