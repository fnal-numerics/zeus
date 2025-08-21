#include <vector>
#include <iostream>
#include <string>
#include "zeus.cuh"
#include "duals.cuh"

#include "gaussian.hpp"
#include "nn.hpp"

#include <cstdlib>    // for rand(), srand()
#include <cmath>      // for exp()

double
square(double x)
{
  return x * 2.5;
}

struct Rosen {
  Rosen(Rosen const&) = delete;
  Rosen& operator=(Rosen const&) = delete; 
  
  template <class T, std::size_t N>
  __host__ __device__
  constexpr T operator()(std::array<T, N>const& x) const
  {
    T sum = T(0);
#pragma unroll
    for (int i = 0; i < N-1; ++i) {
      T t1 = T(1) - x[i];
      T t2 = x[i + 1] - x[i] * x[i];
      sum += t1 * t1 + T(100) * t2 * t2;
    }
    return sum;
  }
};

// templated Rastrigin on T (double or Dual) and N at compile time
  template<typename T, std::size_t N>
  __host__ __device__
  constexpr T rast(const std::array<T,N>& x) {
    T sum = T(10.0 * N);
    using namespace std;
  #pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      sum += x[i]*x[i]
             - T(10.0) * cos(2.0 * M_PI * x[i]);
    }
    return sum;
  }

template<typename T, std::size_t D>
T square2(std::array<T,D> const& x) {
    double sum = 0.0;
    for (std::size_t i = 0; i < D; ++i) {
        sum += x[i] * x[i];
    }
    return sum;
}

template<std::size_t N>
struct Foo {
   static_assert(N >= 2, "Foo<N> requires N >= 2 because it accesses x[1]");

   template <typename T>
    __host__ __device__
   T operator()(const std::array<T,N>& x) const {
     return T(0.5) * x[0] * x[0] + x[1];
   }
};

template<std::size_t N>
struct BukinN6 {
  static_assert(N >= 2, "BukinN6<N> requires N >= 2 because it accesses x[1]");

  template <typename T>
  __host__ __device__
  T operator()(const std::array<T, N>& x) const {
    using std::abs;
    using std::sqrt;
    const T X  = x[0];
    const T Y  = x[1];
    const T X2 = X * X;

    // scores = 100 * sqrt(abs(Y - 0.01 * X^2)) + 0.01 * abs(X + 10)
    const T term1 = sqrt(abs(Y - T(0.01) * X2));
    const T term2 = abs(X + T(10));

    return T(100) * term1 + T(0.01) * term2;
  }
};


int
main(int argc, char* argv[])
{
  if (argc != 4) {
    std::cerr << "Usage " << argv[0] << "<optimization> <bfgs> <run>\n";
    return 1;
  }
  size_t N = std::stoi(argv[1]);
  int bfgs = std::stoi(argv[2]);
  int run = std::stoi(argv[3]);
  
  std::vector ys{1.5, 2.5, 3.5};

  auto result = zeus::fmap(square, ys);
  for (auto val : result) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  util::set_stack_size();
  
#if(0)
  auto res10 = zeus::Zeus(Rosen{}, -5.12, 5.12, host, N, 10000, 5, 100, "rosenbrock", 1e-8, 42, 0);
  auto res3 = zeus::Zeus(Rosen{}, -5.12, 5.12, host, 1, 100, 0, 1, "rosenbrock", 1e-8, 42, 0);
  std::cout << "global minimum for 10d rosenbrock: " << res10.fval << std::endl;
  std::cout << "global minimum for 3d rosenbrock: " << res3.fval << std::endl;

  auto res5 = zeus::Zeus(Rast{}, -5.12, 5.12, host, N, 10000, 10, 100, "rastrigin", 1e-8, 42, 0);
  std::cout << "global minimum for 5d rastrigin: " << res5.fval << std::endl;
#endif
#if(1)
  // positive symmetric matrix
  // matrix of random numbers -> transpose to itself, divide by 2.
  //   
  //auto result2 = zeus::Zeus(rast,-5.0, 5.0,100,1000,100,10,"square",1e-6,42,0);

  BukinN6<2> b;

  auto bukin6 = Zeus(b,-15.0, 3.0, N, bfgs, 20, 100, "bukin6", 1e-8, 42, run);
  std::cout << "best result for bukin6: " << bukin6.fval  <<"\n";

Foo<2> f;
auto foo = Zeus(f,/*lower_bound=*/-20.0,/*upper_bound=*/20.0,/*optimization=*/N,
              /*bfgs_iterations=*/bfgs,/*pso_iterations=*/20,/*required_convergences=*/100,
             /*function_name=*/"foo",/*tolerance=*/1e-8,/*seed=*/42,/*index_of_run=*/run);
std::cout<< "best result: " << foo.fval <<  std::endl;

  constexpr std::size_t D = 4;
  using T = double;
  
  T off = T(0.5);
  std::array<std::array<T, D>, D> C;
  for (std::size_t i = 0; i < D; ++i) {
    for (std::size_t j = 0; j < D; ++j) {
      C[i][j] = (i == j ? T(1) + (D-1)*off : off);
    }
  }  

  Gaussian<D> g{C}; 

  std::cout << "running " << D << "d Gaussian minimization" << std::endl; 
  using namespace std::literals;
  auto res150 = zeus::Zeus(g, -5.00, 5.00, N, 10000, 10, 100, "gaussian"s, 1e-8, 42, run); 
  std::cout << "global minimum for " << D << "d Gaussian: " << res150.fval << std::endl;
#endif  
#if(0)
  constexpr size_t In = 5;
  constexpr size_t H = 15;
  constexpr size_t Out = 10;
  constexpr size_t P = NeuralNet<In,H,Out>::P;

  // toy training example
  std::array<double,In> x0;
  std::array<double,Out> y0;
  for(size_t i = 0; i < In; ++i)
    x0[i] = double(i);
  for(size_t k = 0; k < Out; ++k)
    y0[k] = (std::rand()/double(RAND_MAX) - 0.5) * 0.5;

  //construct the objective 
  // copies x0, y0 to device
  NeuralNet<In,H,Out> objective{x0,y0};

  std::cout << "training " << P << "d neural net..\n";
  // initialize theta0 on the host
  std::array<double,P> theta0;
  for(auto &v : theta0)
    v = (std::rand()/double(RAND_MAX) - 0.5) * 0.1;

  int PSOiters = 2;
  int requiredConverged = 10;
  double tol = 1e-6;

  auto res = zeus::Zeus(objective, theta0 ,-20.0, 20.0,host,N, bfgs, PSOiters, requiredConverged,"neural_net",tol, 42, run);
  std::cout<<"final loss: "<<res.fval<<"\n";
#endif
}
