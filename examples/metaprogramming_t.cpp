#include <vector>
#include <iostream>
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
struct Rast {
  Rast(Rast const&) = delete;
  Rast& operator=(Rast const&) = delete;

  template<class T, std::size_t N>
  __host__ __device__
  constexpr T operator()(const std::array<T,N>& x) const {
    T sum = T(10.0 * N);
    using namespace std;
  #pragma unroll
    for (std::size_t i = 0; i < N; ++i) {
      sum += x[i]*x[i]
             - T(10.0) * cos(2.0 * M_PI * x[i]);
    }
    return sum;
  }
};

int
main(int argc, char* argv[])
{
  if (argc != 3) {
    std::cerr << "Usage " << argv[0] << "<optimization> <run>\n";
    return 1;
  }
  int N = std::stoi(argv[1]);
  int run = std::stoi(argv[2]);

  std::vector ys{1.5, 2.5, 3.5};

  auto result = zeus::fmap(square, ys);
  for (auto val : result) {
    std::cout << val << " ";
  }
  std::cout << std::endl;
  double host[N];
  for (int i = 0; i < N; i++) {
    host[i] = 333777.0;
  }
  util::set_stack_size();
  
#if(0)
  std::array<double,10> x10{};
  auto res10 = zeus::Zeus(Rosen{},x10, -5.12, 5.12, host, N, 10000, 5, 100, "rosenbrock", 1e-8, 42, 0);
  std::array<double, 3> x3{};
  auto res3 = zeus::Zeus(Rosen{},x3, -5.12, 5.12, host, 1, 100, 0, 1, "rosenbrock", 1e-8, 42, 0);
  std::cout << "global minimum for 10d rosenbrock: " << res10.fval << std::endl;
  std::cout << "global minimum for 3d rosenbrock: " << res3.fval << std::endl;

  std::array<double, 5> x5{};
  auto res5 = zeus::Zeus(Rast{},x5, -5.12, 5.12, host, N, 10000, 10, 100, "rastrigin", 1e-8, 42, 0);
  std::cout << "global minimum for 5d rastrigin: " << res5.fval << std::endl;
#endif
#if(0)
  // positive symmetric matrix
  // matrix of random numbers -> transpose to itself, divide by 2.
  //   
  constexpr std::size_t D = 150;
  using T = double;
  T off = T(0.5);
  std::array<std::array<T, D>, D> C;
  for (std::size_t i = 0; i < D; ++i) {
    for (std::size_t j = 0; j < D; ++j) {
      C[i][j] = (i == j ? T(1) + (D-1)*off : off);
    }
  }  

  Gaussian<D> g{C};
  std::array<T, D> x150{};
  x150.fill(T(3.7));
  T fx = g(x150);
  std::cout << "f(x) = " << fx << std::endl;
  std::cout << "running " << D << "d Gaussian minimization" << std::endl; 
  auto res150 = zeus::Zeus(g,x150, -5.00, 5.00, host, N, 10000, 10, 100, "gaussian", 1e-8, 42, run); 
  std::cout << "global minimum for " << D << "d Gaussian: " << res150.fval << std::endl;
#endif  
#if(1)
  constexpr size_t In = 5;
  constexpr size_t H = 20;
  constexpr size_t Out = 5;
  constexpr size_t P = NeuralNet<In,H,Out>::P;

  // toy training example
  std::array<double,In> x0;
  std::array<double,Out> y0;
  for(size_t i = 0; i < In; ++i)
    x0[i] = double(i);
  for(size_t k = 0; k < Out; ++k)
    y0[k] = (std::rand()/double(RAND_MAX) - 0.5) * 0.1;

  //construct the objective 
  // copies x0, y0 to device
  NeuralNet<In,H,Out> objective{x0,y0};

  std::cout << "training " << P << "d neural net..\n";
  // initialize theta0 on the host
  std::array<double,P> theta0;
  for(auto &v : theta0)
    v = (std::rand()/double(RAND_MAX) - 0.5) * 0.1;

  int PSOiters = 10;
  int requiredConverged = 10;
  double tol = 1e-8;

  auto res = zeus::Zeus(objective, theta0 ,-2.0, 2.0,host,N, 10000, PSOiters, requiredConverged,"neural_net",tol, 42, run);
  std::cout<<"final loss: "<<res.fval<<"\n";
#endif
}
