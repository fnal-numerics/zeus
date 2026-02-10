#pragma once
#include <array>
#include "matrix.hpp"

template <size_t In, size_t H, size_t Out>
struct NeuralNet {
  static constexpr size_t P = In*H + H + H*Out + Out;

  // host & device buffers
  Matrix<double> x_host{ In,1 };
  Matrix<double> x_dev{ In,1 };
  Matrix<double> y_host{ Out,1};
  Matrix<double> y_dev{ Out,1};

  const double* x_ptr = nullptr;
  const double* y_ptr = nullptr;

  // build host buffers from std::array, then push to device
  NeuralNet(const std::array<double,In>& x_,
           const std::array<double,Out>& y_)
    : x_host(In,1), x_dev(In,1),
      y_host(Out,1), y_dev(Out,1)
  {
    // fill host_data_ via operator()(i,0)
    for(size_t i=0; i<In; ++i)
      x_host(i,0) = x_[i];
    for(size_t k=0; k<Out; ++k)
      y_host(k,0) = y_[k];

    // copy into device buffers
    x_dev = x_host;
    y_dev = y_host;
  
    x_ptr = x_dev.device_data();
    y_ptr = y_dev.device_data();
  }

  // templated sigmoid, works on double or DualNumber
  template<typename T>
  __host__ __device__
  static T sigmoid(T t) {
    return T(1) / (T(1) + exp(-t));
  }

  template<typename T>
  __host__ __device__
  T operator()(const std::array<T,P>& theta) const {
    size_t idx = 0;
    T hidden[H];

    // first layer
    for(size_t j=0; j<H; ++j) {
      T sum = T(0);
      for(size_t i=0; i<In; ++i)
        
        sum += theta[idx++] * T(x_dev.data()[i]);
      sum += theta[idx++];                 // b1[j]
      hidden[j] = sigmoid(sum);
    }

    // second layer + mean squared loss
    T loss = T(0);
    for(size_t k=0; k<Out; ++k) {
      T sum = T(0);
      for(size_t j=0; j<H; ++j)
        sum += hidden[j] * theta[idx++];
      sum += theta[idx++];                 // b2[k]
      T y_hat = sigmoid(sum);
      T d     = y_hat - T(y_dev.data()[k]);
      loss   += d*d;
    }
    return loss / T(Out);
  }
};

