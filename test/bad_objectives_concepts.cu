/// Compile-time failure verification for C++20 concepts in Zeus
/// This file contains intentionally broken objective functions that violate
/// the ZeusObjective concept. It is expected to FAIL compilation, demonstrating
/// that the concept constraints properly catch common user errors at compile time.

#include "zeus.cuh"
#include <array>

constexpr int DIM = 2;

// ❌ ERROR CASE 1: Non-templated function (only works with double)
// This violates ZeusObjective because it cannot be called with dual::DualNumber
struct NotTemplated {
  static constexpr size_t arity = DIM;
  
  double operator()(const std::array<double, DIM>& x) const {
    return x[0] * x[0] + x[1] * x[1];
  }
  // Missing: template<typename T> T operator()(const std::array<T, DIM>&)
};

// ❌ ERROR CASE 2: Wrong return type
// This violates ZeusObjective because it returns int instead of T
struct WrongReturnType {
  static constexpr size_t arity = DIM;
  
  template<typename T>
  int operator()(const std::array<T, DIM>& x) const {
    return 42;  // Returns int instead of T - breaks CallableWithArray concept
  }
};

// ❌ ERROR CASE 3: Wrong argument signature
// This violates ZeusObjective because it doesn't take std::array<T, DIM>
struct WrongArgType {
  static constexpr size_t arity = DIM;
  
  template<typename T>
  T operator()(T x, T y) const {  // Should be: const std::array<T, DIM>&
    return x * x + y * y;
  }
};

// Test instantiation of concept violation #1
void
instantiate_not_templated()
{
  // Should fail: NotTemplated violates ZeusObjective<NotTemplated, 2>
  // Error: cannot call with std::array<dual::DualNumber, 2>
  auto result = zeus::Zeus(NotTemplated{}, 
                           -5.0, 5.0, 2, 10, 5, 1, 
                           "not_templated", 0.01, 42, 1);
}

// Test instantiation of concept violation #2
void
instantiate_wrong_return()
{
  // Should fail: WrongReturnType violates ZeusObjective<WrongReturnType, 2>
  // Error: return type is not convertible to T
  auto result = zeus::Zeus(WrongReturnType{}, 
                           -5.0, 5.0, 2, 10, 5, 1, 
                           "wrong_return", 0.01, 42, 1);
}

// Test instantiation of concept violation #3
void
instantiate_wrong_args()
{
  // Should fail: WrongArgType violates ZeusObjective<WrongArgType, 2>
  // Error: not callable with std::array<T, DIM>
  auto result = zeus::Zeus(WrongArgType{}, 
                           -5.0, 5.0, 2, 10, 5, 1, 
                           "wrong_args", 0.01, 42, 1);
}

int
main()
{
  // Each of these function calls should trigger a concept violation error
  instantiate_not_templated();
  instantiate_wrong_return();
  instantiate_wrong_args();
}
