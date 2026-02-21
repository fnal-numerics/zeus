#pragma once
#include <cstddef>
#include <type_traits>
#include <array>
#include <concepts>

namespace dual {
  struct DualNumber; // Forward declaration
}

namespace zeus {
  /// Primary template for function traits extraction (intentionally undefined).
  /// Specializations below provide compile-time introspection of callable types
  /// to extract arity (dimensionality) and argument types for Zeus
  /// optimization.
  template <typename...>
  using VoidT = void;

  template <typename F, typename = void>
  struct FnTraits {
    static constexpr std::size_t arity = 0;
  };

  // Specialization for types that have a static 'arity' member.
  template <typename F>
  struct FnTraits<F, VoidT<decltype(F::arity)>> {
    static constexpr std::size_t arity = F::arity;
  };

  // Fallback for templated functors without 'arity' member.
  template <template <std::size_t> class Functor, std::size_t N>
  struct FnTraits<Functor<N>> {
    static constexpr std::size_t arity = N;
  };

  /// Concept: Objective function callable with std::array<T, DIM> returning T
  /// This checks that the function is properly templated on the element type.
  template <typename F, typename T, std::size_t DIM>
  concept CallableWithArray = requires(F f, std::array<T, DIM> arr) {
    { f(arr) } -> std::convertible_to<T>;
  };

  /// Concept: Valid Zeus objective function
  /// 
  /// A Zeus objective must:
  /// 1. Be callable with std::array<double, DIM> and return double
  /// 2. Be callable with std::array<dual::DualNumber, DIM> and return dual::DualNumber
  /// 
  /// This ensures the objective is properly templated for automatic differentiation.
  /// 
  /// Example of a valid objective:
  /// @code
  /// template<typename T>
  /// T rosenbrock(const std::array<T, 2>& x) {
  ///   T t1 = T(1.0) - x[0];
  ///   T t2 = x[1] - x[0] * x[0];
  ///   return t1 * t1 + T(100.0) * t2 * t2;
  /// }
  /// @endcode
  template <typename F, std::size_t DIM>
  concept ZeusObjective = 
    CallableWithArray<F, double, DIM> &&
    CallableWithArray<F, dual::DualNumber, DIM>;
}