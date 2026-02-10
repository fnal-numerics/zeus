#pragma once
#include <type_traits>

namespace zeus {
  /// Primary template for function traits extraction (intentionally undefined).
  /// Specializations below provide compile-time introspection of callable types
  /// to extract arity (dimensionality) and argument types for Zeus optimization.
  template<typename F>
  struct fn_traits;

  /// Specialization for templated callable classes (e.g., Foo<N>, Gaussian<N>).
  /// Extracts dimensionality N from class templates that accept std::size_t as template parameter.
  /// Used for objective functions defined as: template<std::size_t N> class Functor { T operator()(std::array<T,N>) }.
  template< template<std::size_t> class Functor, std::size_t N >
  struct fn_traits< Functor<N> > {
    static constexpr std::size_t arity = N;  ///< Problem dimensionality
    using arg = std::array<double, N>;       ///< Argument type for the objective function
  };
}