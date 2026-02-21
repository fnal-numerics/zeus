#pragma once
#include <cstddef>
#include <type_traits>

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
}