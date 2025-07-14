#pragma once

#include <type_traits>
#include <array>
#include <tuple>

//-----------------------------------------------------------------------------
// 1) array_size: extract N from std::array<T,N>
template<typename> struct array_size;  // primary left undefined

template<typename T, std::size_t N>
struct array_size<std::array<T,N>> {
  static constexpr std::size_t value = N;
};

//-----------------------------------------------------------------------------
// 2) fn_traits_operator: grab the type of F::operator()
//     e.g. if F is struct Foo { double operator()(std::array<double,3>) const; }
//     then fn_traits_operator<F>::type == double (Foo::*)(std::array<double,3>) const
template<typename F>
struct fn_traits_operator {
  using type = decltype(&F::operator());
};

//-----------------------------------------------------------------------------
// 3) fn_traits: introspect a pointer-to-member signature R (C::*)(A) const
//    This is where we pull out “arity” and “the argument type”.
template<typename FuncPtr> 
struct fn_traits;  // primary left undefined

// Specialize for the one-argument, const-member form:
template<typename C, typename R, typename A>
struct fn_traits<R (C::*)(A) const> {
  static constexpr std::size_t arity     = 1;  // exactly one parameter
  using result_type                    = R;  // return type
  using arg0_type                       = A;  // the single parameter’s type
};

//-----------------------------------------------------------------------------
// 4) fn_traits_f: your convenient façade
//     “Give me the fn_traits of your operator() pointer.”
template<typename F>
using fn_traits_f = fn_traits<typename fn_traits_operator<F>::type>;

//-----------------------------------------------------------------------------
// 5) is_array_callable: SFINAE-check that F accepts std::array<Scalar,N>
template <class F, class Scalar, std::size_t N, class = void>
struct is_array_callable : std::false_type {};

template <class F, class Scalar, std::size_t N>
struct is_array_callable<
  F, Scalar, N,
  std::void_t<
    decltype( std::declval<F>()(
      std::declval<const std::array<Scalar,N>&>()
    ) )
  >
> : std::true_type {};

// Helper variable‐template
template <class F, class Scalar, std::size_t N>
inline constexpr bool is_array_callable_v =
  is_array_callable<F,Scalar,N>::value;

//-----------------------------------------------------------------------------
// 6) objective_array concept for Zeus
//    - must have exactly one argument
//    - that argument must be std::array<double,N> for some N
//    - calling f(x) must yield something convertible to double
template<typename F>
concept objective_array =
  (fn_traits_f<F>::arity == 1)
  && is_array_callable_v<
       F,
       double,
       array_size<typename fn_traits_f<F>::arg0_type>::value
     >;

