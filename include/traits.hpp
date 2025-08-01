#pragma once
#include <type_traits>

namespace zeus {
  // primary, left undefined
  template<typename F>
  struct fn_traits;

  // class‐template functor: any `template<std::size_t> class Foo`
  template< template<std::size_t> class Functor, std::size_t N >
  struct fn_traits< Functor<N> > {
    static constexpr std::size_t arity = N;
    using arg = std::array<double, N>;
  };

  // free/static function pointer
  //    R (*)( std::array<double,N> const& )
  template<typename R, std::size_t N>
  struct fn_traits< R (*)( std::array<double, N> const& ) > {
    static constexpr std::size_t arity = N;
    using arg = std::array<double, N>;
  };

  // scalar free functions R(*)(R)
  template<typename R>
  struct fn_traits< R (*)(R) > {
    static constexpr std::size_t arity = 1;
    using arg = std::array<R,1>;
  };

}


