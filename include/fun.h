// fun.h — exposes the standard Zeus benchmark callable types (util::Rosenbrock,
// util::Rastrigin, util::Ackley, util::Himmelblau, util::GoldsteinPrice) to
// the test suite.  The implementations live in examples/*.hpp;
// the examples/ directory must be on the include path (see CMakeLists.txt).
#pragma once
#include "rosenbrock.hpp"
#include "rastrigin.hpp"
#include "ackley.hpp"
#include "himmelblau.hpp"
#include "goldstein_price.hpp"
