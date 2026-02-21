# Zeus Testing Suite

Our test suite is using catch2 to test to carry out integral, and unit testing for the Zeus library

## Testing so far includes: 
- vector addition
- vector subtraction

- fun values of origin, minimum, gradient for the following objectives 
  - Ackley
  - Rastrigin
  - Rosenbrock

- all operator overloads for automatic differentiation

- algorithmic tests
  - pso
  - bfgs

## Compile-Time Failure Tests

These tests verify that invalid objective functions are properly rejected at compile time:

- **`bad_objectives.cu`**: Tests static_assert-based compile-time checks for invalid objective functions
  - Validates that objectives with wrong signatures are rejected
  - Validates that non-templated objectives are rejected
  
- **`bad_objectives_concepts.cu`**: Tests C++20 concept-based compile checks for the `ZeusObjective` concept
  - Verifies concept violations produce clear error messages
  - Tests three common error patterns:
    1. Non-templated functions (missing dual::DualNumber support)
    2. Wrong return type (not returning T)
    3. Wrong argument signature (not taking std::array<T, DIM>)
  
Both files are expected to **fail compilation**. CMake's `try_compile()` verifies they do not compile, ensuring the type safety mechanisms work correctly.
