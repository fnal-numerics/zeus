# Fitting Example C++ Project

This project demonstrates a C++ class with a function call operator member template that takes 4 arguments of type `T`. It is set up to build with Apple Clang using CMake.

## Build Instructions

1. Open a terminal in this directory.

2. Run:

   ```bash
   mkdir -p build && cd build
   cmake ..
   make
   ```

3. Run the executable:

   ```bash
   ./fitting_example
   ```

## Files

- `LogLikelihood.h` / `LogLikelihood.cpp`: Contains the template class and implementation.
- `main.cpp`: Example usage and test.
- `CMakeLists.txt`: Build configuration.

## Requirements

- CMake >= 3.16
- Apple Clang (default on macOS)
