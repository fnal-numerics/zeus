#!/bin/bash

# Environment setup for global-optimizer-gpu on Perlmutter
# This script is sourced before builds and tests.

echo "⚙️  Setting up Perlmutter environment..."

# Load required modules
module load cmake/3.24.3
module load gcc/12.2.0
module load cudatoolkit

# Unset Cray wrappers to use standard GCC/G++
unset CC
unset CXX
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

echo "✅ Environment configured: CC=$CC, CXX=$CXX"
