#!/bin/bash

# Environment setup for global-optimizer-gpu on Perlmutter
# This script is sourced before builds and tests.

echo "⚙️  Setting up Perlmutter environment..."

# Load required modules
module load cmake
module load gcc
module load cudatoolkit

# Unset Cray wrappers to use standard GCC/G++
unset CC
unset CXX
export CC=gcc
export CXX=g++
export CUDAHOSTCXX=g++

echo "✅ Environment configured: CC=$CC, CXX=$CXX"
