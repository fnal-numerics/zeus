#!/bin/bash

# Environment setup for global-optimizer-gpu on Perlmutter
# This script is sourced before builds and tests.

echo "⚙️  Setting up Perlmutter environment..."

# Load required modules
module swap PrgEnv-gnu PrgEnv-nvidia
module load cray-hdf5
module load cray-netcdf
module load cmake

export CC=cc
export CXX=CC
export CUDAHOSTCXX=CC

# Ensure Cray PE libraries (e.g. libfabric) are findable when binaries are
# run directly rather than through srun/PALS.
export LD_LIBRARY_PATH=/opt/cray/libfabric/1.22.0/lib64:${LD_LIBRARY_PATH}

echo "✅ Environment configured: CC=$CC, CXX=$CXX"
