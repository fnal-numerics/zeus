# An Efficient Implementation of BFGS optimization algorithm with Automatic Differentiation using GPUs

Make sure to have NVIDIA GPU Device connected and Driver installed:   
```bash
nvidia-smi
```

## Option 1: Build + Compile using CMake
use the provided shell scripts to remove build directory, build the project and run the tests using: 
```bash
./shell/rebuildcompile_test_run.sh -5.12 5.12 25 131072
```
or on the ODU Wahab HPC Cluster:
```bash
./shell/rebuildcompile_test_run_onwahab.sh -5.12 5.12 25 131072
```

## Option 2: Manual Compilation
No need for input file, because we are generating the random doubles on the GPU to utilize their power. The CUDA file can be compiled and executed using:
```bash
nvcc -o exe main.cu && ./exe  -5.12 5.12 25 131072
```
or on the  ODU Wahab HPC Cluster:
```bash
crun.cuda nvcc -o exe main.cu && ./exe  -5.12 5.12 25 131072
```
