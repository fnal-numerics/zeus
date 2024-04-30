# HPC Research Project for Optimization


Make sure to have GPU Driver installed:   
```bash
nvidia-smi
```

Must have data file named "randomNumbers.bin", if not available, 
then compile and execute:   
```bash
g++ -o generator data-generator.cpp && ./generator
```

then once random numbers are generated, the CUDA file can be compiled and executed using on Wahab Cluster:
```bash
crun nvcc -o exe multipoint-optimization.cu && ./exe
```
or   
```bash
nvcc -o exe multipoint-optimization.cu && ./exe
```

