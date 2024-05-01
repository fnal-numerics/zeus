# HPC Research Project for Optimization


Make sure to have GPU Driver installed:   
```bash
nvidia-smi
```

No need for input file, because we are generating the random doubles on the GPU to utilize their power. The CUDA file can be compiled and executed using on Wahab Cluster:
```bash
crun nvcc -o exe multipoint-optimization.cu && ./exe
```
or   
```bash
nvcc -o exe multipoint-optimization.cu && ./exe
```
on a local machine.

