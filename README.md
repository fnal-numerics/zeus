# HPC Research Project for Optimization


Make sure to have GPU Driver installed:   
```bash
nvidia-smi
```

No need for input file, because we are generating the random doubles on the GPU to utilize their power. The CUDA file can be compiled and executed using on the ODU Wahab HPC Cluster:
```bash
crun.cuda nvcc -o exe main.cu && ./exe -5.0 5.0 
```
or   
```bash
nvcc -o exe main.cu && ./exe -5.0 5.0
```
on a local machine.

