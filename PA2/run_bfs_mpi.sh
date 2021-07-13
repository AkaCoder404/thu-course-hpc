#!/bin/bash

# run on 1 machine * 2 process * 14 threads with process binding, feel free to change it!

export OMP_NUM_THREADS=28
#export OMP_PROC_BIND=true
#export OMP_PLACES=cores

srun -N 1 -n 1 ./bfs_omp_mpi $*

