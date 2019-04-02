#!/bin/bash

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export OMP_NUM_THREADS=1

f2py    --f90flags='-O3 -ftree-vectorize' -lgomp -c  f90routines.f90 -m f90routines
f2py3   --f90flags='-O3 -ftree-vectorize' -lgomp -c  f90routines.f90 -m f90routines
f2py3.5 --f90flags='-O3 -ftree-vectorize' -lgomp -c  f90routines.f90 -m f90routines

