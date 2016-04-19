#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=MayaSim
#SBATCH --output=ms_%j.out
#SBATCH --error=ms_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

srun -n $SLURM_NTASKS python Mayasim_testing.py

