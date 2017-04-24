#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=Maya_ensemble
#SBATCH --output=ms_ensemble_%j.out
#SBATCH --error=ms_ensemble_%j.err
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ../
srun -n $SLURM_NTASKS python mayasim_X_ensemble.py 1
