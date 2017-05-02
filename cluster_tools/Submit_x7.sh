#!/bin/bash
#SBATCH --qos=medium
#SBATCH --job-name=Maya_7
#SBATCH --output=ms_x7_%j.out
#SBATCH --error=ms_x7_%j.err
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

source activate py36

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ../Experiments/
srun -n $SLURM_NTASKS python mayasim_X7_scan_r_es_and_r_agg.py 0