#!/bin/bash
#
#SBATCH --job-name=Planck_s5_d10
#SBATCH --output=Planck_s5_d10.out
#SBATCH --ntasks=1
#SBATCH --partition=wncompute_astro
#SBATCH --reservation=cosmo
# SBATCH --nodelist=wncompute016
#SBATCH --cpus-per-task=40
#SBATCH --nodes=1
#SBATCH --time=24:00:00
# From here the job starts


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export ini=ini_files/Planck_s5_d10.ini
export qe=False

python3 reconstruction.py $ini $qe
