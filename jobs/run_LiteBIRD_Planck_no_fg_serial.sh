#!/bin/bash
#
#SBATCH --job-name=LiteBIRD_Planck_no_fg_serial
#SBATCH --output=LiteBIRD_Planck_no_fg_serial.out
# SBATCH --error=LiteBIRD_Planck_no_fg_serial.err
#SBATCH --ntasks=1
#SBATCH --partition=wncompute_astro
#SBATCH --reservation=cosmo
# SBATCH --nodelist=wncompute016
#SBATCH --cpus-per-task=50
#SBATCH --nodes=1
#SBATCH --time=10:00:00
# From here the job starts


export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Add ini files for the individual experiments.
export ini_1=ini_files/LiteBIRD_no_fg.ini
export ini_2=ini_files/Planck_no_fg.ini
export ini_c=ini_files/LiteBIRD_Planck_no_fg.ini

export qe=False

# Reconstruction for the individual experiments and the combination.
# (serial part)
python3 reconstruction.py $ini_1 $qe
python3 reconstruction.py $ini_2 $qe
python3 reconstruction.py $ini_c $qe
