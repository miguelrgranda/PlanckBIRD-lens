#!/bin/bash
#
#SBATCH --job-name=class
#SBATCH --output=class.out
#SBATCH --ntasks=1
#SBATCH --partition=wncompute_astro
#SBATCH --reservation=cosmo
# SBATCH --nodelist=wncompute016
#SBATCH --cpus-per-task=30
#SBATCH --nodes=1
#SBATCH --time=08:00:00

./class base_2018_plikHM_TTTEEE_lowl_lowE_lensing.ini cl_ref.pre
