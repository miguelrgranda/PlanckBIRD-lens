#!/bin/bash
#SBATCH --job-name=Likelihood_Cobaya_Planck_no_0.005_H
#SBATCH --output=Likelihood_Cobaya_Planck_no_0.005_H.out
#SBATCH --error=Likelihood_Cobaya_Planck_no_0.005_H.err
#SBATCH --partition=wncompute_astro
#SBATCH --reservation=cosmo
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2

module load PYTHON/3.11.1

module load INTEL/2023_hpc
source  /gpfs/hpc_apps/EL9/INTEL/2023_hpc/compiler/latest/env/vars.sh

module use --append /gpfs/hpc_apps/EL9/INTEL/modules/all
module load OpenMPI/4.1.6-GCC-13.2.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mpirun -n 8 python3.11 PhiPhiCobaya_Hartlap_General_TF.py /input/dir/ /output/path/ /path/to/covariance_matrix/ 0 0.005 0 /path/to/Initial_Cov.txt 1
mpirun -n 8 python3.11 PhiPhiCobaya_Hartlap_General_TF.py /input/dir/ /output/path/ /path/to/covariance_matrix/ 0 0.005 1 /path/to/Initial_Cov.txt 1
