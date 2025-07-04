#!/bin/bash
#
#SBATCH --job-name=Planck_s5_d10
#SBATCH --output=Planck_s5_d10.out
#SBATCH --ntasks=1
#SBATCH --partition=wncompute_astro
#SBATCH --reservation=cosmo
# SBATCH --nodelist=wncompute016
#SBATCH --cpus-per-task=25
#SBATCH --nodes=1
#SBATCH --time=01:00:00
# From here the job starts

module use --append /gpfs/hpc_apps/EL9/INTEL/modules/all
module load OpenMPI/4.1.6-GCC-13.2.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

export ini=ini_files/Planck_s5_d10.ini

# mpirun -n ${SLURM_NPROCS} python3 simulation.py $ini
python3 simulation.py $ini
# mpirun -n ${SLURM_NPROCS} python3 filtering.py $ini