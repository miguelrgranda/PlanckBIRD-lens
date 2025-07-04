#!/bin/bash
#
#SBATCH --job-name=LiteBIRD_Planck_s5_d10
#SBATCH --output=LiteBIRD_Planck_s5_d10.out
#SBATCH --error=LiteBIRD_Planck_s5_d10.err
#SBATCH --ntasks=4
#SBATCH --partition=wncompute_astro
#SBATCH --reservation=cosmo
# SBATCH --nodelist=wncompute018
#SBATCH --exclude=wncompute0[09-14]
#SBATCH --cpus-per-task=25
#SBATCH --nodes=1
#SBATCH --time=24:00:00
# From here the job starts

# Add an error fil to put there all the trash from HWLOC.
module use --append /gpfs/hpc_apps/EL9/INTEL/modules/all
module load OpenMPI/4.1.6-GCC-13.2.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Add ini files for the individual experiments.
export ini_1=ini_files/LiteBIRD_s5_d10.ini
export ini_2=ini_files/Planck_s5_d10.ini
export ini_c=ini_files/LiteBIRD_Planck_s5_d10.ini

export qe=True

# Run the simulation for the individual experiments.
mpirun -n ${SLURM_NPROCS} python3 combination.py $ini_c

# Filter the individual experiments and the combination.
mpirun -n ${SLURM_NPROCS} python3 filtering.py $ini_1
mpirun -n ${SLURM_NPROCS} python3 filtering.py $ini_2
mpirun -n ${SLURM_NPROCS} python3 filtering.py $ini_c

# Reconstruct the individual experiments and the combination.
mpirun -n ${SLURM_NPROCS} python3 reconstruction.py $ini_1 $qe
mpirun -n ${SLURM_NPROCS} python3 reconstruction.py $ini_2 $qe
mpirun -n ${SLURM_NPROCS} python3 reconstruction.py $ini_c $qe