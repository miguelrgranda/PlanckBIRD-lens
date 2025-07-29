# How to execute the pipeline in a HPC (slurm)

To execute the full pipeline (except cosmological parameter estimation and delensing) in a HPC you should execute main.sh.

To do that, first you should allow execution permission to the file main.sh and then run it locally (without submitting with sbatch):

`chmod +x main.sh`
`./main.sh`

For running the MCMC with cobaya for each of the experiments, execute run_Cobaya_{experiment}_r_minus_0.005_Hartlap.sh.
