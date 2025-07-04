# PlanckBIRD-lens

This repository contains all the tools for forecasting the lensing sensitivity of Planck + LiteBIRD using an end-to-end pipeline. It includes:

* Simulation of the frequency maps for Planck and LiteBIRD.
* Component-separation using HILC from frequency maps coming from Planck and LiteBIRD.
* Lensing reconstruction at the map and power spectrum level.
* Cosmological parameter estimation using a lensing-only likelihood.
* Semi-analytical delensing efficiency estimation for constraining the tensor-to-scalar ratio.

## Authors

Miguel Ruiz-Granda (ruizm@ifca.unican.es), Christian Gimeno-Amo (gimenoc@ifca.unican.es).

## Citation

If you use any of this code, please cite: M. Ruiz-Granda et al, "LiteBIRD Science Goals and Forecasts: Improved full-sky reconstruction of the gravitational lensing potential through the combination of Planck and LiteBIRD data".

Cite other papers for the dependencies?

## Dependencies

Here are the dependencies needed to run the pipeline. We also include the version

* lensquest (modified from version )
* lensingbiases (modified from version ) 
* cmblensplus 
* pysm3
* healpy
* mpi4py
* pymaster
* lenspyx
* cobaya
* Others: scipy, numpy, pickle, astropy, tqdm, matplotlib.
	
### Modification in lensquest

### Modification in lensingbiases

## General description:
	
* combination.py -- simulates frequency maps using the experiments.yalm information for Planck and LiteBIRD. It performs component separation on those frequency maps for Planck, LiteBIRD and Planck + LiteBIRD.

* filtering.py -- filters the cleaned maps using an Harmonic C-inverse filtering. This is a step needed for running the Quadratic Estimators (QE).

* reconstruction.py -- reconstruct the lensing potential using the Quadratic Estimators formalism. It also includes tools for debiasing the lensing power spectrum, especially for the N0 and N1 bias estimation.

* cosmological_parameter_estimation.py -- runs a MCMC to estimate the constraints on H0-sigma8-Omegam using a lensing-only likelihood.

### Jupyter-notebooks

* [notebooks/HILC_figures_publish.ipynb](https://github.com/miguelrgranda/PlanckBIRD-lens/blob/main/notebooks/HILC_figures_publish.ipynb) -- generates plots of the HILC residuals for the different experiments and foreground complexities.
    
* [notebooks/bias_estimation_publish.ipynb](https://github.com/miguelrgranda/PlanckBIRD-lens/blob/main/notebooks/bias_estimation_publish.ipynb) -- generates plots of the N0 bias, N1 bias, Monte Carlo (MC) normalization correction, Mean-Field (MF), and the Minimum Variance (MV) weights.

* [notebooks/Lensing_map_publish.ipynb](https://github.com/miguelrgranda/PlanckBIRD-lens/blob/main/notebooks/Lensing_map_publish.ipynb) -- generates plots of the lensing maps for the different experiments and for the simple foregrounds.

* [notebooks/qcl_lensing_power_spectrum_publish.ipynb](https://github.com/miguelrgranda/PlanckBIRD-lens/blob/main/notebooks/qcl_lensing_power_spectrum_publish.ipynb) -- generates plots of the lensing bandpowers for the different experiments and foreground complexities. It also computes the lensing Signal-to-noise ratio (SNR) from the lensing bandpowers of the 400 simulations.

* [notebooks/cosmological_parameter_estimation_publish.ipynb](https://github.com/miguelrgranda/PlanckBIRD-lens/blob/main/notebooks/cosmological_parameter_estimation_publish.ipynb) -- plots the constrains on $H_0$, $\Omega_\mathrm{m}$ and $\sigma_8$ from the MCMC chains using the lensing-only likelihood.
  
* [notebooks/delensing_Planck_LiteBIRD_publish.ipynb](https://github.com/miguelrgranda/PlanckBIRD-lens/blob/main/notebooks/delensing_Planck_LiteBIRD_publish.ipynb) -- performs a semi-analytical delensing eficiency estimation and estimates the improvement on the constraints on the tensor-to-scalar ratio constraints.


