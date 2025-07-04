# PlanckBIRD-lens

This repository contains all the tools for forecasting the lensing sensitivity of Planck + LiteBIRD using an end-to-end pipeline. Including:
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
	
	* delensing_Planck_LiteBIRD.ipynb -- performs a semi-analytical delensing eficiency estimation and estimates the improvement on the constraints on the tensor-to-scalar ratio constraints.
	
	* cosmological_parameter_estimation.py -- runs a MCMC to estimate the constraints on H0-sigma8-Omegam using a lensing-only likelihood.

