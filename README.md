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

* [lensQUEST](https://github.com/miguelrgranda/lensquest) (modified version from the [original code](https://github.com/doicbek/lensquest) from Dominic Beck)
* [lensingbiases](https://github.com/miguelrgranda/lensingbiases) (modified version from the [original code](https://github.com/JulienPeloton/lensingbiases) from Julien Peloton) 
* cmblensplus v0.4
* pysm3 v3.4.0
* healpy v1.16.2
* mpi4py v3.0.3
* pymaster v1.4
* lenspyx v2.0.5
* cobaya
* Others: scipy v1.13.1, numpy v1.23.5, pickle v4.0, astropy v5.3, tqdm v4.64.1, matplotlib v3.7.1.
	
### Modification in lensQUEST

We have included the implementation of the Realization-dependent N0 computation using a semi-analytical approach. The function to call is lensquest.quest_norm_RD_iterSims(...), which uses OpenMP parallelization for significantly faster execution.

### Modification in lensingbiases

The N1 computation was modified to allow computing the N1 bias without needing to use a ini file. Aditionally, in this new version the power spectra noise is not calculated internally and now the observed angular power spectra is passed as a parameter. 

All the information is passed via the arguments of the python function LensingBiases.compute_n1_py(...), which computes the unnormalized N1 bias using OPENMP parallelization. We did not integrate the N0 computation accordingly in this modified version, because we are computing it using lensquest or MCN0 implementation in PlanckBIRD-lens.

### About cmblensplus

Installing cmblensplus can be complicated, and in reality is barely used:

* In ``filtering.py``, cmblensplus is imported only for the pixel-based filtering which is not used in our code and left there for comparison purposes. By commenting the line ''import curvedsky as cs``, you solve the import error. 

* In ``reconstruction.py`` only the python scripts ``analysis.py`` and ``binning.py`` from cmblensplus/utils are needed. There are two possible solutions:
	*  Installing cmblensplus/utils by creating a setup.py inside the cmblensplus folder and running in the terminal ``pip3 install .``:
 	```bash
  	#!/usr/bin/env python

	# Made by Miguel Ruiz Granda
	from distutils.core import setup
	
	setup(name='utils',
	      version='1.0',
	      packages=['utils'],
	     )
  	```
 	*    Copying ``analysis.py`` and ``binning.py`` to PlanckBIRD-lens directory and change the following lines in ``reconstruction.py``:
   	```python3
	from utils import analysis as ana --> import analysis as ana
	from utils import binning --> import binning
    	```
### Downloading Planck Galactic masks:

For running the filtering with the Planck Galactic masks, the file HFI_Mask_GalPlane-apo0_2048_R2.00.fits needs to be downloaded into the input directory:

```bash
wget -O HFI_Mask_GalPlane-apo0_2048_R2.00.fits "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
```

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


