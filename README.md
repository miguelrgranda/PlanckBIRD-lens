# PlanckBIRD-lens

This repository contains all the tools for forecasting the lensing sensitivity of *Planck* + *LiteBIRD* using an end-to-end pipeline. It includes:

* Simulation of the frequency maps for *Planck* and *LiteBIRD*.
* Component-separation using HILC from frequency maps coming from *Planck* and *LiteBIRD*.
* Lensing reconstruction at the map and power spectrum level.
* Cosmological parameter estimation using a lensing-only likelihood.
* Semi-analytical delensing efficiency estimation for constraining the tensor-to-scalar ratio.

## Authors

Miguel Ruiz-Granda (ruizm@ifca.unican.es), Christian Gimeno-Amo (gimenoc@ifca.unican.es).

## Citation

If you use any of this code, please cite: M. Ruiz-Granda *et al* 2025, "LiteBIRD Science Goals and Forecasts: Improved full-sky reconstruction of the gravitational lensing potential through the combination of Planck and LiteBIRD data".

## Dependencies

Here are the dependencies needed to run the pipeline. We also include the version

* python v3.9.0
* [lensQUEST](https://github.com/miguelrgranda/lensquest) (modified version from the [original code](https://github.com/doicbek/lensquest) from Dominic Beck)
* [lensingbiases](https://github.com/miguelrgranda/lensingbiases) (modified version from the [original code](https://github.com/JulienPeloton/lensingbiases) from Julien Peloton) 
* [cmblensplus](https://github.com/toshiyan/cmblensplus) v0.4
* pysm3 v3.4.0
* healpy v1.16.2
* mpi4py v3.0.3
* pymaster v1.4
* lenspyx v2.0.5
* cobaya v3.5
* GetDist v1.5.3
* Others: scipy v1.13.1, numpy v1.23.5, pickle v4.0, astropy v5.3, tqdm v4.64.1, matplotlib v3.7.1.
	
### Modification in lensQUEST

We have included the implementation of the Realization-dependent N0 computation using a semi-analytical approach. The function to call is lensquest.quest_norm_RD_iterSims(...), which uses OpenMP parallelization for significantly faster execution.

### Modification in lensingbiases

The N1 computation was modified to allow computing the N1 bias without needing to use an ini file. Additionally, in this new version, the power spectra noise is not calculated internally, and now the observed angular power spectra is passed as a parameter. 

All the information is passed via the arguments of the python function LensingBiases.compute_n1_py(...), which computes the unnormalized N1 bias using OPENMP parallelization. We did not integrate the N0 computation accordingly in this modified version, because we are computing it using lensquest or the MCN0 implementation in PlanckBIRD-lens.

### About cmblensplus

Installing cmblensplus can be complicated, and in reality, it is barely used:

* In [``filtering.py``](filtering.py), ``cmblensplus`` is imported only for the pixel-based filtering, which is not used in our code and left there for comparison purposes. By commenting the line [``import curvedsky as cs``](filtering.py#L27), you solve the import error. 

* In [``reconstruction.py``](reconstruction.py) only the python scripts [``analysis.py``](https://github.com/toshiyan/cmblensplus/blob/master/utils/analysis.py) and [``binning.py``](https://github.com/toshiyan/cmblensplus/blob/master/utils/binning.py) from [``cmblensplus/utils``](https://github.com/toshiyan/cmblensplus/tree/master/utils) are needed. There are two possible solutions:
	*  Installing ``cmblensplus/utils`` by creating a ``setup.py`` inside the cmblensplus folder and running in the terminal ``pip3 install .``:
 	```bash
  	#!/usr/bin/env python

	# Made by Miguel Ruiz Granda
	from distutils.core import setup
	
	setup(name='utils',
	      version='1.0',
	      packages=['utils'],
	     )
  	```
 	*    Copying [``analysis.py``](https://github.com/toshiyan/cmblensplus/blob/master/utils/analysis.py) and [``binning.py``](https://github.com/toshiyan/cmblensplus/blob/master/utils/binning.py) to ``PlanckBIRD-lens/notebooks`` directory and changing the following lines in [``reconstruction.py``](reconstruction.py#L37):
   	```python3
	from utils import analysis as ana --> import analysis as ana
	from utils import binning --> import binning
### Downloading Planck Galactic masks:

For running the filtering with the *Planck* Galactic masks, the file ``HFI_Mask_GalPlane-apo0_2048_R2.00.fits`` needs to be downloaded into the input directory:

```bash
wget -O HFI_Mask_GalPlane-apo0_2048_R2.00.fits "http://pla.esac.esa.int/pla/aio/product-action?MAP.MAP_ID=HFI_Mask_GalPlane-apo0_2048_R2.00.fits"
```

## General description:
	
* [combination.py](combination.py) -- Simulates frequency maps using the experiments.yalm information for *Planck* and *LiteBIRD*. It performs component separation on those frequency maps for *Planck*, *LiteBIRD*, and *Planck* + *LiteBIRD*.

* [filtering.py](filtering.py) -- Filters the cleaned maps using an harmonic C-inverse filtering. This is a step needed for running the quadratic estimators (QE).

* [reconstruction.py](reconstruction.py) -- Reconstruct the lensing potential using the quadratic estimators formalism. It also includes tools for debiasing the lensing power spectrum, especially for the N0 and N1 bias estimation.

* [mcmc_cosmoparams_CMB_lensing.py](mcmc_cosmoparams_CMB_lensing.py) -- Runs an MCMC to estimate the constraints on $H_0$ -- $\sigma_8$ -- $\Omega_\mathrm{m}$ using a lensing-only likelihood. Previously [preprocessing_TF_cosmoparams.py](preprocessing_TF_cosmoparams.py) needs to be run to compute the covariances matrices and to apply the Transfer Function correction.

### Jupyter-notebooks

* [notebooks/HILC_figures_publish.ipynb](notebooks/HILC_figures_publish.ipynb) -- Generates plots of the HILC residuals for the different experiments and foreground complexities.

* [notebooks/harmonic_filtering_publish.ipynb](notebooks/harmonic_filtering_publish.ipynb) -- Generates plots of the harmonic filtering performance.
    
* [notebooks/bias_estimation_publish.ipynb](notebooks/bias_estimation_publish.ipynb) -- Generates plots of the N0 bias, N1 bias, Monte Carlo (MC) normalization correction, mean-field (MF), and the minimum variance (MV) weights.

* [notebooks/lensing_map_publish.ipynb](notebooks/lensing_map_publish.ipynb) -- Generates plots of the lensing maps for the different experiments and the simple foregrounds.

* [notebooks/qcl_lensing_power_spectrum_publish.ipynb](notebooks/qcl_lensing_power_spectrum_publish.ipynb) -- Generates plots of the lensing bandpowers for the different experiments and foreground complexities. It also computes the lensing signal-to-noise ratio (SNR) from the lensing bandpowers of the 400 simulations.

* [notebooks/cosmological_parameter_estimation_publish.ipynb](notebooks/cosmological_parameter_estimation_publish.ipynb) -- Plots the constrains on $H_0$, $\Omega_\mathrm{m}$ and $\sigma_8$ from the MCMC chains using the lensing-only likelihood.
  
* [notebooks/delensing_Planck_LiteBIRD_publish.ipynb](notebooks/delensing_Planck_LiteBIRD_publish.ipynb) -- Performs a semi-analytical delensing efficiency estimation and estimates the improvement on the constraints on the tensor-to-scalar ratio constraints.


