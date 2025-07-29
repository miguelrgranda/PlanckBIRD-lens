#-----------------------------------------------------
# Runs MCMC from Cobaya using the Phi-Phi spectra
# Forecast for Planck, LiteBIRD and Planck+LiteBIRD.
# Author: Christian Gimeno-Amo
#-----------------------------------------------------

import os
import sys
import numpy as np
import healpy as hp
import pickle as plk
from cobaya.run import run
import matplotlib.pyplot as plt
from cobaya.model import get_model

# Folder where the MCMC chains are stored
inp_dir = sys.argv[1]
output_folder = sys.argv[2]
covmat_path = sys.argv[3]
mission = int(sys.argv[4])
rminus_1 = float(sys.argv[5])
mode = int(sys.argv[6])
initial_covmat_path = sys.argv[7]
Hartlap = int(sys.argv[8])

'''
If mission == 0 : planck
If mission == 1 : litebir
If mission == 2 : planck + litebird

If mode == 0 : s1d1
If mode == 1 : s5d10

Hartlap == 0 : Hartlap factor is not applied on the inverse
Hartlap == 1 : Hartlap factor is applied on the inverse

No Foreground case is used to calibrate the TF, then this correction is applied on the s1d1 and s5d10 cases

'''

# Binning functions

def compute_bp(lmin, lmax, n):
    dl = (np.sqrt(lmax)-np.sqrt(lmin))/n
    bp = np.array([(np.sqrt(lmin)+dl*i)**2 for i in range(n+1)])
    return bp

def binning_opt_core(cl,vl,lmin,lmax,n,bp):
    cb = np.zeros(n)
    for i in range(n):
        b0, b1 = int(bp[i])-lmin , int(bp[i+1])-lmin
        cs = cl[b0:b1]
        wl = 1./vl[b0:b1]**2
        if np.count_nonzero(wl) > 0:
            cb[i] = np.sum(wl[wl!=0]*cs[wl!=0]) / np.sum(wl[wl!=0])
        else:
            cb[i] = 0
    return cb

# Correction factor

if mission == 0:
    lmax = 2048
    n = 14
    if mode == 0:
        data = plk.load(open(inp_dir + "/binned_lensing_power_spectrum/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s1_d1_f1_a1_co1_Planck_fsky_0.7_MV.pkl", "rb"))
    if mode == 1:
        data = plk.load(open(inp_dir + "/binned_lensing_power_spectrum/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s5_d10_a1_f1_co3_Planck_fsky_0.7_MV.pkl", "rb"))
    TF = np.load(inp_dir + "/TF/TF_Planck.npy")
elif mission == 1:
    lmax = 1000
    n = 10
    if mode == 0:
        data = plk.load(open(inp_dir + "/binned_lensing_power_spectrum/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s1_d1_f1_a1_co1_LiteBIRD_fsky_0.8_MV.pkl", "rb"))
    if mode == 1:
        data = plk.load(open(inp_dir + "/binned_lensing_power_spectrum/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s5_d10_a1_f1_co3_LiteBIRD_fsky_0.8_MV.pkl", "rb"))
    TF = np.load(inp_dir + "/TF/TF_LiteBIRD.npy")
elif mission == 2:
    lmax = 2048
    n = 14
    if mode == 0:
        data = plk.load(open(inp_dir + "/binned_lensing_power_spectrum/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s1_d1_f1_a1_co1_LiteBIRD_Planck_fsky_0.8_MV.pkl", "rb"))
    if mode == 1:
        data = plk.load(open(inp_dir + "/binned_lensing_power_spectrum/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s5_d10_a1_f1_co3_LiteBIRD_Planck_fsky_0.8_MV.pkl", "rb"))
    TF = np.load(inp_dir + "/TF/TF_PlanckBIRD.npy")

mean_MVMV = np.mean(data["MVMV"][:100], axis = 0)-TF
l = np.arange(lmax+1)
correction_factor = (l*(l+1))**2/2/np.pi
lmin = 2

# Covariance Matrix

Nsims = 300
Hartlap_fac = (Nsims - n - 2)/(Nsims - 1)
print("Hartlap factor: ", Hartlap_fac)

Covariance_Matrix = np.load(covmat_path + "/Cov_PhiPhi.npy")
Inv_Covariance_Matrix = np.linalg.inv(Covariance_Matrix)

if Hartlap == 1:
    Inv_Covariance_Matrix = Hartlap_fac * Inv_Covariance_Matrix

# Bins
bp = compute_bp(lmin, lmax, n)

# The likelihood
def my_like(_self=None):
    # Request the Cl from the provider
    Cl_theo_PP = _self.provider.get_Cl(ell_factor=False, units="muK2")['pp'][:lmax + 1]

    # Get derived parameter
    #sigma8 = _self.provider.get_param('sigma8')
    Corrected_Cl = correction_factor*Cl_theo_PP

    Binned_Cl = binning_opt_core(Corrected_Cl,np.ones(lmax+1),0,lmax,n,bp)
    # Compute the log-likelihood
    p = (mean_MVMV - Binned_Cl)
    logp = -0.5*(p@Inv_Covariance_Matrix@p.T)

    return logp

info = {
    'params': {
        # Fixed
        'tau_reio': 0.0544,
        # LambdaCDM parameters
        'H0': {'prior': {'max': 100, 'min': 40}, 'ref': {'dist': 'norm', 'loc': 67.36, 'scale': 2}, 'proposal': 2, 'latex': 'H_{0}'},
        'n_s': {'prior': {'dist': 'norm', 'loc': 0.965, 'scale': 0.02}, 'ref': {'dist': 'norm',
                            'loc': 0.965,'scale': 0.002}, 'proposal': 0.002, 'latex': 'n_\\mathrm{s}'},
        'A_s': {'latex': 'A_\\mathrm{s}', 'value': 'lambda logA: 1e-10*np.exp(logA)'},
        #'H0': {'latex': 'H_0', 'prior': {'max': 100, 'min': 40},'proposal': 2, 'ref': {'dist': 'norm', 'loc': 67.3, 'scale': 2}},
        'logA': {'drop': True, 'latex': '\\log(10^{10} A_\\mathrm{s})',
                 'prior': {'max': 4, 'min': 2},'proposal': 0.001, 'ref': {'dist': 'norm', 'loc': 3.044, 'scale': 0.001}},
        'omega_b': {'latex': '\\Omega_\\mathrm{b} h^2',
                    'prior': {'dist': 'norm','loc': 0.0224,'scale': 0.0005},
                    'proposal': 0.0001,
                    'ref': {'dist': 'norm','loc': 0.02237,'scale': 0.0005}},
        'omega_cdm': {'latex': '\\Omega_\\mathrm{c} h^2','prior': {'max': 0.99, 'min': 0.005}, 'ref': {'dist': 'norm', 'loc': 0.12, 'scale': 0.001},
                      'proposal': 0.0005},
        # Derived params
        'sigma8' : {'latex': '\\sigma_{8}'},
        'Omega_m' : {'latex': '\Omega_\mathrm{m}'}
        },
    'likelihood': {'my_cl_phi_phi_like': {"external": my_like, "requires": {'Cl': {'pp': lmax}}}},
    'theory': {'classy': {"extra_args": {"N_ncdm": 1, "N_ur": 2.0328, 'non_linear' : 'halofit', 'lensing' : True, 'accurate_lensing' : 1, 'm_ncdm' : 0.06, 'output': 'tCl, pCl, lCl, mPk', 'modes' : "s"}, 'stop_at_error': False}},
    'debug': False,
    'sampler': {'mcmc': {"burn_in": 0, 'Rminus1_stop': rminus_1, 'covmat': initial_covmat_path}},
    'output': output_folder,
    'resume': True,
    'progress': True}

model = get_model(info)
updated_info, sampler = run(info)
