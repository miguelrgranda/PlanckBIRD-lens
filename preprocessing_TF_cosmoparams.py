#-----------------------------------------------------
# Calculates the TF and applies the TF to correct
# the reconstructed lensing power spectrum.
# Forecast for Planck, LiteBIRD and Planck + LiteBIRD.
# Author: Christian Gimeno-Amo
#-----------------------------------------------------

import os
import sys
import numpy as np
import pickle as plk
from astropy.io import ascii

qcl_dir = sys.argv[1]
cov_dir = sys.argv[2]
fid_spectra_dir = sys.argv[3]
TF_dir = sys.argv[4]


def read_pkl(dir, name):
    """
    Reads a `.pkl` (pickle) file from a given directory.

    Args:
        dir (str): Path to the directory where the file is located. Should include trailing slash or be joined using `os.path.join`.
        name (str): Name of the pickle file (including the `.pkl` extension).

    Returns:
        Any: The deserialised object stored in the pickle file.
    """
    return plk.load(open(dir + name, "rb"))

# Binning functions

def compute_bp(lmin, lmax, n):
    """
    Computes bandpower bin edges and centres in multipole (`ℓ`) space using square-root binning.

    The bins are uniformly spaced in sqrt(ℓ), not in ℓ, which is often used in CMB analysis
    to ensure more uniform sampling at large scales.

    Args:
        lmin (float): Minimum multipole (ℓ) value.
        lmax (float): Maximum multipole (ℓ) value.
        n (int): Number of bins.

    Returns:
        tuple:
            - bp (np.ndarray): Array of bin edges of length `n + 1`.
            - bc (np.ndarray): Array of bin centres of length `n`.
    """
    dl = (np.sqrt(lmax)-np.sqrt(lmin))/n
    bp = np.array([(np.sqrt(lmin)+dl*i)**2 for i in range(n+1)])
    bc = np.array([(bp[i]+bp[i-1])/2 for i in range(1, n+1)])
    return bp, bc

def binning_opt_core(cl,vl,lmin,lmax,n,bp):
    """
    Computes weighted binned bandpowers from a given power spectrum.

    This function bins the input power spectrum `cl` into `n` bandpowers using weights `vl`.
    The bin edges are defined by the array `bp`.

    Args:
        cl (np.ndarray): Input power spectrum values, indexed from ℓ = lmin to ℓ = lmax.
        vl (np.ndarray): Weights associated with each multipole. Must be the same shape as `cl`.
        lmin (int): Minimum multipole ℓ in the `cl` and `vl` arrays.
        lmax (int): Maximum multipole ℓ (exclusive upper bound).
        n (int): Number of bins.
        bp (np.ndarray): Array of bin edges (length `n + 1`) in ℓ space. Typically from `compute_bp`.

    Returns:
        np.ndarray: Array of binned power spectrum values (length `n`), computed as the
        weighted average of `cl` within each bin.
    """

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

# Loading input files

data_Planck_no_fg = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_no_fg_Planck_fsky_0.7_MV.pkl")
data_LiteBIRD_no_fg = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_no_fg_LiteBIRD_fsky_0.8_MV.pkl")
data_Planck_LiteBIRD_no_fg = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_no_fg_LiteBIRD_Planck_fsky_0.8_MV.pkl")

data_Planck_s1d1 = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s1_d1_f1_a1_co1_Planck_fsky_0.7_MV.pkl")
data_LiteBIRD_s1d1 = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s1_d1_f1_a1_co1_LiteBIRD_fsky_0.8_MV.pkl")
data_Planck_LiteBIRD_s1d1 = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s1_d1_f1_a1_co1_LiteBIRD_Planck_fsky_0.8_MV.pkl")

data_Planck_s5d10 = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s5_d10_a1_f1_co3_Planck_fsky_0.7_MV.pkl")
data_LiteBIRD_s5d10 = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s5_d10_a1_f1_co3_LiteBIRD_fsky_0.8_MV.pkl")
data_Planck_LiteBIRD_s5d10 = read_pkl(qcl_dir, "/qcl_debiased_MC_norm_binned_TT_TE_EE_TB_EB_s5_d10_a1_f1_co3_LiteBIRD_Planck_fsky_0.8_MV.pkl")

# Splits for covariance matrix (MVMC means = minimum variance estimator)

N_cov = 300
N_avg = 100

Spectra_Planck_no_fg = data_Planck_no_fg["MVMV"][:100, :]
Spectra_LiteBIRD_no_fg = data_LiteBIRD_no_fg["MVMV"][:100, :]
Spectra_Planck_LiteBIRD_no_fg = data_Planck_LiteBIRD_no_fg["MVMV"][:100, :]

Spectra_Planck_s1d1 = data_Planck_s1d1["MVMV"][:100, :]
Cov_Planck_s1d1 = data_Planck_s1d1["MVMV"][100:, :]

Spectra_LiteBIRD_s1d1 = data_LiteBIRD_s1d1["MVMV"][:100, :]
Cov_LiteBIRD_s1d1 = data_LiteBIRD_s1d1["MVMV"][100:, :]

Spectra_Planck_LiteBIRD_s1d1 = data_Planck_LiteBIRD_s1d1["MVMV"][:100, :]
Cov_Planck_LiteBIRD_s1d1 = data_Planck_LiteBIRD_s1d1["MVMV"][100:, :]

Spectra_Planck_s5d10 = data_Planck_s5d10["MVMV"][:100, :]
Cov_Planck_s5d10 = data_Planck_s5d10["MVMV"][100:, :]

Spectra_LiteBIRD_s5d10 = data_LiteBIRD_s5d10["MVMV"][:100, :]
Cov_LiteBIRD_s5d10 = data_LiteBIRD_s5d10["MVMV"][100:, :]

Spectra_Planck_LiteBIRD_s5d10 = data_Planck_LiteBIRD_s5d10["MVMV"][:100, :]
Cov_Planck_LiteBIRD_s5d10 = data_Planck_LiteBIRD_s5d10["MVMV"][100:, :]

mean_MVMV_Planck_no_fg = np.mean(Spectra_Planck_no_fg, axis = 0)
mean_MVMV_LiteBIRD_no_fg = np.mean(Spectra_LiteBIRD_no_fg, axis = 0)
mean_MVMV_Planck_LiteBIRD_no_fg = np.mean(Spectra_Planck_LiteBIRD_no_fg, axis = 0)

mean_MVMV_Planck_s1d1 = np.mean(Spectra_Planck_s1d1, axis = 0)
mean_MVMV_LiteBIRD_s1d1 = np.mean(Spectra_LiteBIRD_s1d1, axis = 0)
mean_MVMV_Planck_LiteBIRD_s1d1 = np.mean(Spectra_Planck_LiteBIRD_s1d1, axis = 0)

mean_MVMV_Planck_s5d10 = np.mean(Spectra_Planck_s5d10, axis = 0)
mean_MVMV_LiteBIRD_s5d10 = np.mean(Spectra_LiteBIRD_s5d10, axis = 0)
mean_MVMV_Planck_LiteBIRD_s5d10 = np.mean(Spectra_Planck_LiteBIRD_s5d10, axis = 0)

# Covariance matrices (one of the inputs in the likelihood)

Cov_Planck_s1d1 = np.cov(Cov_Planck_s1d1.T)
Cov_LiteBIRD_s1d1 = np.cov(Cov_LiteBIRD_s1d1.T)
Cov_PlanckBIRD_s1d1 = np.cov(Cov_Planck_LiteBIRD_s1d1.T)
Cov_Planck_s5d10 = np.cov(Cov_Planck_s5d10.T)
Cov_LiteBIRD_s5d10 = np.cov(Cov_LiteBIRD_s5d10.T)
Cov_PlanckBIRD_s5d10 = np.cov(Cov_Planck_LiteBIRD_s5d10.T)

# Save covariance matrices

if os.path.exist(cov_dir + "/Cov_Mat") == False:
    os.makedirs(cov_dir + "/Cov_Mat/Planck/s1d1")
    os.makedirs(cov_dir + "/Cov_Mat/Planck/s5d10")
    os.makedirs(cov_dir + "/Cov_Mat/LiteBIRD/s1d1")
    os.makedirs(cov_dir + "/Cov_Mat/LiteBIRD/s5d10")
    os.makedirs(cov_dir + "/Cov_Mat/Planck+LiteBIRD/s1d1")
    os.makedirs(cov_dir + "/Cov_Mat/Planck+LiteBIRD/s5d10")

path = cov_dir + "/Cov_Mat/Planck/s1d1/Cov_PhiPhi.npy"
np.save(path, Cov_Planck_s1d1)

path = cov_dir + "/Cov_Mat/LiteBIRD/s1d1/Cov_PhiPhi.npy"
np.save(path, Cov_LiteBIRD_s1d1)

path = cov_dir + "/Cov_Mat/Planck+LiteBIRD/s1d1/Cov_PhiPhi.npy"
np.save(path, Cov_PlanckBIRD_s1d1)


path = cov_dir + "/Cov_Mat/Planck/s5d10/Cov_PhiPhi.npy"
np.save(path, Cov_Planck_s5d10)

path = cov_dir + "/Cov_Mat/LiteBIRD/s5d10/Cov_PhiPhi.npy"
np.save(path, Cov_LiteBIRD_s5d10)

path = cov_dir + "/Cov_Mat/Planck+LiteBIRD/s5d10/Cov_PhiPhi.npy"
np.save(path, Cov_PlanckBIRD_s5d10)

# Load the fiducial spectra

fiducial = fid_spectra_dir + "/base_2018_plikHM_TTTEEE_lowl_lowE_lensing_cl_lensed.dat"
cls = ascii.read(f'{fiducial}', format="commented_header", header_start=10).as_array()

# We divide by the normalization factor given by CLASS code. We skip l=0 because it leads to an
# indeterminate form 0/0

l = np.arange(2501)
cls_corrected = np.zeros(2501)
cls_corrected[1:] = (l[1:]*(l[1:]+1)) * cls['6:phiphi'][1:]

# Binning and transfer function

lmin_Planck = 2
lmax_Planck = 2048
n_Planck = 14

lmin_LiteBIRD = 2
lmax_LiteBIRD = 1000
n_LiteBIRD = 10

bp_Planck, bc_Planck = compute_bp(lmin_Planck, lmax_Planck, n_Planck)
bp_LiteBIRD, bc_LiteBIRD = compute_bp(lmin_LiteBIRD, lmax_LiteBIRD, n_LiteBIRD)

binning_0_Planck = binning_opt_core(cls_corrected[:2049], np.ones(2049), 0, 2048, 14, bp_Planck)
binning_0_LiteBIRD = binning_opt_core(cls_corrected[:1001], np.ones(1001), 0, 1000, 10, bp_LiteBIRD)

# Compute the transfer function with the No Foregrounds case

TF_Planck = mean_MVMV_Planck_no_fg - binning_0_Planck
TF_LiteBIRD = mean_MVMV_LiteBIRD_no_fg - binning_0_LiteBIRD
TF_PlanckBIRD = mean_MVMV_Planck_LiteBIRD_no_fg - binning_0_Planck

if os.path.exist(TF_dir + "/TF") == False:
    os.makedirs(TF_dir + "/TF")

np.save(TF_dir + "/TF/TF_Planck.npy", TF_Planck)
np.save(TF_dir + "/TF/TF_LiteBIRD.npy", TF_LiteBIRD)
np.save(TF_dir + "/TF/TF_Planck+LiteBIRD.npy", TF_PlanckBIRD)
