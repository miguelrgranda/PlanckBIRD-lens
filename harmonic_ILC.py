import numpy as np
import healpy as hp
from scipy import ndimage

"""
Module to perform the Harmonic Internal Linear Combination (HILC) method to clean 
the raw maps from foregrounds and reduce the noise.

Author: Miguel Ruiz-Granda
"""

def covariance_estimation(alms):
    """
    Estimates the C_\ell covariance matrix given the alms (T, E, B).
    It will treat them independently. The alms matrix have dimensions
    (n_freq, lm), being n_freq the number of frequencies and lm the 
    size of the alms.
    
    :param alms: matrix containing the alms.
    :type alms: complex array
    :return covariance matrix
    """
    shape_alms = alms.shape
    n_freq = shape_alms[0]
    lmax = hp.Alm.getlmax(shape_alms[1])
    cov = np.zeros((lmax+1, n_freq, n_freq))
    for i in range(n_freq):
        for j in range(i, n_freq):
            cov[:, i, j] = hp.alm2cl(alms[i, :], alms[j, :])
            cov[:, j, i] = cov[:, i, j]
    return cov


def proccesing_covariance(cov_mat, mode, param_mode):
    """
    Binning the covariance matrix to reduce the noise.
    Two approaches are implemented: uniform and constant.
    
    The uniform consists of a binning of the covariance matrix with a fixed step. This
    step is controled by the parameter param_mode.
     
    The constant approach performa a binning of the covariance matrix with a fixed minimum 
    number of alms per bin. This is controlled by the parameter param_mode.
    """
    
    cov = cov_mat.copy() 
    shape_cov = cov.shape
    lmax = shape_cov[0] - 1 

    if mode == 'uniform':
        # Average the covariances in the bin
        lbins = np.append(np.arange(2, lmax+1, param_mode), lmax+1)
        for lmin, lmax in zip(lbins[:-1], lbins[1:]):
            dof = 2 * np.arange(lmin, lmax) + 1
            w = dof/dof.sum()
            cov[lmin:lmax, ...] = ((w[:, np.newaxis, np.newaxis] * cov[lmin:lmax, ...]).sum(0))[np.newaxis, ...]
    if mode == 'constant':
        # First, calculate the bins of the covariance matrix
        L = np.arange(2, lmax+1)
        alms_per_ell = 2*L + 1
        lbins = [L[0]]
        sum_alms = 0
        for ell in L[:-1]:
            sum_alms += alms_per_ell[ell-2]
            if sum_alms >= param_mode:
                sum_alms = 0
                lbins.append(ell)
        lbins.append(L[-1])
        lbins = np.array(lbins)
        # Bin the covariance matrix
        for lmin, lmax in zip(lbins[:-1], lbins[1:]):
            dof = 2 * np.arange(lmin, lmax) + 1
            w = dof/dof.sum()
            cov[lmin:lmax, ...] = ((w[:, np.newaxis, np.newaxis] * cov[lmin:lmax, ...]).sum(0))[np.newaxis, ...]
    return cov, lbins


def stable_inverse(cov):
    """ 
    
    Calculate the inverse of the inverse covariance matrix. Inverting
    the correlation is numerically unstable, especially in situations
    with noise explotions due to the efect of the noise. It is better
    to calculate the inverse of the covariance matrix through the
    calculation of the inverse of the correlation matrix.

    We calculate the inverse of the covariance matrix in two steps:

        1. We calculate the inverse of the correlation matrix.
     
        2. Using the definition of the correlation matrix, we obtain
           the inverse of the covariance matrix.
    """
    
    # Get the std
    diag = np.sqrt(np.einsum('...ii->...i', cov))
    inv_diag = 1/diag
    inv_std = inv_diag[..., np.newaxis] * inv_diag[..., np.newaxis, :]
    # Calculate the correlation matrix
    corr = inv_std * cov
    # Calculate the pseudo-inverse of the correlation matrix
    inv_corr = np.linalg.pinv(corr)
    # Calculate the pseudo-inverse of the covariance matrix
    inv_cov = inv_std * inv_corr
    
    return inv_cov


def weight_calculation(cov):
    """
    Calculate the weights using the HILC equation.
    
    :param inv_cov: covariance matrix of dimensions (n_stokes, lmax, n_freq, n_freq)
    :type inv_cov: 
    :return HILC weights
    """

    shape_cov = cov.shape
    lmax = shape_cov[0] - 1 
    # The calculation of the inverse is done from ell=2
    # because there are no monopole nor dipole in the CMB data.
    inv_cov = stable_inverse(cov[2:, ...])
    n_freq = inv_cov.shape[-1]
    A = np.ones((n_freq, 1))
    norm = A.T @ inv_cov @ A
    weights = np.zeros((lmax+1, 1, n_freq))
    weights[2:, ...] = (A.T @ inv_cov) / norm
    return weights
    

def weights_smoothing(weights, Delta_ell):
    """
    Smooth the weights to reduce the fluctuations.
    
    :param weights: weights matrix of dimensions (n_stokes, lmax, 1, n_freq)
    :type weights: array
    :param Delta_ell: window size of the smoothing. Default is 15.
    :type Delta_ell: int
    """

    smoothed_weights = weights.copy()
    filter_weights = ndimage.uniform_filter(smoothed_weights[2:,...], size=Delta_ell, axes=(0,))
    smoothed_weights[2:, ...] = filter_weights
    return smoothed_weights


def harmonic_ILC(cov_mat, mode, param_mode, Delta_ell):
    """
    Run the Harmonic Internal Linear Combination (HILC) method.
    
    :param cov_mat: covariance matrix of dimensions (n_stokes, lmax, n_freq, n_freq).
    :type cov_mat: array
    :param mode: mode of the binning. It can be 'uniform' or 'constant'.
    :type mode: str
    :param param_mode: step of the binning(mode=='uniform') or number of alms per bin
                        (moded=='constant').
    :type param_mode: int
    :param Delta_ell: window size of the smoothing.
    :type Delta_ell: int
    """

    cov, lbins = proccesing_covariance(cov_mat, mode, param_mode)
    weights_smoothed = weights_smoothing(weight_calculation(cov), Delta_ell=Delta_ell)
    # Set to zero the weights that are too close to zero
    weights_smoothed[np.abs(weights_smoothed) < np.finfo(float).eps*100] = 0
    return weights_smoothed, lbins


def apply_harmonic_weights(weights, alms):
    """
    Apply harmonic weights to a given alms.
    """
    
    shape_alms = alms.shape
    n_freq = shape_alms[0]
    alms_w = np.zeros(shape_alms[1], dtype='complex128')
    for i in range(n_freq):
        alms_w += hp.almxfl(alms[i, :], weights[:, 0, i])
    return alms_w      