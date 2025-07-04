"""
Class Experiment which contains the characteristics of an experiment.

Author: Miguel Ruiz-Granda
"""


import os.path as op
import yaml
import healpy as hp
import numpy as np


class Experiment:
    """
    A class that contains the characteristics of an experiment.

    :param name: name of the experiment.
    :type name: str
    :param nside: resolution of the map.
    :type nside: int
    :param lmax: maximum multipole of the map and the angular power spectra.
    :type lmax: int
    :param npix: number of pixels of the CMB maps.
    :type npix: int
    """

    def __init__(self, name, nside, lmax):
        """
        Constructor of the class Experiment.

        :param name: name of the map. It should map the name contained in input/experiments.yaml.
        :type name: str
        :param nside: resolution of the map.
        :type nside: int
        :param maxlmax: maximum multipole of the map according to its nside.
        :type maxlmax: int
        :param lmax: maximum multipole of the angular power spectra.
        :type lmax: int
        """

        self.name = name
        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.lmin = 2
        self.lmax = lmax
        # Load the experimental characteristics of the experiment
        with open(op.join('input/experiments.yaml'), 'r') as f:
            exps = yaml.safe_load(f)[name]

        self.freq =  np.array(exps['frequency']) # in GHz
        self.depth_i = np.array(exps['depth_i']) # in uK.arcmin
        self.depth_p = np.array(exps['depth_p']) # in uK.arcmin
        self.nfreq_i = np.sum(~np.isnan(self.depth_i)) # number of temperature channels different from NaN
        self.nfreq_p = np.sum(~np.isnan(self.depth_p)) # number of polarization channels different from NaN
        self.fwhm = np.array(exps['fwhm'])  # in arcmin
        self.nside_native = np.array(exps['nside']) # native nside of the maps of the experiment (needed only for the pixel function)