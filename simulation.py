"""
This class simulates CMB maps, foregrounds and noise for a certain experiment. 
It also performs the Harmonic Internal Linear Combination (HILC) method to clean the CMB maps.

Author: Miguel Ruiz-Granda
"""

import os
import numpy as np
import healpy as hp
from astropy.io import ascii
import lenspyx
from tqdm import tqdm
import pickle as pl
import experiment as exp
from harmonic_ILC import covariance_estimation, harmonic_ILC, apply_harmonic_weights
import time
import configparser
import matplotlib.pyplot as plt
import pysm3
import pysm3.units as u
import mpi4py.MPI as MPI
import sys
import pymaster as nmt
import copy


class Simulation:
    """
    A class that simulates CMB maps.

    """

    def __init__(self, experiment, fileUnlensedCls, fileLensedCls, alms_dir, fg_str, nside_fg,
                  Nsim, parallel, chance_corr, mode, param_mode, delta, mask_bol, 
                  fsky_mask, mask_fname, aposcale, apotype, nside_lensing=2048, lmax_lensing=2500, dlmax=1024):
        """
        Constructor of class SimulationCMBMaps.

        :param experiment: experiment to simulate.
        :type experiment: Experiment
        :param fileUnlensedCls: name of the file containing the CMB unlensed theoretical angular power spectra, from l=0
         to lmax+dlmax. It assumes CLASS type dimensionless total [l(l+1)/2pi] C_l's.
        :type fileUnlensedCls: str
        :param fileLensedCls: name of the file containing the CMB lensed theoretical angular power spectra, from l=0
         to lmax. It assumes CLASS type dimensionless total [l(l+1)/2pi] C_l's.
        :type fileLensedCls: str
        :param alms_dir: path to the directory where all the simulations are stored.
        :type alms_dir: str
        :param fg_str: foreground model to be used in the simulations.
        :type fg_str: str
        :param nside_fg: resolution of the foregrounds simulations.
        :type nside_fg: int
        :param Nsim: number of simulations.
        :type Nsim: int
        :param parallel: if True, it will run the simulations in parallel.
        :type parallel: bool
        :param chance_corr: if True, the covariance matrix will be estimated from the maps.
                            Otherwise, it will be estimated as the sum of the covariance matrices
                            of each of the components: cmb, foregrounds and noise.
        :type chance_corr: bool
        :param mode: binning mode of the HILC method. It can be 'uniform' or 'constant'.
        :type mode: str
        :param param_mode: parameter of the mode. It can be the bin size or the minimum number of alms per bin.
        :type param_mode: int
        :param delta: smoothing parameter of the HILC weights.
        :type delta: int
        :param lmax_lensing: maximum l value for the lensing potential harmonic coefficients.
        :type lmax_lensing: int
        :param dlmax: number of additional multipoles calculated for an accurate lensing.
        :type dlmax: int
        """

        # Mean temperature of the CMB
        self.T_CMB = 2.7255e6  # in muK

        # Store all the data regarding the experiment.
        self.exp = experiment
        self.Nsim = Nsim
        self.parallel = parallel

        # Parameters for lensing the maps
        self.lmax_lensing = lmax_lensing
        self.dlmax = dlmax
        self.nside_lensing = nside_lensing # To get a precise lensing of the maps (only needed for LiteBIRD).

        # Parameters for producing the foregrounds simulations using pysm3
        # Two cases are considered: 'no_fg' (no foregrounds case) and  'sx_dy'
        # foregrounds case in which x is the syncrotron model number and y is
        #  the dust model number.
        self.fg_str = fg_str
        self.nside_fg = nside_fg
        self.fg_maps = None
        self.fg_alms = None
        self.fg_cov_mat = None

        # Store the path to the input directory to store the input files
        # This directory should contain the CLASS files with the lensed 
        # and unlensed angular power spectra, the mask and the experimental
        # bands of the experiment.
        self.input_dir = os.path.join(os.getcwd(), f"input")

        # Parameters for the HILC method
        self.chance_corr = chance_corr
        self.mode = mode
        self.param_mode = param_mode
        self.delta = delta

        # Parameters for the mask used in the HILC
        mask_fsky = {20: 0, 40: 1, 60: 2, 70: 3, 80:4, 90:5, 97:6, 99:7}
        self.mask_bol = mask_bol
        if self.mask_bol:
            self.mask_fsky = fsky_mask/100
            self.mask = hp.ud_grade(hp.read_map(f'{self.input_dir}/{mask_fname}', field=mask_fsky[fsky_mask]),
                            nside_out=self.exp.nside)
            # Apodized mask with the desired scale and type
            self.apomask = nmt.mask_apodization(self.mask, aposcale, apotype=apotype)
            self.w2 = np.mean(self.apomask ** 2)
            print(f"Experiment: Mask {self.exp.name} with fsky={np.mean(self.apomask):.3f} used in the HILC", flush=True)
            self.filename = f"{self.exp.name}_{self.fg_str}_chance_{self.chance_corr}_mask_{self.mask_bol}_{self.mask_fsky}"
        else:
            self.filename = f"{self.exp.name}_{self.fg_str}_chance_{self.chance_corr}_mask_{self.mask_bol}"

        # Lensing potential harmonic coefficients
        self.plm = None

        # ########################################################################################
        # Create directory structure to store the simulations
        # ########################################################################################

        first_level = ['phi', 'signal', self.exp.name]
        second_level = ['noise', 'foregrounds', f"{self.fg_str}_chance_{self.chance_corr}"]
        third_level = ['weights_HILC', 'cleaned_maps_HILC', 'cleaned_spectra_HILC', 'noise_spectra_HILC', 'mean_spectra', 'plots',
                       'results', 'pixel_filter', 'harmonic_filter', 'MF']
        if self.fg_str != 'no_fg':
            third_level.append('resfg_spectra_HILC')

        # Create the first level directories
        for dir1 in first_level:
            os.makedirs(os.path.join(alms_dir, f"{dir1}"), exist_ok=True)
        # Store the first level directories names
        self.phi_dir = os.path.join(alms_dir, 'phi')
        self.signal_dir = os.path.join(alms_dir, 'signal')
        # Create the second level directories
        self.exp_dir = os.path.join(alms_dir, self.exp.name)
        for dir2 in second_level:
            os.makedirs(os.path.join(self.exp_dir, f"{dir2}"), exist_ok=True)
        # Store the second level directories names
        self.fg_dir = os.path.join(self.exp_dir, f"foregrounds")
        self.noise_dir = os.path.join(self.exp_dir, f"noise")
        self.run_dir = os.path.join(self.exp_dir, f"{self.fg_str}_chance_{self.chance_corr}")
        # Create the third level directories
        for dir3 in third_level:
            os.makedirs(os.path.join(self.run_dir, f"{dir3}"), exist_ok=True)
        # Store the third level directories names

        self.WHILC_dir = os.path.join(self.run_dir, f"weights_HILC")
        self.sims_HILC_dir = os.path.join(self.run_dir, f"cleaned_maps_HILC")
        self.sims_HILC_spectra_dir = os.path.join(self.run_dir, f"cleaned_spectra_HILC")
        self.noise_HILC_dir = os.path.join(self.run_dir, f"noise_spectra_HILC")
        self.mean_spectra_dir = os.path.join(self.run_dir, f"mean_spectra")
        self.plots_dir = os.path.join(self.run_dir, f"plots")
        self.pixel_filter_dir = os.path.join(self.run_dir, f"pixel_filter")
        self.harmonic_filter_dir = os.path.join(self.run_dir, f"harmonic_filter")
        self.MF_dir = os.path.join(self.run_dir, f"MF")
        if self.fg_str != 'no_fg':
            self.resfg_HILC_dir = os.path.join(self.run_dir, f"resfg_spectra_HILC")
        
        # ########################################################################################
        # Load the theoretical CMB angular power spectra from CLASS
        # ########################################################################################

        # The column names used in the angular power spectra files are:
        # colnames = ['1:l', '2:TT', '3:EE', '4:TE', '5:BB', '6:phiphi', '7:TPhi', '8:Ephi']
        self.unlenCls = ascii.read(f'{self.input_dir}/{fileUnlensedCls}', format="commented_header", header_start=10).as_array()
        self.lensCls = ascii.read(f'{self.input_dir}/{fileLensedCls}', format="commented_header", header_start=10).as_array()

        # We divide by the normalization factor given by CLASS code. We skip l=0 because it leads to an
        #  indeterminate form 0/0
        factor1 = 2 * np.pi / (self.unlenCls['1:l'][1:] * (self.unlenCls['1:l'][1:] + 1))
        factor2 = 2 * np.pi / (self.lensCls['1:l'][1:] * (self.lensCls['1:l'][1:] + 1))
        for name in self.unlenCls.dtype.names[1:]:
            self.unlenCls[name][1:] = factor1 * self.unlenCls[name][1:]
            self.lensCls[name][1:] = factor2 * self.lensCls[name][1:]

        # Theoretical lensed power spectra.
        self.lensedTheoryCls = np.array([self.lensCls['2:TT'][:self.exp.lmax + 1] * self.T_CMB ** 2,
                                         self.lensCls['3:EE'][:self.exp.lmax + 1] * self.T_CMB ** 2,
                                         self.lensCls['5:BB'][:self.exp.lmax + 1] * self.T_CMB ** 2,
                                         self.lensCls['4:TE'][:self.exp.lmax + 1] * self.T_CMB ** 2])
        

    @classmethod
    def from_ini(cls, ini_file):
        """
        Initialize the class from an ini file
        """
        config = configparser.ConfigParser()
        config.read(ini_file)
        exper = config['Experiment']
        exp_name = exper['exp_name']
        nside = int(exper['nside'])
        lmax = int(exper['lmax'])
        experiment = exp.Experiment(name=exp_name, nside=nside, lmax=lmax)

        sim = config['Simulation']
        fileUnlensedCls = sim['fileUnlensedCls']
        fileLensedCls = sim['fileLensedCls']
        alms_dir = sim['alms_dir']
        fg_str = sim['fg_str']
        nside_fg = int(sim['nside_fg'])
        Nsim = int(sim['Nsim'])
        parallel = config.getboolean('Simulation', 'parallel')
        nside_lensing = int(sim['nside_lensing'])
        lmax_lensing = int(sim['lmax_lensing'])
        dlmax = int(sim['dlmax'])

        hilc = config['HILC']
        chance_corr = config.getboolean('HILC', 'chance_correlation')
        mode = hilc['mode']
        param_mode = int(hilc['parameter_mode'])
        delta = int(hilc['delta'])
        mask_bol = config.getboolean('HILC', 'mask')
        fsky_mask = int(hilc['fsky_mask'])
        mask_fname = hilc['mask_fname']
        aposcale = float(hilc['aposcale'])
        apotype = hilc['apotype']
        return cls(experiment, fileUnlensedCls, fileLensedCls, alms_dir, fg_str, nside_fg,
                    Nsim, parallel, chance_corr, mode, param_mode, delta, mask_bol, 
                    fsky_mask, mask_fname, aposcale, apotype, nside_lensing=nside_lensing,
                      lmax_lensing=lmax_lensing, dlmax=dlmax)
    
        
    def choleskyCorrelation(self):
        """
        Generates unlensed CMB maps with the correlations given by the unlensed power spectra, self.unlenCls.

        First, we generate samples of three uncorrelated Gaussian variables of zero mean and unit variance, h, j and k.
        To do so, a constant power spectrum of value 1.0 is used to generate three sets of alm coefficients using the
        healpy function synalm.

        Then, we obtain correlated tlm_unl, elm_unl and plm_unl coefficients using the unlensed theoretical CMB power
        spectra generated using CLASS and the hlm, jlm and klm coefficients. To do so, a cholesky decomposition
        C(l) = L(l)*L^T(l) of the covariance matrix of the power spectra is performed and, following, a matrix
        multiplication M(l,m, correlated) = L(l) * M(l, m, uncorrelated) which leads to the correlated t_lm, e_lm
        and phi_lm coefficients.

        :return: the spherical harmonic coefficients of the correlated CMB maps, (t_lm, e_lm, b_lm), in muK.
        """

        # print('Running cholesky...', flush=True)
        flatCls = np.ones(self.lmax_lensing + self.dlmax + 1)
        hlm = hp.synalm(flatCls, lmax=self.lmax_lensing + self.dlmax, new=True)
        jlm = hp.synalm(flatCls, lmax=self.lmax_lensing + self.dlmax, new=True)
        klm = hp.synalm(flatCls, lmax=self.lmax_lensing + self.dlmax, new=True)

        # Initialize the arrays
        tlm_unl = np.zeros_like(hlm)
        elm_unl = np.zeros_like(hlm)
        blm_unl = np.zeros_like(hlm)  # zero (no lensing)
        self.plm = np.zeros_like(hlm)

        arr_lensed = self.unlenCls[:self.lmax_lensing + self.dlmax + 1]
        # Fill the arrays using cholesky decomposition technique
        L11 = np.sqrt(arr_lensed['6:phiphi'])
        L21 = np.divide(arr_lensed['7:TPhi'], L11, out=np.zeros_like(L11), where=L11!=0)
        L31 = np.divide(arr_lensed['8:Ephi'], L11, out=np.zeros_like(L11), where=L11!=0)
        L22 = np.sqrt(arr_lensed['2:TT']-L21**2)
        L32 = np.divide(arr_lensed['4:TE']-L21*L31, L22, out=np.zeros_like(L22), where=L22!=0)
        L33 = np.sqrt(arr_lensed['3:EE']-L31**2-L32**2)
       
        # Generate generate correlated Gaussian spherical harmonic coefficients
        philm = hp.almxfl(hlm, L11)
        tlm_unl = hp.almxfl(hlm, L21) + hp.almxfl(jlm, L22)
        elm_unl = hp.almxfl(hlm, L31) + hp.almxfl(jlm, L32) + hp.almxfl(klm, L33)
        return [tlm_unl * self.T_CMB, elm_unl * self.T_CMB, blm_unl * self.T_CMB, philm]


    def lensingMaps(self, tlm_unl, elm_unl, blm_unl, philm):
        """
        Generates (T, Q, U) lensed maps using the software lenspyx from the spherical harmonic coefficients of the
        unlensed CMB maps, (t_lm, e_lm, b_lm), and the lensing potential, phi_lm.

        :param tlm_unl: temperature spherical harmonic coefficients in muK.
        :type tlm_unl: numpy array
        :param elm_unl: E-mode polarization spherical harmonic coefficients in muK.
        :type elm_unl: numpy array
        :param blm_unl: B-mode polarization spherical harmonic coefficients in muK.
        :type blm_unl: numpy array
        :param philm: lensing potential spherical harmonic coefficients.
        :type philm: numpy array

        :return: lensed (T, Q, U) CMB maps in muK.
        """

        # print('Lensing the maps...', flush=True)
        # We transform the lensing potential into spin-1 deflection field.
        dlm = hp.almxfl(philm, np.sqrt(np.arange(self.lmax_lensing + self.dlmax + 1, dtype=float)
                                          * np.arange(1, self.lmax_lensing + self.dlmax + 2)))

        # We compute the position-space deflection.
        # Computes the temperature deflected spin-0 healpix map from tlm_unl and deflection field dlm.
        # Tlen = lenspyx.alm2lenmap(tlm_unl, [Red, Imd], nside, facres=self.facres, verbose=False)
        # Computes a deflected spin-weight healpix map from its gradient, elm_unl, and curl, blm_unl, modes
        # and deflection field dlm. Here we will use an Healpix grid
        geom_info = ('healpix', {'nside': self.nside_lensing})
        Tlen, Qlen, Ulen = lenspyx.alm2lenmap([tlm_unl, elm_unl, blm_unl], dlm, geometry=geom_info, verbose=0)

        # We compute the angular power spectra from the maps.
        self.lensedMapCls = hp.anafast([Tlen, Qlen, Ulen], lmax=self.lmax_lensing, pol=True, use_pixel_weights=True)

        return Tlen, Qlen, Ulen
    

    def signal(self, index):
        """
        Generate the CMB lensed maps for a given experiment and store them in a file. If the
        maps have already been generated, it will load them from the file.

        :param index: map identifier.
        :type index: int
        """

        # Check if the maps have already been generated.
        fname = f"{self.signal_dir}/signal_{index:04d}.pkl"
        fname_cls = f"{self.signal_dir}/cls_signal_{index:04d}.pkl"
        if os.path.isfile(fname):
            return pl.load(open(fname, 'rb'))
        else:
            # Running cholesky to generate the unlensed CMB maps.
            tlm_unl, elm_unl, blm_unl, philm = self.choleskyCorrelation()
            # Saving the phi_lm map used in the current simulation.
            pl.dump(philm, open(f"{self.phi_dir}/phi_{index:04d}.pkl", 'wb'))
            # Lensing the maps.
            Tlen, Qlen, Ulen = self.lensingMaps(tlm_unl, elm_unl, blm_unl, philm)
            cls_signal = hp.anafast([Tlen, Qlen, Ulen], lmax=self.lmax_lensing, pol=True, use_pixel_weights=True)
            del tlm_unl, elm_unl, blm_unl, philm
            # Save the lensed maps and power spectra in files.
            pl.dump([Tlen, Qlen, Ulen], open(fname, 'wb'))
            pl.dump(cls_signal, open(fname_cls, 'wb'))
            return Tlen, Qlen, Ulen


    def calculate_signal(self):
        """
        Calculate the CMB lensed maps for all the experiments. Only if 
        we want to compute all the signal maps at once.

        """

        # Calculate the CMB lensed maps for all the experiments.
        for i in tqdm(range(self.Nsim), desc='Creating signal maps', unit='map'):
            self.signal(i)

        
    def mean_signal(self):
        """
        This function calculates the mean signal of the signal simulations.

        """

        mean_signal = np.zeros((4, self.lmax_lensing + 1))
        fname_msignal = f"{self.signal_dir}/mean_signal_{self.fg_str}_{self.exp.name}_{self.Nsim}.pkl"

        for i in range(self.Nsim):
            fname = f"{self.signal_dir}/cls_signal_{i:04d}.pkl"
            if os.path.isfile(fname):
                mean_signal += pl.load(open(fname, 'rb'))[:4, :]
        mean_signal /= self.Nsim
        pl.dump(mean_signal, open(fname_msignal, 'wb'))
        return mean_signal


    def foregrounds(self):
        """
        Generates foregrounds simulations using pysm3 for the different foreground models.
        All the experiments' simulations use the same foreground simulations.

        """

        # We generate the foregrounds simulations using pysm3.
        sky = pysm3.Sky(nside=self.nside_fg, preset_strings=self.fg_str.split('_'))
        freq = self.exp.freq
        cls_fg = np.zeros((len(freq), 4, self.exp.lmax+1))
        for v in tqdm(freq,desc='Creating Foregrounds', unit='freq'):
            fname = os.path.join(self.fg_dir,f"{self.fg_str}_{self.exp.name}_{int(v)}.fits")
            if not os.path.isfile(fname):
                maps = sky.get_emission(v * u.GHz)
                maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(v*u.GHz))
                if np.isnan(maps.value[0]).any():
                    raise ValueError(f"The foregrounds {self.fg_str} for {self.exp.name} at {v} GHz have NaN values")
                if self.exp.nside != self.nside_fg:
                    upgraded = hp.alm2map(hp.map2alm(maps.value, pol=True, use_pixel_weights=True),
                                                  nside=self.exp.nside, pol=True)
                    cls_fg[np.where(freq == v)[0], :, :] = hp.anafast(upgraded, pol=True, use_pixel_weights=True,
                                                                   lmax=self.exp.lmax)[:4, :]
                    hp.write_map(fname, upgraded, dtype=np.float64)
                else:
                    cls_fg[np.where(freq == v)[0], :, :] = hp.anafast(maps.value, pol=True, use_pixel_weights=True,
                                                                   lmax=self.exp.lmax)[:4, :]
                    hp.write_map(fname, maps.value, dtype=np.float64)
            else:
                continue
        pl.dump(cls_fg, open(f"{self.fg_dir}/cls_{self.fg_str}_{self.exp.name}.pkl", 'wb'))
        print(f"The foregrounds {self.fg_str} for {self.exp.name} have been simulated", flush=True)


    def load_foregrounds(self):
        """
        Load the foregrounds simulations for a given frequency.

        :param freq: frequency of the foregrounds simulations.
        :type freq: float
        """

        fg_maps = np.zeros((len(self.exp.freq), 3, hp.nside2npix(self.exp.nside)))
        fg_alms = np.zeros((len(self.exp.freq), 3, hp.Alm.getsize(self.exp.lmax)), dtype='complex128')
        for index in range(len(self.exp.freq)):
            fname = os.path.join(self.fg_dir,f"{self.fg_str}_{self.exp.name}_{int(self.exp.freq[index])}.fits")
            if os.path.isfile(fname):
                fg_maps[index, :, :] = hp.read_map(fname, field=[0,1,2])
                fg_alms[index, :, :] = hp.map2alm(hp.read_map(fname, field=[0,1,2]), pol=True,
                                                   use_pixel_weights=True, lmax=self.exp.lmax)
                if np.isnan(fg_maps[index]).any():
                    raise ValueError(f"The foregrounds {self.fg_str} for {self.exp.name} at {self.exp.freq[index]} GHz have NaN values")
            else:
                print(f"{fname} does not exist", flush=True)
                return None
        self.fg_maps = fg_maps
        self.fg_alms = fg_alms
        self.fg_cov_mat_T = covariance_estimation(fg_alms[:, 0, :])
        self.fg_cov_mat_E = covariance_estimation(fg_alms[:self.exp.nfreq_p, 1, :])
        self.fg_cov_mat_B = covariance_estimation(fg_alms[:self.exp.nfreq_p, 2, :])
        # np.save(f"{self.fg_dir}/fg_cov_{self.fg_str}_{self.exp.name}.npy", self.fg_cov_mat_T)
    

    def noise(self, index):
        """
        Generate noise maps for all the channels of a given simulation and store them in a file.
        If the experiment has not temperature or polarization information for certain channels,
        it will fill the corresponding noise maps with Nan values.

        :param index: map identifier.
        :type index: int
        """

        # Check if the noise maps have already been generated.
        fname = f"{self.noise_dir}/noise_all_ch_{self.exp.name}_{index:04d}.pkl"
        if os.path.exists(fname):
            noise = pl.load(open(fname, 'rb'))
        else:
            resol = hp.nside2resol(self.exp.nside, arcmin=True)
            sigma_pix_I = self.exp.depth_i / resol
            sigma_pix_P = self.exp.depth_p / resol
            npix = hp.nside2npix(self.exp.nside)
            # Generate the random normal numbers
            noise_T = np.random.randn(self.exp.nfreq_i, npix)
            noise_E = np.random.randn(self.exp.nfreq_p, npix)
            noise_B = np.random.randn(self.exp.nfreq_p, npix)
            # Multiply the noise by the standard deviation
            noise_T *= sigma_pix_I[:self.exp.nfreq_i, None]
            noise_E *= sigma_pix_P[:self.exp.nfreq_p, None]
            noise_B *= sigma_pix_P[:self.exp.nfreq_p, None]
            noise = [noise_T, noise_E, noise_B]
            pl.dump(noise, open(fname, 'wb'))
        
        return noise
    

    def change_resolution(self, maps):
        """
        Change the resolution of the maps to the target resolution.
        It takes into account that Planck has less channels in
        polarization than in temperature.
        """

        nfreq_i = len(maps[0][0])
        nfreq_p = len(maps[1][0])
        new_maps_T = np.zeros((nfreq_i, hp.nside2npix(self.exp.nside)))
        new_maps_E = np.zeros((nfreq_p, hp.nside2npix(self.exp.nside)))
        new_maps_B = np.zeros((nfreq_p, hp.nside2npix(self.exp.nside)))
        for i in range(nfreq_i):
            if i < nfreq_p:
                new_maps_T[i, :], new_maps_E[i, :], new_maps_B[i, :] = hp.alm2map(hp.map2alm([maps[0][i], maps[1][i], maps[2][i]],
                                 pol=True, use_pixel_weights=True, lmax=self.exp.lmax),
                                 nside=self.exp.nside, pol=True)
            else:
                new_maps_T[i, :] = hp.alm2map(hp.map2alm(maps[0][i], pol=False, use_pixel_weights=True, lmax=self.exp.lmax),
                                 nside=self.exp.nside, pol=False)
        return [new_maps_T, new_maps_E, new_maps_B]


    def modelling_freq_bands(self, index):
        """
        Generates simulations of all frequency (T, Q, U) maps for a given CMB experiment. The simulations includes CMB 
        lensed signal, foregrounds and noise. It can be configured to run without foregrounds or with foregrounds (sxdy).

        This code is implemented for Planck and LiteBIRD combination, so the only missing channels that can handle
        is from polarization. Also, the missing polarization channels should be at the end of the frequency list.
         
        :param index: map identifier.
        :type index: int
        """

        # Load the files required to generate the simulations of the different bands.
        Ts, Qs, Us = self.signal(index)
   
        # Check if the map resolution is lower than the map combination. If so, lower the resolution of the map.
        if self.exp.nside != self.nside_lensing:
            Ts, Qs, Us = hp.alm2map(hp.map2alm([Ts, Qs, Us], pol=True, use_pixel_weights=True, lmax=self.exp.lmax),
                                 nside=self.exp.nside, pol=True)
        
        # Load the noise simulations
        Tn, Qn, Un = self.noise(index)

        # print('Simulating the maps and noise for the different frequency bands...', flush=True)
        maps_alm_T = np.zeros((self.exp.nfreq_i, hp.Alm.getsize(self.exp.lmax)), dtype='complex128')
        maps_alm_E = np.zeros((self.exp.nfreq_p, hp.Alm.getsize(self.exp.lmax)), dtype='complex128')
        maps_alm_B = np.zeros((self.exp.nfreq_p, hp.Alm.getsize(self.exp.lmax)), dtype='complex128')
        noise_alm_T = np.zeros((self.exp.nfreq_i, hp.Alm.getsize(self.exp.lmax)), dtype='complex128')
        noise_alm_E = np.zeros((self.exp.nfreq_p, hp.Alm.getsize(self.exp.lmax)), dtype='complex128')
        noise_alm_B = np.zeros((self.exp.nfreq_p, hp.Alm.getsize(self.exp.lmax)), dtype='complex128')
        cls_maps = np.zeros((len(self.exp.freq), self.exp.lmax+1))
        for i in range(len(self.exp.freq)):
            pixwin = hp.pixwin(self.exp.nside_native[i], pol=True, lmax=self.exp.lmax)
            pixwin[1][0] = 1
            pixwin[1][1] = 1
            gauss_beam = hp.gauss_beam(fwhm=np.radians(self.exp.fwhm[i]/60), lmax=self.exp.lmax, pol=True)
            beam_T = gauss_beam[:, 0]*pixwin[0]
            beam_P = gauss_beam[:, 1]*pixwin[1]
            beam = [beam_T, beam_P, beam_P]
            if ~np.isnan(self.exp.depth_p[i]):
                # Smooth the maps with the corresponding beam.
                if self.fg_str == 'no_fg':
                    T = hp.smoothing(Ts, beam_window=beam_T, use_pixel_weights=True)
                    _, Q, U = hp.smoothing([Ts, Qs, Us], beam_window=beam_P, use_pixel_weights=True)
                else:
                    T = hp.smoothing(Ts+self.fg_maps[i, 0, :], beam_window=beam_T, use_pixel_weights=True)
                    _, Q, U = hp.smoothing([Ts+self.fg_maps[i, 0, :], Qs+self.fg_maps[i, 1, :], Us+self.fg_maps[i, 2, :]],
                                            beam_window=beam_P, use_pixel_weights=True)
                # Add gaussian random noise to each frequency band.
                T += Tn[i, :]
                Q += Qn[i, :]
                U += Un[i, :]
                # Beam deconvolution in harmonic space for the maps and the noise.
                maps_alm_T[i, :], maps_alm_E[i, :], maps_alm_B[i, :] = hp.map2alm([T, Q, U], pol=True, use_pixel_weights=True, lmax=self.exp.lmax)
                noise_alm_T[i, :], noise_alm_E[i, :], noise_alm_B[i, :] = hp.map2alm([Tn[i, :], Qn[i, :], Un[i, :]], pol=True, use_pixel_weights=True, lmax=self.exp.lmax)
                maps_alm_T[i, :], maps_alm_E[i, :], maps_alm_B[i, :] = [hp.almxfl(maps_alm_T[i, :], 1 / beam[0]),
                                                                        hp.almxfl(maps_alm_E[i, :], 1 / beam[1]),
                                                                        hp.almxfl(maps_alm_B[i, :], 1 / beam[2])]
                noise_alm_T[i, :], noise_alm_E[i, :], noise_alm_B[i, :] = [hp.almxfl(noise_alm_T[i, :], 1 / beam[0]),
                                                                           hp.almxfl(noise_alm_E[i, :], 1 / beam[1]),
                                                                           hp.almxfl(noise_alm_B[i, :], 1 / beam[2])]
            else:
                # Smooth the maps with the corresponding beam and add gaussian random noise to each frequency band.
                if self.fg_str == 'no_fg':
                    T = hp.smoothing(Ts, beam_window=beam_T, use_pixel_weights=True) + Tn[i, :]
                else:
                    T = hp.smoothing(Ts+self.fg_maps[i, 0, :], beam_window=beam_T, use_pixel_weights=True) + Tn[i, :]
                # Beam deconvolution in harmonic space for the maps and the noise.
                maps_alm_T[i, :] = hp.map2alm(T, pol=True, use_pixel_weights=True, lmax=self.exp.lmax)
                noise_alm_T[i, :] = hp.map2alm(Tn[i, :], pol=True, use_pixel_weights=True, lmax=self.exp.lmax)
                maps_alm_T[i, :] = hp.almxfl(maps_alm_T[i, :], 1 / beam[0])
                noise_alm_T[i, :] = hp.almxfl(noise_alm_T[i, :], 1 / beam[0])
            
            cls_maps[i] = hp.alm2cl(maps_alm_T[i, :], lmax=self.exp.lmax)
        pl.dump(cls_maps, open(f"{self.sims_HILC_spectra_dir}/cls_maps_{self.exp.name}_{self.fg_str}_{index:04d}.pkl", 'wb'))
            
        maps_alms = [maps_alm_T, maps_alm_E, maps_alm_B]
        noise_alms = [noise_alm_T, noise_alm_E, noise_alm_B]    
        # Delete the inputs when are not needed anymore.
        del Ts, Qs, Us, Tn, Qn, Un
        return maps_alms, noise_alms


    def run_harmonic_ILC(self, index, maps_alms, noise_alms):
        """
        Run the Harmonic ILC method to clean the frequency maps.

        With the simulations of the different frequency bands, we perform 
        component separation method named Harmonic Internal Linear Combination (HILC).

        We stored different products after applying HILC: the cleaned CMB maps,
        the weights, the residual noise and foreground power spectra (the latter
        only when foregrounds are present).
        """
        
        fname_weights = f"{self.WHILC_dir}/WHILC_{self.filename}_{index:04d}.pkl"

        # Mask used only in temperature maps.
        if self.mask_bol:
            maps_alms = copy.deepcopy(maps_alms)
            for i in range(len(self.exp.freq)):
                T = hp.alm2map(maps_alms[0][i, :], nside=self.exp.nside)
                maps_alms[0][i, :] = hp.map2alm(self.apomask*T, use_pixel_weights=True, lmax=self.exp.lmax)
                    
        # Running Harmonic ILC
        # print('Running harmonic ILC...', flush=True)
        if self.chance_corr:
            cov_mat_T = covariance_estimation(maps_alms[0])
            cov_mat_E = covariance_estimation(maps_alms[1])
            cov_mat_B = covariance_estimation(maps_alms[2])
        else:
            noise_cov_mat_T = covariance_estimation(noise_alms[0])
            noise_cov_mat_E = covariance_estimation(noise_alms[1])
            noise_cov_mat_B = covariance_estimation(noise_alms[2])

            if self.fg_str == 'no_fg':
                cov_mat_T = noise_cov_mat_T
                cov_mat_E = noise_cov_mat_E
                cov_mat_B = noise_cov_mat_B
            else:
                cov_mat_T = noise_cov_mat_T + self.fg_cov_mat_T
                cov_mat_E = noise_cov_mat_E + self.fg_cov_mat_E
                cov_mat_B = noise_cov_mat_B + self.fg_cov_mat_B

        weights_T = harmonic_ILC(cov_mat_T, self.mode, self.param_mode, self.delta)[0]
        weights_E = harmonic_ILC(cov_mat_E, self.mode, self.param_mode, self.delta)[0]
        weights_B = harmonic_ILC(cov_mat_B, self.mode, self.param_mode, self.delta)[0]
        weights = [weights_T, weights_E, weights_B]
        pl.dump(weights, open(fname_weights, 'wb'))

        del maps_alms

        return weights

    
    def weighted_maps(self, index, weights, maps_alms, noise_alms):
        """
        Calculate the weighted maps and the power spectra of the weighted maps.
        """

        fname_cl_noise = f"{self.noise_HILC_dir}/cls_noise_HILC_{self.filename}_{index:04d}.pkl"
        fname_sims = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename}_{index:04d}.pkl"
        fname_cls_sims = f"{self.sims_HILC_spectra_dir}/cls_cleaned_{self.filename}_{index:04d}.pkl"
        weights_T = weights[0]
        weights_E = weights[1]
        weights_B = weights[2]
        hilc_maps = [apply_harmonic_weights(weights_T, maps_alms[0]),
                    apply_harmonic_weights(weights_E, maps_alms[1]),
                    apply_harmonic_weights(weights_B, maps_alms[2])]
        noise_maps = [apply_harmonic_weights(weights_T, noise_alms[0]),
                      apply_harmonic_weights(weights_E, noise_alms[1]),
                      apply_harmonic_weights(weights_B, noise_alms[2])]
        
        # The noise is considered isotropic, so we just calculate the power spectra
        # from the full sky maps.
        cl_noise = hp.alm2cl(noise_maps, lmax=self.exp.lmax)[:3, :]
        cl_hilc = hp.alm2cl(hilc_maps, lmax=self.exp.lmax)[:3, :]
        if self.mask_bol:
            T_cleaned = hp.alm2map(hilc_maps[0], nside=self.exp.nside)
            # Apply the mask only to the temperature maps and compute the power spectra.
            cl_hilc[0] = hp.anafast(self.apomask*T_cleaned, lmax=self.exp.lmax)/self.w2
        
        # Save the hilc cleaned map and PS and noise power spectra.
        pl.dump(hilc_maps, open(fname_sims, 'wb'))
        pl.dump(cl_hilc, open(fname_cls_sims, 'wb'))
        pl.dump(cl_noise, open(fname_cl_noise, 'wb'))

        # Save the power specforegrounds residuals.
        if self.fg_str != 'no_fg':
            resfg = [apply_harmonic_weights(weights_T, self.fg_alms[:, 0, :]),
                    apply_harmonic_weights(weights_E, self.fg_alms[:self.exp.nfreq_p, 1, :]),
                    apply_harmonic_weights(weights_B, self.fg_alms[:self.exp.nfreq_p, 2, :])]
            
            cl_resfg = hp.alm2cl(resfg, lmax=self.exp.lmax)[:3, :]
            if self.mask_bol:
                # Apply the mask only to the temperature maps.
                T_resfg = hp.alm2map(resfg[0], nside=self.exp.nside)
                cl_resfg[0] = hp.anafast(self.apomask*T_resfg, lmax=self.exp.lmax)/self.w2

            pl.dump(cl_resfg, open(f"{self.resfg_HILC_dir}/cls_resfg_HILC_{self.filename}_{index:04d}.pkl", 'wb'))
        
        # fname_cls_chance_s_n = f"{self.sims_HILC_spectra_dir}/cls_chance_s_n_{self.fg_str}_{self.exp.name}_{index:04d}.pkl"
        # fname_cls_chance_fg_n = f"{self.sims_HILC_spectra_dir}/cls_chance_n_fg_{self.fg_str}_{self.exp.name}_{index:04d}.pkl"
        # fname_cls_chance_s_fg = f"{self.sims_HILC_spectra_dir}/cls_chance_s_fg_{self.fg_str}_{self.exp.name}_{index:04d}.pkl"
        # Ts, Qs, Us = self.signal(index)
        # signal = hp.map2alm([Ts, Qs, Us], pol=True, use_pixel_weights=True, lmax=self.exp.lmax)
        # chance_s_n = [2*hp.alm2cl(signal[0], noise_maps[0], lmax=self.exp.lmax),
        #                 2*hp.alm2cl(signal[1], noise_maps[1], lmax=self.exp.lmax),
        #                  2*hp.alm2cl(signal[2], noise_maps[2], lmax=self.exp.lmax)] 
        # pl.dump(chance_s_n, open(fname_cls_chance_s_n, 'wb'))
        # if self.fg_str != 'no_fg':
        #     chance_s_fg = [2*hp.alm2cl(signal[0], resfg[0], lmax=self.exp.lmax),
        #                     2*hp.alm2cl(signal[1], resfg[1], lmax=self.exp.lmax),
        #                      2*hp.alm2cl(signal[2], resfg[2], lmax=self.exp.lmax)]
        #     chance_fg_n = [2*hp.alm2cl(resfg[0], noise_maps[0], lmax=self.exp.lmax),
        #                     2*hp.alm2cl(resfg[1], noise_maps[1], lmax=self.exp.lmax),
        #                      2*hp.alm2cl(resfg[2], noise_maps[2], lmax=self.exp.lmax)]
        #     pl.dump(chance_s_fg, open(fname_cls_chance_s_fg, 'wb'))
        #     pl.dump(chance_fg_n, open(fname_cls_chance_fg_n, 'wb'))
        del noise_maps
        if self.fg_str != 'no_fg':
            del resfg

    def simulate(self):
        """
        Simulate self.Nsim simulations of the HILC cleaned CMB maps for one of the
          individual experiments.

        """

        if self.parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            jobs = np.arange(rank, self.Nsim, size)
            # Simulate the foregrounds if necessary.
            #  It will be only simulated by the rank 0 process.
            if self.fg_str != 'no_fg' and rank == 0:
                self.foregrounds()
            # The rest of the processes will wait until the foregrounds are simulated.
            comm.Barrier()
            st = time.time()
            if self.fg_str != 'no_fg':
                self.load_foregrounds()
            if rank == 0:
                print(f"Running simulations for {self.exp.name} experiment", flush=True)
            # tqdm working only for rank 0.
            for index in tqdm(jobs) if rank == 0 else jobs:
                fname = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename}_{index:04d}.pkl"
                # Check if the simulations have already been generated.
                if os.path.isfile(fname):
                    continue
                # Run the simulations. 
                maps_alms, noise_alms = self.modelling_freq_bands(index)
                weights = self.run_harmonic_ILC(index, maps_alms, noise_alms)
                self.weighted_maps(index, weights, maps_alms, noise_alms)
            ft = time.gmtime(time.time() - st)
            comm.Barrier()
            print(f"Time rank {rank}:", time.strftime("%H:%M:%S", ft), flush=True)
            if rank == 0:
                self.plot_mean_spectra_map()
                self.plot_maps(0)

        else:
            # Simulate the foregrounds (if they are not already simulated and they are required).
            if self.fg_str != 'no_fg':
                self.foregrounds()
                self.load_foregrounds()
            print(f"Running simulations for {self.exp.name} experiment", flush=True)
            for index in tqdm(range(self.Nsim)):
                fname = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename}_{index:04d}.pkl"
                # Check if the simulations have already been generated.
                if os.path.isfile(fname):
                    continue
                # Run the simulations.
                maps_alms, noise_alms = self.modelling_freq_bands(index)
                weights = self.run_harmonic_ILC(index, maps_alms, noise_alms)
                self.weighted_maps(index, weights, maps_alms, noise_alms)
            self.plot_mean_spectra_map()
            self.plot_maps(0)

        
    def mean_map(self):
        """
        This function calculates the mean map of the simulations.

        :param N: number of simulations
        :type N: int
        """

        mean_map = np.zeros((3, self.exp.lmax + 1))
        fname_mmap = f"{self.mean_spectra_dir}/mean_map_{self.filename}_{self.Nsim}.pkl"

        for i in range(self.Nsim):
            fname = f"{self.sims_HILC_spectra_dir}/cls_cleaned_{self.filename}_{i:04d}.pkl"
            if os.path.isfile(fname):
                mean_map += pl.load(open(fname, 'rb'))
        mean_map /= self.Nsim
        pl.dump(mean_map, open(fname_mmap, 'wb'))
        return mean_map
            

    def mean_noise(self):
        """
        This function calculates the mean noise of the noise simulations.

        """

        mean_noise = np.zeros((3, self.exp.lmax + 1))
        fname_mnoise = f"{self.mean_spectra_dir}/mean_noise_{self.filename}_{self.Nsim}.pkl"

        for i in range(self.Nsim):
            fname = f"{self.noise_HILC_dir}/cls_noise_HILC_{self.filename}_{i:04d}.pkl"
            if os.path.isfile(fname):
                mean_noise += pl.load(open(fname, 'rb'))
            else:
                print('Noise file not found', flush=True)
                raise FileNotFoundError
        mean_noise /= self.Nsim
        pl.dump(mean_noise, open(fname_mnoise, 'wb'))
        return mean_noise
    

    def mean_chance(self):
        """
        This function calculates the mean chance correlation
        between the different components of the simulations.

        """

        mean_chance_s_n = np.zeros((3, self.exp.lmax + 1))
        fname_mchance_s_n = f"{self.mean_spectra_dir}/mean_chance_s_n_{self.fg_str}_{self.exp.name}_{self.Nsim}.pkl"
        fname_mchance_s_fg = f"{self.mean_spectra_dir}/mean_chance_s_fg_{self.fg_str}_{self.exp.name}_{self.Nsim}.pkl"
        fname_mchance_n_fg = f"{self.mean_spectra_dir}/mean_chance_n_fg_{self.fg_str}_{self.exp.name}_{self.Nsim}.pkl"

        for i in tqdm(range(self.Nsim)):
            fname_cls_chance_s_n = f"{self.sims_HILC_spectra_dir}/cls_chance_s_n_{self.fg_str}_{self.exp.name}_{i:04d}.pkl"
            if os.path.isfile(fname_cls_chance_s_n):
                mean_chance_s_n += pl.load(open(fname_cls_chance_s_n, 'rb'))
            else:
                print('Noise file not found', flush=True)
                raise FileNotFoundError
        mean_chance_s_n /= self.Nsim
        pl.dump(mean_chance_s_n, open(fname_mchance_s_n, 'wb'))

        if self.fg_str != 'no_fg':
            mean_chance_s_fg = np.zeros((3, self.exp.lmax + 1))
            mean_chance_n_fg = np.zeros((3, self.exp.lmax + 1))
            for i in range(self.Nsim):
                fname_cls_chance_s_fg = f"{self.sims_HILC_spectra_dir}/cls_chance_s_fg_{self.fg_str}_{self.exp.name}_{i:04d}.pkl"
                fname_cls_chance_n_fg = f"{self.sims_HILC_spectra_dir}/cls_chance_n_fg_{self.fg_str}_{self.exp.name}_{i:04d}.pkl"
                if os.path.isfile(fname_cls_chance_n_fg) and os.path.isfile(fname_cls_chance_s_fg):
                    mean_chance_n_fg += pl.load(open(fname_cls_chance_n_fg, 'rb'))
                    mean_chance_s_fg += pl.load(open(fname_cls_chance_s_fg, 'rb'))
                else:
                    print('Noise file not found', flush=True)
                    raise FileNotFoundError
            mean_chance_n_fg /= self.Nsim
            mean_chance_s_fg /= self.Nsim
            pl.dump(mean_chance_n_fg, open(fname_mchance_n_fg, 'wb'))
            pl.dump(mean_chance_s_fg, open(fname_mchance_s_fg, 'wb'))
        else:
            return mean_chance_s_n
        return mean_chance_s_n, mean_chance_s_fg, mean_chance_n_fg
              

    def mean_resfg(self):
        """
        This function calculates the mean foreground residuals of the simulations.

        """

        mean_resfg = np.zeros((3, self.exp.lmax + 1))
        fname_mresfg = f"{self.mean_spectra_dir}/mean_resfg_{self.filename}_{self.Nsim}.pkl"

        for i in range(self.Nsim):
            fname_resfg = f"{self.resfg_HILC_dir}/cls_resfg_HILC_{self.filename}_{i:04d}.pkl"
            if os.path.isfile(fname_resfg):
                mean_resfg += pl.load(open(fname_resfg, 'rb'))
            else:
                print('File not found.')
                raise FileNotFoundError
        mean_resfg /= self.Nsim
        pl.dump(mean_resfg, open(fname_mresfg, 'wb'))
        return mean_resfg
        

    def mean_spectra(self):
        """
        This function calculates the mean spectra of the signal, HILC noise, HILC map and
        foreground residuals.

        :param N: number of simulations
        :type N: int
        """
        
        print(f'Calculating the mean spectra of the {self.exp.name} simulations...', flush=True)
        msignal = self.mean_signal()
        mnoise = self.mean_noise()
        mmap = self.mean_map()
        if  self.fg_str != 'no_fg':
            mresfg = self.mean_resfg()
            return msignal, mnoise, mmap, mresfg
        return msignal, mnoise, mmap


    def plot_mean_spectra_map(self):
        """
        Plot the theoretical and simulation angular power spectra for comparison purposes for a single map.

        """

        L = np.arange(0, self.exp.lmax + 1)
        factor = (L * (L + 1)) / (2 * np.pi)
        factor2 = (L * (L + 1))**2 / (2 * np.pi)

        # I want to plot in the same figure the power spectra of the
        # signal, the noise, the foregrounds and the HILC map.
        if self.fg_str != 'no_fg':
            msignal, mnoise, mmap, mresfg = self.mean_spectra()
        else:
            msignal, mnoise, mmap = self.mean_spectra()
        
        mean_sn = np.zeros((4, self.exp.lmax + 1))
        mean_sn[:3] = self.lensedTheoryCls[:3] + mnoise
        mean_sn[3] = self.lensedTheoryCls[3]
        import lensquest
        plt.figure()
        N0_EB = lensquest.quest_norm(self.lensedTheoryCls, mean_sn, lmin=self.exp.lmin, lmax=self.exp.lmax,
                                                lminCMB=self.exp.lmin, lmaxCMB=self.exp.lmax)['EB']
        plt.plot(factor2*self.unlenCls[:self.exp.lmax + 1]['6:phiphi'], label='Lensing potential')
        plt.plot(factor2*N0_EB, label='N0 EB')
        plt.savefig(f"{self.plots_dir}/N0_EB_{self.filename}.pdf")

        for index in range(3):
            plt.figure()
            plt.plot(L, factor * msignal[index][:self.exp.lmax+1], label=self.exp.name + ' signal')
            plt.plot(L, factor * mmap[index], label=self.exp.name + ' map')
            plt.plot(L, factor * mnoise[index], label=self.exp.name + ' noise')
            # plt.plot(L, factor * np.abs(mchance_s_n[index]), label=self.exp.name + 'abs chance s_n')
            if self.fg_str != 'no_fg':
                plt.plot(L, factor * mresfg[index], label=self.exp.name + ' resfg')
                plt.plot(L, factor * (mnoise[index]+mresfg[index]), label=self.exp.name + ' total residuals')
                # plt.plot(L, factor * np.abs(mchance_n_fg[index]), label=self.exp.name + 'abs chance n_fg')
                # plt.plot(L, factor * np.abs(mchance_s_fg[index]), label=self.exp.name + 'abs chance s_fg')
            plt.xlim([2, self.exp.lmax])
            # plt.semilogx()
            plt.semilogy()
            plt.legend()
            plt.savefig(f"{self.plots_dir}/mean_spectra_HILC_map_{index}_{self.filename}.pdf")


    def plot_weights(self, index):
        """
        Plots the weights of the HILC method for a given simulation.
        """
        weights = pl.load(open(f"{self.WHILC_dir}/WHILC_{self.filename}_{index:04d}.pkl", 'rb'))

        plt.figure()
        for i in range(len(self.exp.nfreq_i)):
            plt.plot(weights[0][:, 0, i], label=self.exp.freq[i])
        plt.savefig(f"{self.plots_dir}/weights_map_0_{self.filename}.pdf")

        plt.figure()
        for i in range(len(self.exp.nfreq_q)):
            plt.plot(weights[1][:, 0, i], label=self.exp.freq[i])
        plt.savefig(f"{self.plots_dir}/weights_map_1_{self.filename}.pdf")

        plt.figure()
        for i in range(len(self.exp.nfreq_q)):
            plt.plot(weights[2][:, 0, i], label=self.exp.freq[i])
        plt.savefig(f"{self.plots_dir}/weights_map_2_{self.filename}.pdf")


    def plot_maps(self, index):
        """
        Plot the HILC cleaned CMB maps for a given simulation and the foreground residuals
        (only when foregrounds are present). This will be done for the three types of 
        foregrounds in consideration: 'no_fg' or 'sx_dy'.

        :param index: map identifier.
        :type index: int
        """

        # Path to the HILC cleaned CMB maps.
        fname_sim = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename}_{index:04d}.pkl"
        # Load the HILC cleaned CMB maps.
        beam = 80 # in arcmins
        sim = hp.alm2map(pl.load(open(fname_sim, 'rb')), nside=self.exp.nside, pol=True)
        T = hp.smoothing(sim[0], fwhm=np.radians(beam/60), pol=False)
        _, Q, U = hp.smoothing(sim, fwhm=np.radians(beam/60), pol=True)


        # Plot the maps
        plt.figure()
        hp.mollview(T, title=f"{self.exp.name} T HILC map {index}", min=-200, max=200, cmap='jet')
        plt.savefig(f"{self.plots_dir}/HILC_T_map_{self.filename}.pdf")
        plt.figure()
        hp.mollview(Q, title=f"{self.exp.name} Q HILC map {index}", min=-2.5, max=2.5, cmap='jet')
        plt.savefig(f"{self.plots_dir}/HILC_Q_map_{self.filename}.pdf")
        plt.figure()
        hp.mollview(U, title=f"{self.exp.name} U HILC map {index}", min=-2.5, max=2.5, cmap='jet')
        plt.savefig(f"{self.plots_dir}/HILC_U_map_{self.filename}.pdf")
        
        # Plot the foreground residual maps
        fname_weights = f"{self.WHILC_dir}/WHILC_{self.filename}_{index:04d}.pkl"
        weights = pl.load(open(fname_weights, 'rb'))
        if self.fg_str != 'no_fg':
            resfg = [apply_harmonic_weights(weights[0], self.fg_alms[:, 0, :]),
                    apply_harmonic_weights(weights[1], self.fg_alms[:self.exp.nfreq_p, 1, :]),
                    apply_harmonic_weights(weights[2], self.fg_alms[:self.exp.nfreq_p, 2, :])]
            resfg_map = hp.alm2map(resfg, nside=self.exp.nside, pol=True)
            resfg_smooth = hp.smoothing(resfg_map, fwhm=np.radians(beam/60), pol=True)
            plt.figure()
            hp.mollview(resfg_smooth[0], title=f"{self.exp.name} T resfg map {index}", min=-200, max=200, cmap='jet')
            plt.savefig(f"{self.plots_dir}/resfg_T_map_{self.filename}.pdf")
            plt.figure()
            hp.mollview(resfg_smooth[1], title=f"{self.exp.name} Q resfg map {index}", min=-2.5, max=2.5, cmap='jet')
            plt.savefig(f"{self.plots_dir}/resfg_Q_map_{self.filename}.pdf")
            plt.figure()
            hp.mollview(resfg_smooth[2], title=f"{self.exp.name} U resfg map {index}", min=-2.5, max=2.5, cmap='jet')
            plt.savefig(f"{self.plots_dir}/resfg_U_map_{self.filename}.pdf")
        plt.close('all')

if __name__ == "__main__":

    ini_file = sys.argv[1]
    simul = Simulation.from_ini(ini_file)
    # Run the simulations.
    simul.simulate()
    # simul.plot_maps(0)