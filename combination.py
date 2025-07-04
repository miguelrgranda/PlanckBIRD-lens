"""
This class simulates and combines the frequency maps of two CMB experiments. 
It performs the Harmonic Internal Linear Combination (HILC) method to clean the CMB maps,
creating a cleaned maps for the individual experiments and a combined map of the two experiments.

Author: Miguel Ruiz-Granda
"""

import os
import numpy as np
import healpy as hp
from tqdm import tqdm
import pickle as pl
from simulation import Simulation as sim
from harmonic_ILC import covariance_estimation, harmonic_ILC, apply_harmonic_weights
import time
import configparser
import matplotlib.pyplot as plt
import mpi4py.MPI as MPI
import sys
import pymaster as nmt
import copy


class Combination:
    """
    Combination class.
    """

    def __init__(self, exp_name, ini_1, ini_2, nside, lmax, alms_dir, fg_str, nside_fg,
                    Nsim, parallel, chance_corr, mode, param_mode, delta, mask_bol, 
                    fsky_mask, mask_fname, aposcale, apotype):
        self.exp_name = exp_name
        # Experiment 1 (lower resolution)
        self.simul_1 = sim.from_ini(ini_1)
        # Experiment 2 (higher resolution)
        self.simul_2 = sim.from_ini(ini_2)
        
        self.nside = nside
        self.lmax = lmax

        # Run some tests to check that everything is coherent.
        if f"{self.simul_1.exp.name}_{self.simul_2.exp.name}" != exp_name:
            raise ValueError("The names or the experiment are not correct.")
        
        if self.simul_1.exp.lmax > self.simul_2.exp.lmax:
            raise ValueError("The experiments are not sorted correctly. "
                             "Lowest resolution experiment must be the first one.")
        
        if self.simul_2.exp.lmax != self.lmax:
            raise ValueError("The lmax of the combination does not match the lmax of the "
                             "highest resolution experiment.")
        
        if self.simul_1.fg_str != self.simul_2.fg_str:
            raise ValueError("Foregrounds must be the same for both experiments.")
        
        if self.simul_1.fg_str != fg_str:
            raise ValueError("Foreground model in the combination ini file is different from " 
                             "the foreground model in the individual experiment.")
        
        if self.simul_1.chance_corr != self.simul_2.chance_corr:
            raise ValueError("HILC chance correlation is different for two experiments.")
        
        # Mean temperature of the CMB
        self.T_CMB = 2.7255e6  # in muK

        # Store all the data regarding the experiment.
        self.Nsim = Nsim
        self.parallel = parallel

        # Parameters for producing the foregrounds simulations using pysm3
        # Two cases are considered: 'no_fg' (no foregrounds case) and  'sx_dy'
        # foregrounds case in which x is the syncrotron model number and y is
        #  the dust model number.
        self.fg_str = fg_str
        self.nside_fg = nside_fg

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

        # Parameters for the mask used in the HILC (coded for using Planck galactic masks)
        mask_fsky = {20: 0, 40: 1, 60: 2, 70: 3, 80:4, 90:5, 97:6, 99:7}
        self.mask_bol = mask_bol
        if self.mask_bol:
            self.mask_fsky = fsky_mask/100
            # Read the mask and downgrade/upgrade if necessary
            self.mask_1 = hp.ud_grade(hp.read_map(f'{self.input_dir}/{mask_fname}', field=mask_fsky[fsky_mask]),
                            nside_out=self.simul_1.exp.nside)
            self.mask_2 = hp.ud_grade(hp.read_map(f'{self.input_dir}/{mask_fname}', field=mask_fsky[fsky_mask]),
                            nside_out=self.simul_2.exp.nside)
            # Apodized mask with the desired scale and type
            self.apomask_1 = nmt.mask_apodization(self.mask_1, aposcale, apotype=apotype)
            self.apomask_2 = nmt.mask_apodization(self.mask_2, aposcale, apotype=apotype)
            self.w2_1 = np.mean(self.apomask_1 ** 2)
            self.w2_2 = np.mean(self.apomask_2 ** 2)
            print(f"Combination: Mask {self.simul_1.exp.name} with fsky={np.mean(self.apomask_1):.3f} used in the HILC", flush=True)
            print(f"Combination: Mask {self.simul_2.exp.name} with fsky={np.mean(self.apomask_2):.3f} used in the HILC", flush=True)
            self.filename = f"{self.exp_name}_{self.fg_str}_chance_{self.chance_corr}_mask_{self.mask_bol}_{self.mask_fsky}"
        else:
            self.filename = f"{self.exp_name}_{self.fg_str}_chance_{self.chance_corr}_mask_{self.mask_bol}"

        if self.chance_corr != self.simul_1.chance_corr:
            raise ValueError("HILC chance correlation of the combination is different from the individual the experiment.")


        # ########################################################################################
        # Create directory structure to store the simulations
        # ########################################################################################

        first_level = ['phi', 'signal', self.exp_name]
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
        self.exp_dir = os.path.join(alms_dir, self.exp_name)
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
    
    @classmethod
    def from_ini(cls, ini_file):
        """
        Initialize the class from an ini file
        """
        config = configparser.ConfigParser()
        config.read(ini_file)
        exper = config['Experiment']
        exp_name = exper['exp_name']
        ini_1 = exper['ini_1']
        ini_2 = exper['ini_2']
        nside = int(exper['nside'])
        lmax = int(exper['lmax'])

        sim = config['Simulation']
        alms_dir = sim['alms_dir']
        fg_str = sim['fg_str']
        nside_fg = int(sim['nside_fg'])
        Nsim = int(sim['Nsim'])
        parallel = config.getboolean('Simulation', 'parallel')

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
        return cls(exp_name, ini_1, ini_2, nside, lmax, alms_dir, fg_str, nside_fg,
                    Nsim, parallel, chance_corr, mode, param_mode, delta, mask_bol, 
                    fsky_mask, mask_fname, aposcale, apotype)
    

    def simulate(self):
        """
        Simulate Nsim simulations of the HILC cleaned CMB maps for the two 
        experiments.

        """

        if self.parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            jobs = np.arange(rank, self.Nsim, size)
            # Simulate the foregrounds if necessary.
            #  It will be only simulated by the rank 0 process.
            if rank == 0:
                if self.fg_str != 'no_fg':
                    self.simul_1.foregrounds()
                    self.simul_2.foregrounds()
            # The rest of the processes will wait until the foregrounds are simulated.
            comm.Barrier()
            st = time.time()
            if self.fg_str != 'no_fg':
                self.simul_1.load_foregrounds()
                self.simul_2.load_foregrounds()
            if rank == 0:
                print(f"Running simulations for {self.simul_1.exp.name}, {self.simul_2.exp.name} and {self.exp_name} experiment",
                       flush=True)
            # tqdm working only for rank 0.
            for index in tqdm(jobs) if rank == 0 else jobs:
                fname = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename}_{index:04d}.pkl"
                # Check if the simulations have already been generated.
                if os.path.isfile(fname):
                    continue
                # Run the simulations for the individual experiments. 
                maps_alms_1, noise_alms_1 = self.simul_1.modelling_freq_bands(index)
                maps_alms_2, noise_alms_2 = self.simul_2.modelling_freq_bands(index)
                # Run the HILC for the individual experiments.
                weights_1 = self.simul_1.run_harmonic_ILC(index, maps_alms_1, noise_alms_1)
                self.simul_1.weighted_maps(index, weights_1, maps_alms_1, noise_alms_1)
                weights_2 = self.simul_2.run_harmonic_ILC(index, maps_alms_2, noise_alms_2)
                self.simul_2.weighted_maps(index, weights_2, maps_alms_2, noise_alms_2)
                # Run HILC for the combination of the two experiments.
                self.run_harmonic_ILC_combination(index, maps_alms_1, noise_alms_1, maps_alms_2, noise_alms_2, weights_2)
            ft = time.gmtime(time.time() - st)
            comm.Barrier()
            print(f"Time rank {rank}:", time.strftime("%H:%M:%S", ft), flush=True)
            if rank == 0:
                # Plot the mean power spectra and a map
                self.simul_1.plot_mean_spectra_map()
                self.simul_2.plot_mean_spectra_map()
                self.plot_mean_spectra_comparison()
                self.plot_mean_spectra_map()
                # self.plot_maps(0)

        else:
            # Simulate the foregrounds (if they are not already simulated and they are required).
            if self.fg_str != 'no_fg':
                self.simul_1.foregrounds()
                self.simul_2.foregrounds()
                self.simul_1.load_foregrounds()
                self.simul_2.load_foregrounds()
            print(f"Running simulations for {self.exp_name} experiment", flush=True)
            for index in tqdm(range(self.Nsim)):
                fname = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename}_{index:04d}.pkl"
                # Check if the simulations have already been generated.
                if os.path.isfile(fname):
                    continue
                # Run the simulations.
                # Run the simulations for the individual experiments. 
                maps_alms_1, noise_alms_1 = self.simul_1.modelling_freq_bands(index)
                maps_alms_2, noise_alms_2 = self.simul_2.modelling_freq_bands(index)
                # Run the HILC for the individual experiments.
                weights_1 = self.simul_1.run_harmonic_ILC(index, maps_alms_1, noise_alms_1)
                self.simul_1.weighted_maps(index, weights_1, maps_alms_1, noise_alms_1)
                weights_2 = self.simul_2.run_harmonic_ILC(index, maps_alms_2, noise_alms_2)
                self.simul_2.weighted_maps(index, weights_2, maps_alms_2, noise_alms_2)
                # Run HILC for the combination of the two experiments.
                self.run_harmonic_ILC_combination(index, maps_alms_1, noise_alms_1, maps_alms_2, noise_alms_2, weights_2)
            # Plot the mean power spectra and a map
            self.simul_1.plot_mean_spectra_map()
            self.simul_2.plot_mean_spectra_map()
            self.plot_mean_spectra_map()
            self.plot_mean_spectra_comparison()
            # self.plot_maps(0)


    def run_harmonic_ILC_combination(self, index, maps_alms_1, noise_alms_1, maps_alms_2, noise_alms_2, weights_2):
        """
        Run the Harmonic ILC method to clean the frequency maps for the combination of two experiments.

        With the simulations of the different frequency bands, we perform 
        component separation method named Harmonic Internal Linear Combination (HILC).

        We stored different products after applying HILC: the cleaned CMB maps,
        the weights, the residual noise and foreground power spectra (the latter
        only when foregrounds are present).
        """
        
        fname_weights = f"{self.WHILC_dir}/WHILC_{self.filename}_{index:04d}.pkl"
        
        # Running Harmonic ILC

        # First part: HILC of the two experiments.
        wT_c, wE_c, wB_c = self.HILC_low_ell(maps_alms_1, noise_alms_1, maps_alms_2, noise_alms_2)
        # Second part: HILC of the highest resolution experiment (Planck for this project).
        wT_2, wE_2, wB_2 = weights_2
        # wT_2, wE_2, wB_2 = self.HILC_high_ell(maps_alms_2, noise_alms_2)
        # Combine the weights of the two experiments.
        weights_T = np.zeros((self.simul_2.exp.lmax+1, 1, self.simul_1.exp.nfreq_i + self.simul_2.exp.nfreq_i))
        weights_E = np.zeros((self.simul_2.exp.lmax+1, 1, self.simul_1.exp.nfreq_p + self.simul_2.exp.nfreq_p))
        weights_B = np.zeros((self.simul_2.exp.lmax+1, 1, self.simul_1.exp.nfreq_p + self.simul_2.exp.nfreq_p))
        weights_T[:self.simul_1.exp.lmax+1, :, :] = wT_c
        weights_T[self.simul_1.exp.lmax+1:, :, self.simul_1.exp.nfreq_p:] = wT_2[self.simul_1.exp.lmax+1:, :, :]
        weights_E[:self.simul_1.exp.lmax+1, :, :] = wE_c
        weights_E[self.simul_1.exp.lmax+1:, :, self.simul_1.exp.nfreq_p:] = wE_2[self.simul_1.exp.lmax+1:, :, :]
        weights_B[:self.simul_1.exp.lmax+1, :, :] = wB_c
        weights_B[self.simul_1.exp.lmax+1:, :, self.simul_1.exp.nfreq_p:] = wB_2[self.simul_1.exp.lmax+1:, :, :]
        # Store the weights in a file and return them.
        weights = [weights_T, weights_E, weights_B]
        pl.dump(weights, open(fname_weights, 'wb'))
        self.weighted_maps(index, maps_alms_1, noise_alms_1, maps_alms_2, noise_alms_2, weights)


    def HILC_low_ell(self, maps_alms_1, noise_alms_1, maps_alms_2, noise_alms_2):
        """
        Run HILC on the frequency maps of the two experiments for the low ell range.
        """

        # The mask is only applied to temperature maps.
        if self.mask_bol:
            # First, lets mask the frequency maps associated to the lowest resolution experiment.
            maps_alms_1 = copy.deepcopy(maps_alms_1)
            for i in range(len(self.simul_1.exp.freq)):
                T = hp.alm2map(maps_alms_1[0][i, :], nside=self.simul_1.exp.nside)
                maps_alms_1[0][i, :] = hp.map2alm(self.apomask_1*T, use_pixel_weights=True, lmax=self.simul_1.exp.lmax)
            # Second, lets mask the frequency maps associated to the highest resolution experiment.
            maps_alms_2 = copy.deepcopy(maps_alms_2)
            for i in range(len(self.simul_2.exp.freq)):
                T = hp.alm2map(maps_alms_2[0][i, :], nside=self.simul_2.exp.nside)
                maps_alms_2[0][i, :] = hp.map2alm(self.apomask_2*T, use_pixel_weights=True, lmax=self.simul_2.exp.lmax)

        # Change the lmax of the alms to the lowest resolution experiment.
        new_maps_alms_2 = [self.alm_change_lmax(maps_alms_2[0], self.simul_1.exp.lmax),
                            self.alm_change_lmax(maps_alms_2[1], self.simul_1.exp.lmax),
                            self.alm_change_lmax(maps_alms_2[2], self.simul_1.exp.lmax)]
        new_noise_alms_2 = [self.alm_change_lmax(noise_alms_2[0], self.simul_1.exp.lmax),
                            self.alm_change_lmax(noise_alms_2[1], self.simul_1.exp.lmax),
                            self.alm_change_lmax(noise_alms_2[2], self.simul_1.exp.lmax)]
        # Combine the alms of the two experiments.
        maps_alms = [np.concatenate((maps_alms_1[0], new_maps_alms_2[0]), axis=0),
                     np.concatenate((maps_alms_1[1], new_maps_alms_2[1]), axis=0),
                     np.concatenate((maps_alms_1[2], new_maps_alms_2[2]), axis=0)]
        noise_alms = [np.concatenate((noise_alms_1[0], new_noise_alms_2[0]), axis=0),
                        np.concatenate((noise_alms_1[1], new_noise_alms_2[1]), axis=0),
                        np.concatenate((noise_alms_1[2], new_noise_alms_2[2]), axis=0)]
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
                # Calculate the foreground covariance matrix of the combination
                fg_alms_T = np.concatenate((self.simul_1.fg_alms[:, 0, :], 
                            self.alm_change_lmax(self.simul_2.fg_alms[:, 0, :], self.simul_1.exp.lmax)))
                fg_alms_E = np.concatenate((self.simul_1.fg_alms[:self.simul_1.exp.nfreq_p, 1, :],
                            self.alm_change_lmax(self.simul_2.fg_alms[:self.simul_2.exp.nfreq_p, 1, :], self.simul_1.exp.lmax)))
                fg_alms_B = np.concatenate((self.simul_1.fg_alms[:self.simul_1.exp.nfreq_p, 2, :],
                            self.alm_change_lmax(self.simul_2.fg_alms[:self.simul_2.exp.nfreq_p, 2, :], self.simul_1.exp.lmax)))
                fg_cov_mat_T = covariance_estimation(fg_alms_T)
                fg_cov_mat_E = covariance_estimation(fg_alms_E)
                fg_cov_mat_B = covariance_estimation(fg_alms_B)
                # Add the foregrounds to the noise covariance matrix.
                cov_mat_T = noise_cov_mat_T + fg_cov_mat_T
                cov_mat_E = noise_cov_mat_E + fg_cov_mat_E
                cov_mat_B = noise_cov_mat_B + fg_cov_mat_B
        
        # Calculate the weights for the low ell range
        weights_T = harmonic_ILC(cov_mat_T, self.mode, self.param_mode, self.delta)[0]
        weights_E = harmonic_ILC(cov_mat_E, self.mode, self.param_mode, self.delta)[0]
        weights_B = harmonic_ILC(cov_mat_B, self.mode, self.param_mode, self.delta)[0]

        del maps_alms_1, maps_alms_2

        return weights_T, weights_E, weights_B
    

    def HILC_high_ell(self, maps_alms, noise_alms):
        """
        Run the Harmonic ILC method to clean the frequency maps.

        This function works only on the full sky (deprecated)
        """
        
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
                cov_mat_T = noise_cov_mat_T + self.simul_2.fg_cov_mat_T
                cov_mat_E = noise_cov_mat_E + self.simul_2.fg_cov_mat_E
                cov_mat_B = noise_cov_mat_B + self.simul_2.fg_cov_mat_B

        weights_T = harmonic_ILC(cov_mat_T, self.mode, self.param_mode, self.delta)[0]
        weights_E = harmonic_ILC(cov_mat_E, self.mode, self.param_mode, self.delta)[0]
        weights_B = harmonic_ILC(cov_mat_B, self.mode, self.param_mode, self.delta)[0]

        return weights_T, weights_E, weights_B


    def weighted_maps(self, index, maps_alms_1, noise_alms_1, maps_alms_2, noise_alms_2, weights):  
        """
        Multiply the weights with the alms to obtain the cleaned maps.
        """

        fname_cl_noise = f"{self.noise_HILC_dir}/cls_noise_HILC_{self.filename}_{index:04d}.pkl"
        fname_sims = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename}_{index:04d}.pkl"
        fname_cls_sims = f"{self.sims_HILC_spectra_dir}/cls_cleaned_{self.filename}_{index:04d}.pkl"

        # Store the weights in different variables.
        weights_T = weights[0]
        weights_E = weights[1]
        weights_B = weights[2]

        # Combine the alms of the two experiments for the weights multiplication.

        # Change the lmax of the alms to the lowest resolution experiment.
        new_maps_alms_1 = [self.alm_change_lmax(maps_alms_1[0], self.simul_2.exp.lmax),
                            self.alm_change_lmax(maps_alms_1[1], self.simul_2.exp.lmax),
                            self.alm_change_lmax(maps_alms_1[2], self.simul_2.exp.lmax)]
        new_noise_alms_1 = [self.alm_change_lmax(noise_alms_1[0], self.simul_2.exp.lmax),
                            self.alm_change_lmax(noise_alms_1[1], self.simul_2.exp.lmax),
                            self.alm_change_lmax(noise_alms_1[2], self.simul_2.exp.lmax)]
        if self.fg_str != 'no_fg':
            new_fg_alms_1 = [self.alm_change_lmax(self.simul_1.fg_alms[:, 0, :], self.simul_2.exp.lmax),
                            self.alm_change_lmax(self.simul_1.fg_alms[:, 1, :], self.simul_2.exp.lmax),
                            self.alm_change_lmax(self.simul_1.fg_alms[:, 2, :], self.simul_2.exp.lmax)]
        # Combine the alms of the two experiments.
        maps_alms = [np.concatenate((new_maps_alms_1[0], maps_alms_2[0]), axis=0),
                     np.concatenate((new_maps_alms_1[1], maps_alms_2[1]), axis=0),
                     np.concatenate((new_maps_alms_1[2], maps_alms_2[2]), axis=0)]
        noise_alms = [np.concatenate((new_noise_alms_1[0], noise_alms_2[0]), axis=0),
                        np.concatenate((new_noise_alms_1[1], noise_alms_2[1]), axis=0),
                        np.concatenate((new_noise_alms_1[2], noise_alms_2[2]), axis=0)]
        if self.fg_str != 'no_fg':
            fg_alms = [np.concatenate((new_fg_alms_1[0], self.simul_2.fg_alms[:, 0, :]), axis=0),
                        np.concatenate((new_fg_alms_1[1], self.simul_2.fg_alms[:self.simul_2.exp.nfreq_p, 1, :]), axis=0),
                        np.concatenate((new_fg_alms_1[2], self.simul_2.fg_alms[:self.simul_2.exp.nfreq_p, 2, :]), axis=0)]
        
        # Weighted maps
        hilc_maps = [apply_harmonic_weights(weights_T, maps_alms[0]),
                    apply_harmonic_weights(weights_E, maps_alms[1]),
                    apply_harmonic_weights(weights_B, maps_alms[2])]
        noise_maps = [apply_harmonic_weights(weights_T, noise_alms[0]),
                      apply_harmonic_weights(weights_E, noise_alms[1]),
                      apply_harmonic_weights(weights_B, noise_alms[2])]
        
        # The noise is considered isotropic, so we just calculate the power spectra
        # from the full sky maps.
        cl_noise = hp.alm2cl(noise_maps, lmax=self.lmax)[:3, :]
        cl_hilc = hp.alm2cl(hilc_maps, lmax=self.lmax)[:3, :]
        if self.mask_bol:
            T_cleaned = hp.alm2map(hilc_maps[0], nside=self.nside)
            # Apply the mask only to the temperature maps and compute the power spectra.
            cl_hilc[0] = hp.anafast(self.apomask_2*T_cleaned, lmax=self.lmax)/self.w2_2

        del noise_maps
        
        # Save the hilc cleaned map and PS and noise power spectra.
        pl.dump(hilc_maps, open(fname_sims, 'wb'))
        pl.dump(cl_hilc, open(fname_cls_sims, 'wb'))
        pl.dump(cl_noise, open(fname_cl_noise, 'wb'))

        # Save the power spectra of foregrounds residuals.
        if self.fg_str != 'no_fg':
            resfg = [apply_harmonic_weights(weights_T, fg_alms[0]),
                    apply_harmonic_weights(weights_E, fg_alms[1]),
                    apply_harmonic_weights(weights_B, fg_alms[2])]
            cl_resfg = hp.alm2cl(resfg, lmax=self.lmax)[:3, :]
            if self.mask_bol:
                # Apply the mask only to the temperature maps.
                T_resfg = hp.alm2map(resfg[0], nside=self.nside)
                cl_resfg[0] = hp.anafast(self.apomask_2*T_resfg, lmax=self.lmax)/self.w2_2

            pl.dump(cl_resfg, open(f"{self.resfg_HILC_dir}/cls_resfg_HILC_{self.filename}_{index:04d}.pkl", 'wb'))
            del resfg


    def mean_signal(self):
        """
        This function calculates the mean signal of the signal simulations.

        """

        mean_signal = np.zeros((4, self.lmax + 1))
        fname_msignal = f"{self.signal_dir}/mean_signal_{self.filename}_{self.Nsim}.pkl"

        for i in range(self.Nsim):
            fname = f"{self.signal_dir}/cls_signal_{i:04d}.pkl"
            if os.path.isfile(fname):
                mean_signal += pl.load(open(fname, 'rb'))[:4, :self.lmax+1]
        mean_signal /= self.Nsim
        pl.dump(mean_signal, open(fname_msignal, 'wb'))
        return mean_signal
    

    def mean_map(self):
        """
        This function calculates the mean map of the simulations.

        :param N: number of simulations
        :type N: int
        """

        mean_map = np.zeros((3, self.lmax + 1))
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

        mean_noise = np.zeros((3, self.lmax + 1))
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

        Deprecated
        """

        mean_chance_s_n = np.zeros((3, self.lmax + 1))
        fname_mchance_s_n = f"{self.mean_spectra_dir}/mean_chance_s_n_{self.filename}_{self.Nsim}.pkl"
        fname_mchance_s_fg = f"{self.mean_spectra_dir}/mean_chance_s_fg_{self.filename}_{self.Nsim}.pkl"
        fname_mchance_n_fg = f"{self.mean_spectra_dir}/mean_chance_n_fg_{self.filename}_{self.Nsim}.pkl"

        for i in tqdm(range(self.Nsim)):
            fname_cls_chance_s_n = f"{self.sims_HILC_spectra_dir}/cls_chance_s_n_{self.filename}_{i:04d}.pkl"
            if os.path.isfile(fname_cls_chance_s_n):
                mean_chance_s_n += pl.load(open(fname_cls_chance_s_n, 'rb'))
            else:
                print('Noise file not found', flush=True)
                raise FileNotFoundError
        mean_chance_s_n /= self.Nsim
        pl.dump(mean_chance_s_n, open(fname_mchance_s_n, 'wb'))

        if self.fg_str != 'no_fg':
            mean_chance_s_fg = np.zeros((3, self.lmax + 1))
            mean_chance_n_fg = np.zeros((3, self.lmax + 1))
            for i in range(self.Nsim):
                fname_cls_chance_s_fg = f"{self.sims_HILC_spectra_dir}/cls_chance_s_fg_{self.filename}_{i:04d}.pkl"
                fname_cls_chance_n_fg = f"{self.sims_HILC_spectra_dir}/cls_chance_n_fg_{self.filename}_{i:04d}.pkl"
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

        mean_resfg = np.zeros((3, self.lmax + 1))
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
        
        print(f'Calculating the mean spectra of the {self.exp_name} simulations...', flush=True)
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

        :param N: the number of maps used to compute the mean spectra.
        :type N: int
        """

        L = np.arange(0, self.lmax + 1)
        factor = (L * (L + 1)) / (2 * np.pi)

        # I want to plot in the same figure the power spectra of the
        # signal, the noise, the foregrounds and the HILC map.
        if self.fg_str != 'no_fg':
            msignal, mnoise, mmap, mresfg = self.mean_spectra()
        else:
            msignal, mnoise, mmap = self.mean_spectra()

        for index in range(3):
            plt.figure()
            plt.plot(L, factor * msignal[index][:self.lmax+1], label=self.exp_name + ' signal')
            plt.plot(L, factor * mmap[index], label=self.exp_name + ' map')
            plt.plot(L, factor * mnoise[index], label=self.exp_name + ' noise')
            if self.fg_str != 'no_fg':
                plt.plot(L, factor * mresfg[index], label=self.exp_name + ' resfg')
                plt.plot(L, factor * (mnoise[index]+mresfg[index]), label=self.exp_name + ' total residuals')
            plt.xlim([2, self.lmax])
            # plt.semilogx()
            plt.semilogy()
            plt.legend()
            plt.savefig(f"{self.plots_dir}/mean_spectra_HILC_map_{index}_{self.fg_str}_{self.exp_name}_{self.Nsim}_chance_{self.chance_corr}_{self.param_mode}.pdf")


    def plot_mean_spectra_comparison(self):
        """
        Plot the theoretical and simulation angular power spectra for comparison purposes for
        Planck, LiteBIRD and the combination of both experiments.

        """

        L = np.arange(0, self.lmax + 1)
        factor = (L * (L + 1)) / (2 * np.pi)

        # I want to plot in the same figure the power spectra of the
        # signal, the noise, the foregrounds and the HILC map.
        if self.fg_str != 'no_fg':
            msignal_1, mnoise_1, mmap_1, mresfg_1 = self.simul_1.mean_spectra()
            msignal_2, mnoise_2, mmap_2, mresfg_2 = self.simul_2.mean_spectra()
            msignal_c, mnoise_c, mmap_c, mresfg_c = self.mean_spectra()
        else:
            msignal_1, mnoise_1, mmap_1 = self.simul_1.mean_spectra()
            msignal_2, mnoise_2, mmap_2 = self.simul_2.mean_spectra()
            msignal_c, mnoise_c, mmap_c = self.mean_spectra()

        for index in range(3):
            plt.figure()
            plt.plot(L, factor * msignal_c[index][:self.lmax+1], label='Signal')
            lmax_1 = self.simul_1.exp.lmax
            if self.fg_str != 'no_fg':
                plt.plot(L[:lmax_1+1], factor[:lmax_1+1] * (mnoise_1[index]+mresfg_1[index]), label=self.simul_1.exp.name + ' residuals')
                plt.plot(L, factor * (mnoise_2[index]+mresfg_2[index]), label=self.simul_2.exp.name + ' residuals')
                plt.plot(L, factor * (mnoise_c[index]+mresfg_c[index]), label=self.exp_name + ' residuals')
            else:
                plt.plot(L[:lmax_1+1], factor[:lmax_1+1] * mnoise_1[index], label=self.simul_1.exp.name + ' residuals')
                plt.plot(L, factor * mnoise_2[index], label=self.simul_2.exp.name + ' residuals')
                plt.plot(L, factor * mnoise_c[index], label=self.exp_name + ' residuals')
            plt.xlim([2, self.lmax])
            # plt.semilogx()
            plt.semilogy()
            plt.legend()
            plt.savefig(f"{self.plots_dir}/mean_spectra_HILC_comparison_{index}_{self.fg_str}_{self.Nsim}_chance_{self.chance_corr}_{self.param_mode}.pdf")
    

    def plot_maps(self, index):
        """
        Plot the HILC cleaned CMB maps for a given simulation and the foreground residuals
        (only when foregrounds are present). This will be done for the three types of 
        foregrounds in consideration: 'no_fg' or 'sx_dy'.

        :param index: map identifier.
        :type index: int
        """

        # Path to the HILC cleaned CMB maps.
        fname_sim = f"{self.sims_HILC_dir}/cleaned_maps_{self.fg_str}_{self.exp_name}_{index:04d}.pkl"
        # Load the HILC cleaned CMB maps.
        beam = 80 # in arcmins
        sim = hp.alm2map(pl.load(open(fname_sim, 'rb')), nside=self.nside, pol=True)
        T = hp.smoothing(sim[0], fwhm=np.radians(beam/60), pol=False)
        _, Q, U = hp.smoothing(sim, fwhm=np.radians(beam/60), pol=True)


        # Plot the maps
        plt.figure()
        hp.mollview(T, title=f"{self.exp_name} T HILC map {index}", min=-200, max=200, cmap='jet')
        plt.savefig(f"{self.plots_dir}/HILC_T_map_{self.fg_str}_{self.exp_name}_{index:04d}_chance_{self.chance_corr}.pdf")
        plt.figure()
        hp.mollview(Q, title=f"{self.exp_name} Q HILC map {index}", min=-2.5, max=2.5, cmap='jet')
        plt.savefig(f"{self.plots_dir}/HILC_Q_map_{self.fg_str}_{self.exp_name}_{index:04d}_chance_{self.chance_corr}.pdf")
        plt.figure()
        hp.mollview(U, title=f"{self.exp_name} U HILC map {index}", min=-2.5, max=2.5, cmap='jet')
        plt.savefig(f"{self.plots_dir}/HILC_U_map_{self.fg_str}_{self.exp_name}_{index:04d}_chance_{self.chance_corr}.pdf")


    def alm_change_lmax(self, alm, new_lmax):
        """
        Changes the lmax of the alms according to Healpix ordering.

        :param alm: spherical harmonic coefficients to be changed.
        :param new_lmax: the new lmax to be changed.
        :return: new alms
        """
        alm_shape = alm.shape
        new_alm = np.zeros((alm_shape[0], hp.Alm.getsize(new_lmax)), dtype='complex_')
        lmax = hp.Alm.getlmax(alm_shape[1])
        min_lmax = np.min([lmax, new_lmax])
        for freq in range(alm_shape[0]):
            for m in range(min_lmax + 1):
                aux1 = hp.Alm.getidx(new_lmax, m, m)
                aux2 = hp.Alm.getidx(lmax, m, m)
                new_alm[freq, aux1:aux1 + min_lmax - m + 1] = alm[freq, aux2:aux2 + min_lmax - m + 1]
        return new_alm
    
if __name__ == "__main__":

    ini_file = sys.argv[1]
    comb = Combination.from_ini(ini_file)
    # Run the simulations.
    comb.simulate()