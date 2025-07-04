"""
The reconstruction class takes the C-inverse filtering (Wiener filter) 
of the simulated maps and uses them to reconstruct the graviitational
lensing potential. 

This class contains the necessary functions to:

    * Run the Quadratic Estimators (QE) on the maps.
    * Estimate the Mean Field (MF) correction caused by the mask and all of the 
      non-lensing statistical anisotropies.
    * Calculate the analytical normalisation of the QE, R_ana.
    * Calculate the analytical Monte Carlo N0 bias, MC-N0.
    * Calculate the semi-analytical realisation-dependent bias, RDN0.
    * Calculate the analytical Monte Carlo N1 bias, N1.
    * Calculate the Monte Carlo correction to the lensing power spectrum, C_L^MC.
    * Calculate an unbiased estimate of the reconstructed lensing power spectrum, C_L^{\phi\phi}.
    * Calculate the signal to noise ratio (with some plots).

Author: Miguel Ruiz-Granda
"""

import os
import sys
import numpy as np
import healpy as hp
from astropy.io import ascii
import pymaster as nmt
from tqdm import tqdm
import pickle as pl
import lensquest
import LensingBiases
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
import configparser
import experiment as exp
from utils import analysis as ana
from utils import binning
import scipy.interpolate


class Reconstruction:
    """
    A class that reconstructs the CMB lensing map and the CMB lensing power spectrum.

    """

    def __init__(self, experiment, fileUnlensedCls, fileLensedCls, alms_dir, fg_str, Nsim, parallel,
                 chance_corr, filename_HILC, qe_list, num_bins, bin_opt, mask_fname, fsky_mask = 80,
                   apotype="C1", aposcale = 2.0):
        """
        Constructor of class Reconstruction.

        :param experiment: experiment to simulate.
        :type experiment: Experiment
        :param fileUnlensedCls: name of the file containing the CMB unlensed theoretical angular power spectra, from l=0
         to lmax+dlmax. It assumes CLASS type dimensionless total [l(l+1)/2pi] C_l's.
        :type fileUnlensedCls: str
        :param fileLensedCls: name of the file containing the CMB lensed theoretical angular power spectra, from l=0
         to lmax. It assumes CLASS type dimensionless total [l(l+1)/2pi] C_l's.
        :type fileLensedCls: str
        :param alms_dir: name of the directory where all the simulations are stored.
        :type alms_dir: str
        :param fg_str: foreground model to be used in the simulations.
        :type fg_str: str
        :param Nsim: number of simulations.
        :type Nsim: int
        :param parallel: flag to parallelize the code.
        :type parallel: bool
        :param chance_corr: flag to include the chance correlation in the simulations.
        :type chance_corr: bool
        :param filename_HILC: name of the file containing the HILC sims.
        :type filename_HILC: str
        :param qe_list: list of Quadratic Estimators to be used. Example for the five estimators:
                        ['TT', 'EE', 'TE', 'TB', 'EB'].
        :type qe_list: list
        :param num_bins: number of bins to be used in the binning of the reconstructed lensing
                            power spectrum.
        :type num_bins: int
        :param bin_opt: binning option to be used. It can be '', 'log', 'log10', 'p2' or 'p3'.
        :type bin_opt: str
        :param mask_fname: name of the mask file, which is inside the input directory.
        :type mask_fname: str
        :param fsky_mask: fsky outside the mask (in %). Default is 80 % Planck mask.
        :type fsky_mask: int
        :param apotype: apodization type. Default is "C1".
        :type apotype: str
        :param aposcale: apodization scale. Default is 2.0.
        :type aposcale: float
        """

        # Mean temperature of the CMB
        self.T_CMB = 2.7255e6  # in muK

        # Store all the data regarding the experiment.
        self.exp = experiment
        self.Nsim = Nsim
        self.parallel = parallel

        # For the N1 calculation.
        self.alms_dir = alms_dir

        # Parameters for producing the foregrounds simulations using pysm3.
        self.fg_str = fg_str
        self.chance_corr = chance_corr
        self.filename_HILC = filename_HILC


        # Store the path to the input directory to store the input files
        # This directory should contain the CLASS files with the lensed 
        # and unlensed angular power spectra, the mask and the experimental
        # bands of the experiment.
        self.input_dir = os.path.join(os.getcwd(), f"input")

        # Load the Planck mask with the desired coveraged. If the mask has higher resolution
        # then required, ud_grade is used to downgrade the mask to the desired resolution.
        mask_fsky = {20: 0, 40: 1, 60: 2, 70: 3, 80:4, 90:5, 97:6, 99:7}
        self.mask_fsky = fsky_mask/100
        self.mask = hp.ud_grade(hp.read_map(f'{self.input_dir}/{mask_fname}', field=mask_fsky[fsky_mask]),
                        nside_out=self.exp.nside)
        
        # Apodized mask with the desired scale and type
        self.apomask = nmt.mask_apodization(self.mask, aposcale, apotype=apotype)
        self.w2 = np.mean(self.apomask ** 2)
        self.w4 = np.mean(self.apomask ** 4)
        print('w2', self.w2, 'w4', self.w4)

        # Define the custom sorting order
        custom_order = {'TT': 0, 'TE': 1, 'EE': 2, 'TB': 3, 'EB': 4}
        # Sort the list using the custom order
        self.qe_list = sorted(qe_list, key=lambda x: custom_order[x])
        self.qe_list_MV = self.qe_list + ['MV']
        # All the QE pairs with non-zero N0 noise bias
        self.noise_pairs = ['TTTT', 'TTTE', 'TTEE', 'TETE', 'TEEE', 'EEEE', 'TBTB', 'TBEB', 'EBEB']
        # All the QE pairs
        self.pairs = []
        for i in range(len(self.qe_list)):
            for j in range(i, len(self.qe_list)):
                self.pairs.append(self.qe_list[i] + self.qe_list[j])
        self.num_bins = num_bins
        self.bin_opt = bin_opt


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
        # self.MF_dir = os.path.join(self.run_dir, f"MF")
        if self.fg_str != 'no_fg':
            self.resfg_HILC_dir = os.path.join(self.run_dir, f"resfg_spectra_HILC")
        # Create the results directory and qe subdirectory
        self.results_dir = os.path.join(self.run_dir, f"results")
        self.results_qe_dir = os.path.join(self.results_dir, f"qe")
        os.makedirs(os.path.join(self.results_qe_dir), exist_ok=True)
        
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
        
        # Theoretical lensing power spectrum
        self.cl_phi = self.lensCls['6:phiphi'][:self.exp.lmax + 1]

        # Store the mean field maps
        self.phi_MF_1 = None
        self.phi_MF_2 = None


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
        Nsim = int(sim['Nsim'])

        hilc = config['HILC']
        chance_corr = config.getboolean('HILC', 'chance_correlation')
        mask_bol_hilc = config.getboolean('HILC', 'mask')
        fsky_mask_hilc = int(hilc['fsky_mask'])
        if mask_bol_hilc:
            filename_HILC = f"{exp_name}_{fg_str}_chance_{chance_corr}_mask_{mask_bol_hilc}_{fsky_mask_hilc/100}"
        else:
            filename_HILC = f"{exp_name}_{fg_str}_chance_{chance_corr}_mask_{mask_bol_hilc}"

        filt = config['Filtering']
        mask_fname = filt['mask_fname']
        fsky_mask = int(filt['fsky_mask'])
        apotype = filt['apotype']
        aposcale = float(filt['aposcale'])

        recon = config['Reconstruction']
        qe_list = config.get('Reconstruction', 'qe_list').split(',')
        parallel = config.getboolean('Reconstruction', 'parallel')
        num_bins = int(recon['nbins'])
        bin_opt = recon['bin_opt']

        return cls(experiment, fileUnlensedCls, fileLensedCls, alms_dir, fg_str, Nsim, parallel, 
                   chance_corr, filename_HILC,  qe_list, num_bins, bin_opt, mask_fname, fsky_mask=fsky_mask,
                     apotype=apotype, aposcale=aposcale)
        

    def mean_signal_noise(self):
        """
        This function calculates the sum of the signal plus the mean HILC noise.
        """
        
        mean_noise = np.zeros((4, self.exp.lmax + 1))
        fname_signal_noise = f"{self.mean_spectra_dir}/mean_signal_noise_{self.filename_HILC}_{self.Nsim}.pkl"
        self.mean_sn = np.zeros((4, self.exp.lmax + 1))
        if os.path.isfile(fname_signal_noise):
            self.mean_sn = pl.load(open(fname_signal_noise, 'rb'))[:, :self.exp.lmax+1]
            return self.mean_sn
        else:
            fname_mnoise = f"{self.mean_spectra_dir}/mean_noise_{self.filename_HILC}_{self.Nsim}.pkl"
            if os.path.isfile(fname_mnoise):
                mean_noise = pl.load(open(fname_mnoise, 'rb'))[:, :self.exp.lmax+1]
            else:
                print('Mean noise file not found', flush=True)
                raise FileNotFoundError
            self.mean_sn[:3] = self.lensedTheoryCls[:3] + mean_noise
            self.mean_sn[3] = self.lensedTheoryCls[3]
            pl.dump(self.mean_sn, open(fname_signal_noise, 'wb'))
            return self.mean_sn
    

    def get_cls_map(self, index):
        """
        Calculate the CMB angular power spectra of one of the masked simulations.

        :param index: index of the simulation.
        """

        fname_cls = f"{self.harmonic_filter_dir}/cls_wiener_filtered_TEB_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{index:04d}.pkl"
        cls_maps = pl.load(open(fname_cls, 'rb'))
        cls_squared = np.vstack((self.mean_sn[:3, :]**2, self.mean_sn[0, :]*self.mean_sn[1, :]))
        return cls_maps[:4, :] * cls_squared / self.w2


    def mean_spectra_map(self):
        """
        Calculate the mean angular power spectra of the masked simulations. We
        have corrected it by a constant factor w2 to account for the masking effect.

        """

        fname = os.path.join(self.mean_spectra_dir, f"cls_map_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{self.Nsim}.pkl")
        # To have a different name from the previous function.
        if os.path.isfile(fname):
            self.meanCls = np.mean(pl.load(open(fname,"rb")), axis=1)
        else:
            print('Calculating the mean angular power spectra of the masked maps...', flush=True)
            clsMap = np.zeros((4, self.Nsim, self.exp.lmax + 1))
            for index in tqdm(range(self.Nsim)):
                clsMap[:, index] = self.get_cls_map(index)
            self.meanCls = np.mean(clsMap, axis=1)
            pl.dump(clsMap, open(fname,"wb"))


    def calculate_R_ana(self):
        """
        Calculate the analytical normalisation of the quadratic estimators.

        """
        
        fname = os.path.join(self.results_dir, f"R_ana_{self.fg_str}_{self.exp.name}_Nsim_{self.Nsim}.pkl")
        if os.path.isfile(fname):
            return pl.load(open(fname,"rb"))
        else:
            print('Calculating the analytical normalisation...', flush=True) 
            st = time.time()
            R_ana = lensquest.quest_norm(self.lensedTheoryCls, self.mean_sn, lmin=self.exp.lmin, lmax=self.exp.lmax,
                                                lminCMB=self.exp.lmin, lmaxCMB=self.exp.lmax)
            pl.dump(R_ana, open(fname,"wb"))
            print('Time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - st)), flush=True)
            return R_ana
        

    def calculate_R_MC(self):
        """
        Calculate the MC normalisation correction of the Quadratic Estimators.

        """

        fname = os.path.join(self.results_dir,
                 f"R_MC_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_Nsim_{self.Nsim}_new.pkl")
        if os.path.isfile(fname) and False:
            return pl.load(open(fname,"rb"))
        else:
            print('Calculating the MC normalisation correction...', flush=True)
            R_MC = {qe: np.zeros((self.Nsim, self.exp.lmax + 1)) for qe in self.qe_list_MV}
            L = np.arange(self.exp.lmax + 1)
            fl = L * (L + 1) / 2
            for index in tqdm(range(self.Nsim)):
                # Load the input phi map
                fname_input = f"{self.phi_dir}/phi_{index:04d}.pkl"
                phi_input = alm_change_lmax(pl.load(open(fname_input, 'rb')), self.exp.lmax)
                # Multiply by fl the phi input and then we mask it.
                klm = hp.almxfl(phi_input,fl)
                kmap = hp.alm2map(klm, nside=self.exp.nside) * self.apomask
                klm_n = hp.map2alm(kmap, lmax=self.exp.lmax)
                # Correct by the factor fl
                phi_input = hp.almxfl(klm_n, 1/fl)
                # Get the QE result
                phi_qe = self.run_qe(index)
                phi_MF = {qe: (self.phi_MF_1[qe] + self.phi_MF_2[qe])/2 for qe in self.qe_list_MV}
                for qe in self.qe_list_MV:
                    phi_cross = hp.alm2cl(phi_qe[qe]-phi_MF[qe], phi_input, lmax=self.exp.lmax)
                    phi_auto = hp.alm2cl(phi_input, lmax=self.exp.lmax)
                    R_MC[qe][index] = phi_cross/phi_auto
            R_MC = {qe: np.mean(R_MC[qe], axis=0)/(self.w4/self.w2) for qe in self.qe_list_MV}
            pl.dump(R_MC, open(fname,"wb"))
            return R_MC
    

    def run_qe(self, ind, weights=None):
        """
        Run the Quadratic Estimators on C-inverse filtered maps.

        :param ind: index of the simulation.
        :type ind: int
        :param weights: weights to calculate the MV QE.
                         The QE must be calculated before.
        :type weights: numpy array. Default is None.
        """
        
        fname_qe = os.path.join(self.results_qe_dir,
                     f"qe_{ind:04d}_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        if os.path.isfile(fname_qe):
            # print(ind, flush=True)
            if weights is None:
                return pl.load(open(fname_qe,"rb"))
            else:
                # Calculate the MV QE and include it in the QE file.
                qe = pl.load(open(fname_qe,"rb"))
                qe['MV'] = np.zeros(hp.Alm.getsize(self.exp.lmax), dtype=np.complex128)
                for spec in self.qe_list:
                    qe['MV'] += hp.almxfl(qe[spec], weights[spec])
                pl.dump(qe, open(fname_qe,"wb"))
        else:
            # Load the C-inverse filtered harmonic alms
            fname_map = f"{self.harmonic_filter_dir}/Wiener_filtered_TEB_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{ind:04d}.pkl"
            sims = pl.load(open(fname_map, 'rb'))
            
            # Note: the dcl = self.meanCls included here is not used for the QE calculation. It is only required
            # for the analytical N0 bias calculation (normalization of the QE).
            questhelper = lensquest.quest(sims, self.lensedTheoryCls, self.mean_sn, lmin=2, lmax=self.exp.lmax,
                                           lminCMB=self.exp.lmin, lmaxCMB=self.exp.lmax, nside=self.exp.nside)

            # Runing the QE
            qe_dict = dict()
            norm = self.calculate_R_ana()
            for qe in self.qe_list:
                qe_dict[qe] = questhelper.grad(qe, norm=norm[qe], store='True')
            pl.dump(qe_dict, open(fname_qe,"wb"))
            return qe_dict
    

    def job_run_qe(self):
        """
        Run the Quadratic Estimators on all the C-inverse filtered maps.

        """

        if self.parallel:
            st = time.time()
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            if rank == 0:
                print('Calculating the quadratic estimators...', flush=True)
                # Create a progress bar for rank 0
                for ind in tqdm(range(rank, self.Nsim, size)):
                    self.run_qe(ind)
            else:
                for ind in range(rank, self.Nsim, size):
                    self.run_qe(ind)
            
            st_end = time.gmtime(time.time() - st)
            comm.Barrier()
            print(f"Rank {rank} time:", time.strftime("%H:%M:%S", st_end), flush=True)
            comm.Barrier()

        else:
            for ind in tqdm(range(self.Nsim)):
                self.run_qe(ind)
    
        
    def calculate_MC_N0(self, ind_sim, ind_dict, MCN0_cl):
        """
        Calculate the N0 bias from the Reconstructed potential using filtered Fields
        with different CMB fields.

        :param ind: index of the simulation.
        :type ind: int
        :param MCN0_cl: dictionary to store the MCN0 bias.
        :type MCN0_cl: dict
        :param parallel: flag to parallelize the code. Default is True.
        :type parallel: bool
        """

        fname1 = f"{self.harmonic_filter_dir}/Wiener_filtered_TEB_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{ind_sim:04d}.pkl"
        fname2 = f"{self.harmonic_filter_dir}/Wiener_filtered_TEB_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{ind_sim+1:04d}.pkl"
        map_1 = pl.load(open(fname1, 'rb'))
        map_2 = pl.load(open(fname2, 'rb'))
        
        questhelper1 = lensquest.quest(map_1, self.lensedTheoryCls, self.meanCls, lmin=2,
                                        lmax=self.exp.lmax, lminCMB=self.exp.lmin, lmaxCMB=self.exp.lmax, map2=map_2,
                                            nside=self.exp.nside)
        
        questhelper2 = lensquest.quest(map_2, self.lensedTheoryCls, self.meanCls, lmin=2,
                                        lmax=self.exp.lmax, lminCMB=self.exp.lmin, lmaxCMB=self.exp.lmax, map2=map_1,
                                            nside=self.exp.nside)

        glm1 = dict()
        glm2 = dict()
        for qe in self.qe_list:
            # Leg1: map1, Leg2: map2
            glm1[qe] = questhelper1.grad(qe, norm=self.calculate_R_ana()[qe], store='False')
            glm2[qe] = questhelper2.grad(qe, norm=self.calculate_R_ana()[qe], store='False')
        
        for pair in self.pairs:
            # Auto-spectra
            if pair[2:] == pair[:2]:
                qe = pair[2:]
                MCN0_cl[pair][ind_dict] = hp.alm2cl(glm1[qe]+glm2[qe])/(2*self.w4)

            # Cross-spectra
            else:
                qe1 = pair[:2]
                qe2 = pair[2:]
                MCN0_cl[pair][ind_dict] = hp.alm2cl(glm1[qe1]+glm2[qe1], glm1[qe2]+glm2[qe2])/(2*self.w4)

    
    def job_calculate_MC_N0(self, mean=True, MV=None): 
        """
        Calculate the Monte Carlo N0 bias of the Quadratic Estimators.

        :param mean: flag to calculate the mean of the MCN0 bias. Default is True.
        :type mean: bool
        :param MV: array corresponding to the MV estimator. Default is None
                    if no addition want to be done.
        :type MV: np array.
        """
        fname = os.path.join(self.results_dir,f"MCN0_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        if os.path.isfile(fname):
            MCN0_cl = pl.load(open(fname,'rb'))
            if MV is not None:
                MCN0_cl['MVMV'] = MV
                pl.dump(MCN0_cl, open(fname,'wb'))
            # Calculate and return the MCN0 bias
            if mean:
                return {pair: np.mean(MCN0_cl[pair], axis=0) for pair in MCN0_cl.keys()}
            return MCN0_cl
        else:
            if self.parallel:
                st = time.time()
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()

                if rank == 0:
                    print('Calculating the MCN0 normalisation correction...', flush=True)
                    sims = np.arange(0, self.Nsim, 2) # Step 2, can be changed to 1 if higher precision is needed.
                    quotient, remainder = divmod(len(sims), size)
                    counts = [quotient + 1 if rank < remainder else quotient for rank in range(size)]
                    starts = [sum(counts[:rank]) for rank in range(size)]
                    ends = [sum(counts[:rank+1]) for rank in range(size)]
                    sims_array = [sims[starts[rank]:ends[rank]] for rank in range(size)]
                else:
                    sims_array = None

                # Distribute the simulations among the ranks
                sims_array = comm.scatter(sims_array, root = 0)

                # Calculate the MCN0 bias for each rank
                MCN0_cl_rank = {pair: np.zeros((len(sims_array), self.exp.lmax + 1)) for pair in self.pairs}

                ind_dict = 0
                # Create a progress bar for rank 0
                for ind_sim in tqdm(sims_array) if rank == 0 else sims_array:
                    self.calculate_MC_N0(ind_sim, ind_dict, MCN0_cl_rank)
                    ind_dict += 1
                st_end = time.gmtime(time.time() - st)

                # Gather the results
                MCN0_cl_rank = comm.gather(MCN0_cl_rank, root=0)
                print(f"Rank {rank} time:", time.strftime("%H:%M:%S", st_end), flush=True)

                # Concatenate dictionaries into a single dictionary
                if rank == 0:
                    MCN0_cl = dict()
                    for pair in self.pairs:
                        MCN0_cl[pair] = np.concatenate([MCN0_cl_rank[i][pair] for i in range(len(MCN0_cl_rank))], axis=0)
                    pl.dump(MCN0_cl, open(fname,'wb'))
            else:
                sims = np.arange(0, self.Nsim, 2) # Step 2, can be changed to 1 if higher precision is needed.
                MCN0_cl = {pair: np.zeros((int(self.Nsim/2), self.exp.lmax + 1)) for pair in self.pairs}
                for ind_sim in tqdm(sims_array):
                    self.calculate_MC_N0(ind_sim, ind_sim//2, MCN0_cl)
                pl.dump(MCN0_cl, open(fname,'wb'))
                # Calculate and return the MCN0 bias
                if mean:
                    return {pair: np.mean(MCN0_cl[pair], axis=0) for pair in MCN0_cl.keys()}
                return MCN0_cl


    def calculate_MF(self, weights=None):
        """
        Calculate the mean field correction of the Quadratic Estimators.
        For now, we are going to use the cross-spectrum estimator. We divide 
        the simulations in two sets, estimate the mean-field for each one
        and calculate the cross-spectrum between them.

        """

        fname = os.path.join(self.results_dir, f"MF_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        fname_phi_MF_1 = os.path.join(self.results_dir, f"phi_MF_1_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        fname_phi_MF_2 = os.path.join(self.results_dir, f"phi_MF_2_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        if os.path.isfile(fname_phi_MF_1) and os.path.isfile(fname_phi_MF_2):
            self.phi_MF_1 = pl.load(open(fname_phi_MF_1,"rb"))
            self.phi_MF_2 = pl.load(open(fname_phi_MF_2,"rb"))
            if weights is not None:
                print('Calculating the MV MF map...', flush=True)
                self.phi_MF_1['MV'] = np.zeros(hp.Alm.getsize(self.exp.lmax), dtype=np.complex128)
                self.phi_MF_2['MV'] = np.zeros(hp.Alm.getsize(self.exp.lmax), dtype=np.complex128)
                for spec in self.qe_list:
                    self.phi_MF_1['MV'] += hp.almxfl(self.phi_MF_1[spec], weights[spec])
                    self.phi_MF_2['MV'] += hp.almxfl(self.phi_MF_2[spec], weights[spec])
                pl.dump(self.phi_MF_1, open(fname_phi_MF_1,"wb"))
                pl.dump(self.phi_MF_2, open(fname_phi_MF_2,"wb"))

        else: 
            print('Calculating the mean field correction...', flush=True)
            phi_MF_1 = {qe: np.zeros(hp.Alm.getsize(self.exp.lmax), dtype=np.complex128) for qe in self.qe_list}
            phi_MF_2 = {qe: np.zeros(hp.Alm.getsize(self.exp.lmax), dtype=np.complex128) for qe in self.qe_list}
            for ind in tqdm(range(int(self.Nsim/2))): 
                qe_1 = self.run_qe(ind)
                qe_2 = self.run_qe(self.Nsim-ind-1)
                for qe in self.qe_list:
                    phi_MF_1[qe] += qe_1[qe]
                    phi_MF_2[qe] += qe_2[qe]
            self.phi_MF_1 = {qe: phi_MF_1[qe]/(self.Nsim/2) for qe in self.qe_list}
            self.phi_MF_2 = {qe: phi_MF_2[qe]/(self.Nsim/2) for qe in self.qe_list}
            cl_MF = {pair: (hp.alm2cl(self.phi_MF_1[pair[:2]], self.phi_MF_2[pair[2:]], lmax=self.exp.lmax) +
                             hp.alm2cl(self.phi_MF_1[pair[2:]], self.phi_MF_2[pair[:2]], lmax=self.exp.lmax))/(2*self.w4) for pair in self.pairs}
            pl.dump(self.phi_MF_1, open(fname_phi_MF_1,"wb"))
            pl.dump(self.phi_MF_2, open(fname_phi_MF_2,"wb"))
            pl.dump(cl_MF, open(fname,"wb"))
            return cl_MF
                

    def analytical_N0(self, MV=None):
        """
        Calculate the analytical N0 bias. It is the same as the analytical normalisation
        of the QE.

        :param MV: array corresponding to the MV estimator. Default is None
                    if no addition want to be done.
        :type MV: np array.
        """

        fname = os.path.join(self.results_dir, f"N0_analytical_Nsim_{self.Nsim}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        if os.path.isfile(fname):
            N0 = pl.load(open(fname,"rb"))
            if MV is not None:
                N0['MVMV'] = MV
                pl.dump(N0, open(fname,"wb"))
            return N0
        else:
            print('Calculating the analytical N0 bias...', flush=True)
            st = time.time()
            N0 = lensquest.quest_norm(self.lensedTheoryCls, self.meanCls, self.mean_sn, lmin=self.exp.lmin, lmax=self.exp.lmax, 
                                        lminCMB=self.exp.lmin, lmaxCMB=self.exp.lmax, rdcl=self.meanCls, bias=True)[1]
            pl.dump(N0, open(fname,"wb"))
            print('Time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - st)), flush=True)
            return N0
        
        
    def semianalytical_RDN0_iter(self, MV=None):
        """
        Calculate the semi-analytical realisation-dependent bias RDN0. It is basically a Taylor
        expansion of the usual analytical N0 bias equation:

        $RDN0 = hatN0[<C_\ell>, \hat{C}_\ell] - 2*N0[<C_\ell>]$ 

        where hatN0 is the realization-dependent N0 bias correction and N0 is the usual analytical
        N0 bias calculated evaluated at the mean angular power spectra of the simulations.

        References:
            * J. Peloton, M. Schmittfull, A. Lewis, J. Carron, and O. Zahn. "Full covariance of CMB
              and lensing reconstruction power spectra". Phys. Rev. D 95, 043508, 2017.
            * ATC Collaboration. "The Atacama Cosmology Telescope: A Measurement of the DR6 CMB Lensing
              Power Spectrum and its Implications for Structure Growth". arXiv:2304.05202, 2023.

        :param MV: array corresponding to the MV estimator. Default is None
                    if no addition want to be done.
        :type MV: np array.
        """

        
        # Load the paths           
        fname_cls_map = os.path.join(self.mean_spectra_dir, f"cls_map_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{self.Nsim}.pkl")
        fname_RDN0 = os.path.join(self.results_dir, f"RDN0_analytical_iter_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        if os.path.isfile(fname_RDN0):
            RDN0 =  pl.load(open(fname_RDN0,"rb"))
            if MV is None:
                return RDN0
            else:
                # Include the RDN0 MV in the file
                RDN0['MVMV'] = MV
                pl.dump(RDN0, open(fname_RDN0,"wb"))
        else:
            st = time.time()
            # Load the angular power spectra of the simulations
            cls_map = pl.load(open(fname_cls_map,"rb"))
            # Calculate the analytical N0 bias for all the qe combinations
            N0_ana = self.analytical_N0()
            # Create the dictionary to store the results for RDN0
            RDN0_ana = {pair: np.zeros((self.Nsim, self.exp.lmax + 1)) for pair in self.noise_pairs}

            # Take into account that the Realization-dependent bias modification in lensquest is 
            # only implemented for the [1] terms.
            print('Calculating the semi-analytical realisation-dependent bias...', flush=True)
            aux_RDN0 = lensquest.quest_norm_RD_iterSims(self.lensedTheoryCls, self.meanCls, self.mean_sn, rdcl=cls_map, lmin=self.exp.lmin,
                                                            lmax=self.exp.lmax, lminCMB=self.exp.lmin, lmaxCMB=self.exp.lmax)[1]
            for pair in self.noise_pairs:
                RDN0_ana[pair] = aux_RDN0[pair] - 2*N0_ana[pair]

            # We save the two results in two different files
            pl.dump(RDN0_ana, open(fname_RDN0,"wb"))
            print('Time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - st)), flush=True)
            return RDN0_ana

    def N1(self, MV=None):
        """
        Calculate the N1 bias using an analytical and flat-sky approximation. Eventhough 
        we do not need all the QE, the function calculates all. For that reason, we are 
        going to save all of them.


        :param MV: array corresponding to the MV estimator. Default is None
                    if no addition want to be done.
        :type MV: np array.
        """

        fname_N1 = os.path.join(self.results_dir, f"N1_Nsim_{self.Nsim}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        if os.path.isfile(fname_N1):
            N1 = pl.load(open(fname_N1,"rb"))
            if MV is None:
                return N1
            else:
                # Include the N1 MV in the file
                N1['MVMV'] = MV
                pl.dump(N1, open(fname_N1,"wb"))
        else:
            print('Calculating the N1 bias...', flush=True)
            st = time.time()
            ind = {'TT': 0,'EE': 1,'EB': 2,'TE': 3,'TB': 4, 'BB':5}
            path_dir = f"{self.exp_dir}/{self.fg_str}_chance_{self.chance_corr}/results"
            
            
            # Fortran code starts from L=1 because in fortran the first element of an array is 1.
            # Padding with zeros is necessary to match the array size in the fortran code.
            Cl_pp = np.pad(self.cl_phi[1:], (0, 8000 - self.exp.lmax), 'constant')
            Cl_theory = np.pad(self.lensedTheoryCls[:, 1:], ((0, 0), (0, 8000 - self.exp.lmax)), 'constant')
            Cl_obs = np.pad(self.mean_sn[:, 1:], ((0, 0), (0, 8000 - self.exp.lmax)), 'constant')
            
            # Compute N1s and form MV using lensingbiases package
            n1_mat = LensingBiases.compute_n1_py(Cl_pp=Cl_pp, Cl_theory=Cl_theory, Cl_obs=Cl_obs,
                                                    lmin=self.exp.lmin, lmaxout=self.exp.lmax, 
                                                    lmax=self.exp.lmax, lmax_TT=self.exp.lmax,
                                                    lcorr_TT=0, tmp_output=path_dir)
            
            # Save the results in a dictionary
            # The N1 bias is normalized with the curved-sky analytical N0 bias.
            # Some differences have been observed at large scales when using the flat-sky normalization.
            # We exclude the BB, due to its low signal to noise ratio.
            R_ana = self.calculate_R_ana()
            N1_bias = {p: R_ana[p[:2]]*R_ana[p[2:]]*np.concatenate(([0, 0], n1_mat[ind[p[:2]]][ind[p[2:]]])) for p in self.pairs}
            
            # Save the N1 bias in a file
            pl.dump(N1_bias, open(fname_N1,"wb"))
            print('Time:', time.strftime("%H:%M:%S", time.gmtime(time.time() - st)), flush=True)

            return N1_bias
        
    def calculate_MV_weights(self):
        """
        Calculate the MV estimator weights of QE.
        """

        print("Calculating the MV estimator weights...", flush=True)
        MCN0 = self.job_calculate_MC_N0(mean=True)
        N1 = self.N1()

        # Calculate the MV weights
        weight = {qe: np.zeros(self.exp.lmax + 1) for qe in self.qe_list}
        nmv = np.zeros(self.exp.lmax + 1)

        nspec=len(self.qe_list)
        for L in tqdm(range(self.exp.lmin, self.exp.lmax+1)):
            mat=np.zeros((nspec,nspec))
            k = 0
            for i in range(nspec):
                for j in range(i, nspec):
                    index = self.pairs[k]
                    k += 1
                    if index in self.noise_pairs:
                        mat[i,j] = MCN0[index][L] + N1[index][L]
                    else:
                        mat[i,j] = N1[index][L]
                    if i != j:
                        mat[j,i] = mat[i,j] # The covariance matrix is symmetric
            try:
                mat=np.linalg.inv(mat)
                nmv[L]=1./np.sum(mat)
            except:
                print('Matrix singular for L='+str(L))
                nmv[L]=0.
            for i in range(len(self.qe_list)):
                weight[self.qe_list[i]][L]=nmv[L]*np.sum(mat[i,:])

            pl.dump(weight, open(f"{self.results_dir}/MV_weights_{self.Nsim}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl","wb"))

        return weight

    def run_qe_MF_N0_N1_MV(self):
        """
        Calculate the MV QE, the MV Mean-Field (MF) and the 
        MV N0, MCN0, RDN0 and N1 biases. 
        """

        print('Calculating the MV quadratic estimator...', flush=True)
        # First, calculate the weights for the MV estimator
        weights = self.calculate_MV_weights()
        
        # OPENMPI (the QE should be calculated before)
        for ind in range(self.Nsim):
            self.run_qe(ind, weights=weights)
        MCN0 = self.job_calculate_MC_N0(mean=False)

        # NO OPENMPI
        self.calculate_MF(weights=weights)
        N0 = self.analytical_N0()
        RDN0 = self.semianalytical_RDN0_iter()
        N1 = self.N1()

        # MV biases
        N0_MV = np.zeros(self.exp.lmax + 1)
        MCN0_MV = np.zeros((self.Nsim//2, self.exp.lmax + 1))
        RDN0_MV = np.zeros((self.Nsim, self.exp.lmax + 1))
        N1_MV = np.zeros(self.exp.lmax + 1)

        nspec=len(self.qe_list)
        mat_N0=np.zeros((nspec,nspec, self.exp.lmax+1))
        mat_N1=np.zeros((nspec,nspec, self.exp.lmax+1))

        # Filling the covariance matrices. Calculating the N1 and N0
        k = 0
        for i in range(nspec):
            for j in range(i, nspec):
                index = self.pairs[k]
                k += 1
                mat_N1[i,j] = N1[index]
                if index in self.noise_pairs:
                    mat_N0[i,j] = N0[index]

                # The covariance matrix is symmetric    
                if i != j:
                    mat_N1[j,i] = mat_N1[i,j]
                    if index in self.noise_pairs:
                        mat_N0[j,i] = mat_N0[i,j]

        w = np.array([weights[qe] for qe in self.qe_list])
        # Calculating the MCN0, RDN0 and N1 biases for the MV estimator
        N0_MV = np.sum(w[np.newaxis,:,:] * np.sum(mat_N0 * w[np.newaxis,:,:], axis=1), axis=1)[0]
        N1_MV = np.sum(w[np.newaxis,:,:] * np.sum(mat_N1 * w[np.newaxis, :,:], axis=1), axis=1)[0]

        # Calculating the MV estimator for the MCN0 for each simulation
        for n in tqdm(range(self.Nsim//2)):
            k = 0
            mat_MCN0=np.zeros((nspec,nspec, self.exp.lmax+1))
            for i in range(nspec):
                for j in range(i, nspec):
                    index = self.pairs[k]
                    k += 1
                    if index in self.noise_pairs:
                        mat_MCN0[i,j] = MCN0[index][n]
                        # The covariance matrix is symmetric    
                        if i != j:
                            mat_MCN0[j,i] = mat_MCN0[i,j]
            MCN0_MV[n] = np.sum(w[np.newaxis,:,:] * np.sum(mat_MCN0 * w[np.newaxis,:,:], axis=1), axis=1)[0]

        # Calculating the MV estimator for the RDN0 for each simulation
        for n in range(self.Nsim): 
            k = 0
            mat_RDN0=np.zeros((nspec,nspec, self.exp.lmax+1))
            for i in range(nspec):
                for j in range(i, nspec):
                    index = self.pairs[k]
                    k += 1
                    if index in self.noise_pairs:
                        mat_RDN0[i,j] = RDN0[index][n]
                        # The covariance matrix is symmetric    
                        if i != j:
                            mat_RDN0[j,i] = mat_RDN0[i,j]
            RDN0_MV[n] = np.sum(w[np.newaxis,:,:] * np.sum(mat_RDN0 * w[np.newaxis,:,:], axis=1), axis=1)[0]


        # Save the results in the corresponding files   
        MCN0 = self.job_calculate_MC_N0(MV=MCN0_MV)
        N0 = self.analytical_N0(MV=N0_MV)
        RDN0 = self.semianalytical_RDN0_iter(MV=RDN0_MV)
        N1 = self.N1(MV=N1_MV)
        
    def run_parallel_QE_MCN0(self):
        """
        Calculate the QE and MCN0 bias in parallel.
        """

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if rank == 0:
            print(f'Calculating the QE and MCN0 bias for {self.exp.name}', flush=True)
            self.mean_signal_noise()   
            self.mean_spectra_map()
            self.calculate_R_ana()
        
        comm.Barrier()

        # Load the attributes calculated in rank 0
        if rank != 0:
            self.mean_signal_noise()   
            self.mean_spectra_map()

        # Calculate the QE and the MCN0 bias
        self.job_run_qe()
        comm.Barrier()
        self.job_calculate_MC_N0()


    def calculate_SNR(self, qcl):
        """
        Function to calculate the Signal-to-Noise Ratio (SNR) of the QE estimators.

        :param qcl: dictionary containing qe debiased reconstructed power spectra.
        :type qcl: dict
        """

        fname_qcl_debiased_MC_norm_binned = os.path.join(self.results_dir, f"qcl_debiased_MC_norm_binned_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_MV.pkl")
        fname_amp_l_debiased_MC_norm_binned = os.path.join(self.results_dir, f"amp_l_debiased_MC_norm_binned_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_MV.pkl")
        fname_final_C_l_debiased_MC_norm_binned = os.path.join(self.results_dir, f"final_cl_b_debiased_MC_norm_binned_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_MV.pkl")
        # Bin the results
        binner = binning.multipole_binning(self.num_bins, lmin=self.exp.lmin, lmax=self.exp.lmax, spc=self.bin_opt)
        B = binner.bc

        L = np.arange(self.exp.lmax + 1)
        factor = (L*(L+1))**2/(2*np.pi)

        lens_theory_binned = binning.binning(factor*self.cl_phi, binner)
        clens_interp = scipy.interpolate.interp1d(L, factor*self.cl_phi, kind='cubic')
        qcl_binned = {qe: np.zeros((self.Nsim, self.num_bins)) for qe in qcl.keys()}
        amp_binned = {qe: np.zeros((self.Nsim, self.num_bins)) for qe in qcl.keys()}
        final_clb_binned = {qe: np.zeros((self.Nsim, self.num_bins)) for qe in qcl.keys()}

        C_phiphi_Lb = clens_interp(binner.bc)
        for qe in qcl.keys():
            for i in tqdm(range(self.Nsim)):
                qcl_binned[qe][i] = binning.binning(factor*qcl[qe][i], binner)
                amp_binned[qe][i] = qcl_binned[qe][i]/lens_theory_binned
                final_clb_binned[qe][i] = amp_binned[qe][i] * C_phiphi_Lb
        qcl_binned['B'] = binner.bc
        amp_binned['B'] = binner.bc
        final_clb_binned['B'] = binner.bc
        
        pl.dump(qcl_binned, open(fname_qcl_debiased_MC_norm_binned,"wb"))
        pl.dump(amp_binned, open(fname_amp_l_debiased_MC_norm_binned,"wb"))
        pl.dump(final_clb_binned, open(fname_final_C_l_debiased_MC_norm_binned,"wb"))

        # Plot the binned results
        print('Plotting the binned results...', flush=True)
        fig, axs = plt.subplots(8, 2, figsize=(16, 30))
        # Flatten the 2D array of subplots for easier iteration
        axs = axs.flatten()
        # Iterate over subplots and plot data
        for ax, qe in zip(axs, qcl_binned.keys()):
            ax.errorbar(B, y=np.mean(final_clb_binned[qe], axis=0), yerr=np.std(final_clb_binned[qe], axis=0), fmt='o', capsize=5, label=f"qcls {qe}")
            ax.plot(factor*self.cl_phi, label='clpp input')
            ax.set_xlim([2, self.exp.lmax])
            ax.set_ylim([-1e-7, 3e-7])
            ax.semilogx()
            # ax.semilogy()
            ax.legend()
        plt.savefig(f"{self.plots_dir}/mean_qe_{self.fg_str}_{self.exp.name}_chance_{self.chance_corr}_nbins_{self.num_bins}.pdf")

        # Calculate the SNR
        SN_full = {qe: np.zeros(self.num_bins) for qe in self.qe_list_MV}

        for i in tqdm(range(len(self.qe_list_MV))):
            for nbins_cut in range(2, self.num_bins):
                stat = ana.statistics(ocl=1.,scl=qcl_binned[self.qe_list_MV[i]+self.qe_list_MV[i]][:, :nbins_cut])
                stat.get_amp(fcl=qcl_binned[self.qe_list_MV[i]+self.qe_list_MV[i]][:, :nbins_cut].mean(axis=0), diag=False)
                SN_full[self.qe_list_MV[i]][nbins_cut] = 1/stat.sA

        # Plot the SNR
        print('Plotting the S/N...', flush=True)
        plt.figure(figsize=(8,6))
        for qe in self.qe_list_MV:
            plt.plot(SN_full[qe], label=f"Full {qe} = {np.max(SN_full[qe][2:]):.2f}")
        plt.xlim([0, self.num_bins-1])
        plt.ylim([-2, 85])
        plt.xlabel('Bins')
        plt.ylabel('S/N')
        plt.legend()
        plt.savefig(f"{self.plots_dir}/SN_full_{self.fg_str}_{self.exp.name}_chance_{self.chance_corr}_nbins_{self.num_bins}.pdf")
    

    def run_qcl(self):
        """
        Calculate the unbiased and normalized angular power spectrum of the lensing potential
        for the different QE estimators (including the Minimum-Variance (MV)).

        It assumes that the function run_parallel_QE_MCN0() or the functions job_run_qe() and 
        job_calculate_MC_N0() have been run before (ideally in parallel).
        """
        
        print(f'Reconstructing {self.exp.name} lensing power spectra', flush=True)
        fname_qcl_no_MF = os.path.join(self.results_dir, f"qcl_no_MF_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}.pkl")
        fname_qcl_debiased_MC_norm = os.path.join(self.results_dir, f"qcl_debiased_MC_norm_{'_'.join(self.qe_list)}_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_MV.pkl")
        if os.path.isfile(fname_qcl_debiased_MC_norm):
            qcl = pl.load(open(fname_qcl_debiased_MC_norm,"rb"))
            self.calculate_SNR(qcl)
        else:
            pairs_MV = self.pairs + ['MVMV']
            qe = {pair: np.zeros((self.Nsim, self.exp.lmax + 1)) for pair in pairs_MV}
            qcl_no_MF = {pair: np.zeros((self.Nsim, self.exp.lmax + 1)) for pair in pairs_MV}
            qcl_no_MF_norm = {pair: np.zeros((self.Nsim, self.exp.lmax + 1)) for pair in pairs_MV}

            # Load the mean field power spectra
            self.mean_signal_noise()   
            self.mean_spectra_map()

            # Calculate the MF, N0, RDN0 and N1 biases
            self.calculate_MF()
            self.semianalytical_RDN0_iter()
            self.N1()

            # Calculate the QE MV and MV MF, N0, RDN0 and N1 biases
            self.run_qe_MF_N0_N1_MV()
            # Calculate the MC normalisation correction (including the MV estimator)
            R_MC = self.calculate_R_MC()

            # Load the data, now with the MV estimator included
            MCN0 = self.job_calculate_MC_N0()
            RDN0 = self.semianalytical_RDN0_iter()
            N1 = self.N1()
            
            print('Calculating qcl for all the simulations...', flush=True)
            f1 = self.Nsim/(self.Nsim-2)
            for ind in tqdm(range(self.Nsim)):
                for pair in pairs_MV:
                    norm_corr = R_MC['MV'] 
                    # norm_corr = np.sqrt(R_MC[pair[:2]]*R_MC[pair[2:]])
                    # MF subtraction at the map level
                    phi1 = f1*(self.run_qe(ind)[pair[:2]] - self.phi_MF_1[pair[:2]])
                    phi2 = self.run_qe(ind)[pair[2:]] - self.phi_MF_2[pair[2:]]
                    qe[pair][ind] = hp.alm2cl(phi1, phi2, lmax=self.exp.lmax)/self.w4
                    if pair in self.noise_pairs + ['MVMV']:
                        qcl_no_MF[pair][ind] = qe[pair][ind] - MCN0[pair] - RDN0[pair][ind] - N1[pair]
                    else:
                        qcl_no_MF[pair][ind] = qe[pair][ind] - N1[pair]
                    qcl_no_MF_norm[pair][ind] = qcl_no_MF[pair][ind]/norm_corr
                   
            pl.dump(qcl_no_MF, open(fname_qcl_no_MF,"wb"))
            pl.dump(qcl_no_MF_norm, open(fname_qcl_debiased_MC_norm,"wb"))

            self.calculate_SNR(qcl_no_MF_norm)


def alm_change_lmax(alm, new_lmax):
    """
    Changes the lmax of the alms according to Healpix ordering.

    :param alm: spherical harmonic coefficients to be changed.
    :param new_lmax: the new lmax to be changed.
    :return: new alms
    """
    new_alm = np.zeros(hp.Alm.getsize(new_lmax), dtype='complex_')
    lmax = hp.Alm.getlmax(len(alm))
    min_lmax = np.min([lmax, new_lmax])
    for m in range(min_lmax + 1):
        aux1 = hp.Alm.getidx(new_lmax, m, m)
        aux2 = hp.Alm.getidx(lmax, m, m)
        new_alm[aux1:aux1 + min_lmax - m + 1] = alm[aux2:aux2 + min_lmax - m + 1]
    return new_alm


if __name__ == "__main__":
    ini_file = sys.argv[1]
    qe = sys.argv[2]
    recons = Reconstruction.from_ini(ini_file)

    if qe == 'True':
        # Run the QE and MCN0 bias in parallel.
        recons.run_parallel_QE_MCN0()
    else:
        # Run the remaining things in serial 
        # after the QE and MCN0 bias have been calculated.
        recons.run_qcl()