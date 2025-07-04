"""
The filtering class takes as an input the HILC cleaned maps and performs a C-inverse
filtering  (Wiener filter) to them. Two different implementations are provided:
the pixel-based filtering and the harmonic-based filtering.
 
The pixel-based filtering is the exact one, but computationally expensive. It is the one 
used in the paper "LiteBIRD Science Goals and Forecasts: A full-sky measurement of 
gravitational lensing of CMB" as part of a LiteBIRD collaboration paper led by Anto 
Lonappan.

The harmonic filtering is an approximation to the exact one. It is computationally cheaper
and assumes that the covariance matrix is diagonal in harmonic space. It was used previously
in "The Atacama Cosmology Telescope (ACT): A Measurement of the DR6 CMB Lensing Power Spectrum and
its Implications for Structure Growth". The implementation was slighly modified to be used in a
full-sky case, as LiteBIRD, unlike ACT which is a ground-base experiment.

Author: Miguel Ruiz-Granda
"""

import os
import numpy as np
import healpy as hp
from astropy.io import ascii
import pymaster as nmt
from tqdm import tqdm
import pickle as pl
import curvedsky as cs # only needed if you want to use the pixel-based filtering
import mpi4py.MPI as MPI
import time
import matplotlib.pyplot as plt
import experiment as exp
import configparser
import sys

class Filtering:
    """
    A class that filters the CMB maps using the pixel-based and harmonic-based filtering.

    """

    def __init__(self, experiment, fileUnlensedCls, fileLensedCls, alms_dir, fg_str, Nsim, 
                 parallel, chance_corr, filename_HILC, mask_fname, fsky_mask = 80, apotype="C1",
                   aposcale = 2.0):
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
        :param alms_dir: name of the directory where all the simulations are stored.
        :type alms_dir: str
        :param fg_str: foreground model to be used in the simulations.
        :type fg_str: str
        :param Nsim: number of simulations.
        :type Nsim: int
        :param parallel: whether to run the simulations in parallel or not.
        :type parallel: bool
        :param chance_corr: whether to include the chance correlation in the simulations.
        :type chance_corr: bool
        :param filename: name of the file used for the simulations.
        :type filename: str
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

        # Store all the data regarding the experiment and the HILC cleaning.
        self.exp = experiment
        self.Nsim = Nsim
        self.parallel = parallel
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

         # Parameters for producing the foregrounds simulations using pysm3
        # Two cases are considered: 'no_fg' (no foregrounds case) and  'sx_dy'
        # foregrounds case in which x is the syncrotron model number and y is
        #  the dust model number.
        self.fg_str = fg_str

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
        parallel = config.getboolean('Filtering', 'parallel')
        mask_fname = filt['mask_fname']
        fsky_mask = int(filt['fsky_mask'])
        apotype = filt['apotype']
        aposcale = float(filt['aposcale'])
        

        return cls(experiment, fileUnlensedCls, fileLensedCls, alms_dir, fg_str, Nsim, parallel,
                    chance_corr, filename_HILC, mask_fname, fsky_mask=fsky_mask, apotype=apotype,
                      aposcale=aposcale)
    
    def cli(self, cl):
        """
        Returns the inverse of the input array, where the input array is greater than zero.
        """
        ret = np.zeros_like(cl)
        ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
        return ret
    
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

    def harmonic_filtering(self):
        """
        Filters the CMB maps using the harmonic-based filtering implementation.
        """

        if self.parallel:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()
            jobs = np.arange(rank, self.Nsim, size)
            st = time.time()
            if rank == 0:
                print(f"Running the harmonic filter for {self.exp.name} with a mask fsky={np.mean(self.apomask)}", flush=True)
            for index in tqdm(jobs) if rank == 0 else jobs:
                self.harmonic_filtering_index(index, self.mean_sn)
            ft = time.gmtime(time.time() - st)
            comm.Barrier()
            print(f"Time rank {rank}:", time.strftime("%H:%M:%S", ft), flush=True)
            if rank == 0:
                self.plot_filtered_spectra(0)
        else:
            print(f"Running the harmonic filter for {self.exp.name}", flush=True)
            for index in tqdm(range(self.Nsim)):
                self.harmonic_filtering_index(index, self.mean_sn)
            self.plot_filtered_spectra(0)


    def harmonic_filtering_index(self, index, cls_mean):
        """
        Filters the CMB maps using the harmonic-based filtering implementation given 
        a map index.

        """

        fname_map = f"{self.harmonic_filter_dir}/Wiener_filtered_TEB_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{index:04d}.pkl"
        fname_cls = f"{self.harmonic_filter_dir}/cls_wiener_filtered_TEB_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{index:04d}.pkl"
        if not os.path.exists(fname_map):
            # Load HILC cleaned simulation and calculate their alms coefficients
            fname_HILC = f"{self.sims_HILC_dir}/cleaned_maps_{self.filename_HILC}_{index:04d}.pkl"
            alms = pl.load(open(fname_HILC,'rb'))
            TQU = hp.alm2map(alms, nside=self.exp.nside)
            TQU_masked = np.array([self.apomask*TQU[0], self.apomask*TQU[1], self.apomask*TQU[2]])

            # TQU from filtered alms
            alms[0] = hp.almxfl(alms[0], self.cli(cls_mean[0]))
            alms[1] = hp.almxfl(alms[1], self.cli(cls_mean[1]))
            alms[2] = hp.almxfl(alms[2], self.cli(cls_mean[2]))
            TQU_filtered = hp.alm2map(alms, nside=self.exp.nside)
            QU_filtered_masked = np.array([self.apomask*TQU_filtered[1], self.apomask*TQU_filtered[2]])

            # For temperature and E-mode polarization we use the unfiltered maps
            tlm, elm = hp.map2alm(TQU_masked, lmax=self.exp.lmax)[:2, :]
            # We have to divide by the spectrum (harmonic filtering)
            tlm = hp.almxfl(tlm, self.cli(cls_mean[0]))
            elm = hp.almxfl(elm, self.cli(cls_mean[1]))

            # For B-mode polarization we use the filtered maps and then we mask them
            blm = hp.map2alm_spin(QU_filtered_masked, lmax=self.exp.lmax, spin=2)[1]
            pl.dump([tlm, elm, blm], open(fname_map,'wb'))
            # We compute the power spectra of the filtered maps
            cls = hp.alm2cl([tlm, elm, blm], lmax=self.exp.lmax)[:4, :]
            pl.dump(cls, open(fname_cls,'wb'))


    def pixel_filtering(self, use_mask=True):
        """
        Filters the CMB maps using the pixel-based filtering implemented in cmblensplus module.

        It was used in the paper "LiteBIRD Science Goals and Forecasts: A full-sky measurement of
        gravitational lensing of CMB" as part of a LiteBIRD collaboration paper led by Anto Lonappan.

        This function was used for comparison purposes with the harmonic-based filtering implementation.
        TODO: Be aware that there is an additional Tcmb factor that needs to be corrected somewhere in this
        function to match the harmonic-based filtering implementation.

        
        :param mask: whether to mask the sky or not.
        :type mask: bool
        """

        if use_mask:
            mask = self.mask
        else:
            mask = np.ones_like(self.mask)

        # Mean noise over the Nsim simulations
        clsTEBn = self.mean_noise(self.Nsims_noise)/self.T_CMB**2

        # Beam used to convolve the inv-noise used in the C-inv procedure.
        beam_size = 15 # in arcmins. 
        beam = hp.gauss_beam(fwhm=np.radians(beam_size/60), lmax=self.exp.lmax)
        Bl = np.reshape(beam,(1,self.exp.lmax+1))
        convNoise = np.reshape(np.array((self.cli(clsTEBn[0, :self.exp.lmax+1]*beam**2),
                                         self.cli(clsTEBn[1, :self.exp.lmax+1]*beam**2),
                                          self.cli(clsTEBn[2, :self.exp.lmax+1]*beam**2))),(3,1,self.exp.lmax+1))

        # Load the theory power spectra
        wcl = np.reshape(np.array([self.lensCls['2:TT'][:self.exp.lmax+1], self.lensCls['3:EE'][:self.exp.lmax+1],
                        self.lensCls['5:BB'][:self.exp.lmax+1]]), (3,self.exp.lmax+1))

        # In the pixel-based filtering we assign infinite noise to the masked pixels.
        ninv = np.reshape(np.array((mask, mask, mask)),(3,1,self.exp.npix))

        print(f"Running the pixel c-inv filter with mask={use_mask}", flush=True)
        for index in tqdm(range(self.Nsims)):
            # Load HILC cleaned simulation
            TEB = pl.load(open(f"{self.sims_HILC_dir}/sims_{index:04d}.pkl",'rb'))

            # Convolve the component separated map with the beam and change the convolved ALMs to MAPS
            hp.almxfl(TEB[0], beam, inplace=True)
            hp.almxfl(TEB[1], beam, inplace=True)
            hp.almxfl(TEB[2], beam, inplace=True)
            TQU_convolved = hp.alm2map(TEB, nside=self.exp.nside)

            # C inv Filter for the component separated maps
            TQU_masked = np.reshape(np.array((TQU_convolved[0]*mask, TQU_convolved[1]*mask,
                                              TQU_convolved[2]*mask)),(3,1,self.exp.npix))/self.T_CMB

            # Wiener filtered Cinv

            fname_map = f"{self.pixel_filter_dir}/Wiener_TQU_{index:04d}_mask_{use_mask}.pkl"

            if not os.path.exists(fname_map):
                iterations = 1000
                eps = 1e-5
                T_W, E_W, B_W = cs.cninv.cnfilter_freq(3,1,int(self.exp.nside),int(self.exp.lmax),wcl,Bl,ninv,TQU_masked,chn=1,
                                                       itns=[iterations],eps=[eps],filter='',ro=5,inl=convNoise,stat='',
                                                       verbose=True)

                T_W_map = cs.utils.hp_alm2map(self.exp.nside,self.exp.lmax,self.exp.lmax, T_W)
                Q_W_map, U_W_map = cs.utils.hp_alm2map_spin(self.exp.nside,self.exp.lmax,self.exp.lmax,2, E_W, B_W)
                pl.dump([T_W_map, Q_W_map, U_W_map], open(fname_map,'wb'))


    def plot_filtered_spectra(self, index):
        """
        Plot the theoretical and simulation angular power spectra for comparison purposes for a single map.

        :param map: map whose spectrum is going to be plotted.
        :type map: Map
        """

        L = np.arange(0, self.exp.lmax + 1)
        factor = (L * (L + 1)) / (2 * np.pi)
        fname_cls = f"{self.harmonic_filter_dir}/cls_wiener_filtered_TEB_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{index:04d}.pkl"
        cls_filt = pl.load(open(fname_cls, 'rb'))
        
        # Theory filtered power spectra
        cls_theo_filt = np.vstack((self.cli(self.mean_sn[:3, :]), self.lensedTheoryCls[3]*self.cli(self.mean_sn[0, :]*self.mean_sn[1, :])))

        # Simulated vs theoretical power spectra after HILC.
        fig, ax = plt.subplots(2, 2, figsize=(12, 6))
        ind = ['TT', 'EE', 'BB', 'TE']
        i = 0
        for ax in ax.reshape(-1):
            ax.plot(L, factor * cls_filt[i], color='b', label=f"Filtered {index} {ind[i]}")
            ax.plot(L, factor * self.w2 * cls_theo_filt[i], color='r', label=f"Theory {ind[i]}")
            ax.set_ylabel(r'$\frac{[\ell(\ell +1)]}{2\pi}C_\ell$', size=18)
            ax.set_xlim([2, self.exp.lmax])
            if i != 3:
                ax.semilogy()
            ax.semilogx()
            ax.set_xlim([2, self.exp.lmax])
            ax.legend(fontsize=10)
            i += 1
        plt.savefig(f"{self.plots_dir}/filtered_spectra_{self.fg_str}_{self.exp.name}_fsky_{self.mask_fsky}_{index:04d}.pdf")

if __name__ == "__main__":
    ini_file = sys.argv[1]
    filt = Filtering.from_ini(ini_file)
    filt.mean_signal_noise()
    # Run the filtering over simulations.
    filt.harmonic_filtering()
    