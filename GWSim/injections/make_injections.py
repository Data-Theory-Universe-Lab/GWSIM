import numpy as np
import lal
import h5py
import pickle
import lalsimulation as lalsim
from scipy.interpolate import interp1d
from scipy.stats import ncx2
from scipy.integrate import quad
import pkg_resources
from GWSim.catalog.real_catalog import RealCatalog
from GWSim.universe.create_universe import Universe
from GWSim.cosmology import Cosmology
from GWSim.injections.distributions import Redshift
from GWSim.utils import Interpolate_function,Integrate_1d
import bilby as bl
import logging
import multiprocess as mp
import time
import os
import signal
import itertools
import copy
import astropy
from bilby.gw import utils as gwutils

class Counter(object):
    def __init__(self, initval = 0, maxval = 0):
        self.val = mp.Value('i',initval)
        self.maxval = mp.Value('i',maxval)
        self.lock = mp.Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value

    def max_value(self):
        with self.lock:
            return self.maxval.value


class Select_galaxies(object):

    def __init__(self,file_path,lumi_weighting,red_weighting,N,red_evol_parameters, cosmology = None, galaxy_cat_parameters = None,fake_universe=False ):

        #Added z_err and cosmology to be passed
        self.z_err = None
        self.cosmo = cosmology #cosmology is a object
        self.galaxy_parameters = galaxy_cat_parameters #Dictionary for galaxy parameters
        self.fraction = 1
        self.zcut = red_evol_parameters['zcut']
        if fake_universe:
            self.select = self.select_fake_universe
        else:
            self.select = self.select_zbins
        
        try:
            if file_path[-2:]=='.p':
                data = pickle.load(open(file_path,'rb'))
                if self.cosmo is None:
                    # we consider the cosmological model of the fake universe
                    # else we use the cosmology object of the __init__ function
                    self.cosmo = data.cosmology
                self.ra = np.array(data.ra)
                self.dec = np.array(data.dec)
                self.z = np.array(data.z)
                self.abs_magnitudes = np.array(data.abs_magn)

            elif file_path[-5:]=='.hdf5':
                data = h5py.File(file_path, "r")
                try:
                    cosmo_parameters = pickle.load(open(file_path[:-5]+'_cosmo_parameters.p','rb'))
                except:
                    raise ValueError("Cant find cosmological paramaters file.")
                if self.cosmo is None:
                    self.cosmo = Cosmology(cosmo_parameters)
                self.ra = np.array(data['ra'])
                self.dec = np.array(data['dec'])
                self.z =  np.array(data['z'])
                self.abs_magnitudes = np.array(data['abs_magn'])

            elif file_path=='/home/federico.stachurski/LVK/load_catalog/reduced_catalogs/':

                # Cosmology described in CosmoHub for MiceCatv2
                if self.cosmo is None:
                    cosmo_dic = {'H0': 70, 'w0':-1, 'Omega_m':0.25, 'cosmo_model': 'flatLCDM'}
                    self.cosmo = Cosmology(cosmo_dic)

                self.zmax = self.galaxy_parameters['zmax']
                self.zmin = self.galaxy_parameters['zmin']
                self.z_err_perc = self.galaxy_parameters['z_err_perc']
                self.magnitude_band = self.galaxy_parameters['magnitude_band']
                self.magnitude_type = self.galaxy_parameters['magnitude_type']
                self.NSIDE = self.galaxy_parameters['NSIDE']
                self.Nth = self.galaxy_parameters['Nth']
                self.catalog_name = self.galaxy_parameters['catalog_name']
                self.mth_cat = self.galaxy_parameters['mth_cat']


                if self.zcut == 'None':
                    self.zmax = 2.0
                else:
                    self.zmax = self.zcut
                print("Reading catalog with zcut={}".format(self.zmax))
                data = RealCatalog(pathname = file_path, catalog_name = self.catalog_name, zmax = self.zmax,
                                   zmin = self.zmin, z_err_perc = self.z_err_perc, magnitude_band = self.magnitude_band ,
                                   magnitude_type = self.magnitude_type, NSIDE = self.NSIDE, Nth = self.Nth, mth_cat = self.mth_cat)
                self.fraction = data.fraction_cat
                data = data.catalog_data

                self.ra = np.deg2rad(np.array(data['ra']))
                self.dec = np.deg2rad(np.array(data['dec']))
                self.z =  np.array(data['z'])
                print("Got {} galaxies in catalog, with zcut={}.".format(len(self.z),self.zmax))
                if self.magnitude_type == 'apparent':
                    #Apply magnitude redshift evolution
                    obs_mag_evl = np.array(data['app_magn'] - 0.8 * (np.arctan(1.5 * self.z) - 0.1489))

                    self.abs_magnitudes = self.cosmo.M_mdL(obs_mag_evl, self.z)

                else:

                    self.abs_magnitudes = np.array(data['abs_magn'])

                self.z_err = np.array(data['sigma_z'])
                del data



            else:
                raise ValueError("Universe/catalog file not supported for path {}.".format(file_path))
        except:
            raise ValueError("Can't find the universe/catalog file for path {}.".format(file_path))

        if self.zcut!='None':
            self.zcut = float(self.zcut)
            if self.zcut<np.max(self.z):
                print("Applying redshift cut of {} to universe with zmax = {}".format(self.zcut,round(np.max(self.z),2)))
                idx = np.where(self.z<=self.zcut)
                self.z = self.z[idx]
                self.ra = self.ra[idx]
                self.dec = self.dec[idx]
                self.abs_magnitudes = self.abs_magnitudes[idx]
            else:
                print("Redshift cut of {} is bigger or equal to zmax = {} of universe. Continuing without applying cut.".format(self.zcut,round(np.max(self.z),2)))
                self.zcut = np.max(self.z)
        else:
            self.zcut = np.max(self.z)
        red_evol_parameters['zcut'] = self.zcut
        print("We will use {} galaxies of catalog for hosting GW events between zmin={} and zmax={}.".format(len(self.z),np.min(self.z),np.max(self.z)))

        self.N_requested = 0
        self.N = N
        self.lumi_weighting = lumi_weighting
        self.red_weighting = red_weighting
        self.T_obs = red_evol_parameters['T_obs']
        if self.T_obs == 0: self.N_requested = float(self.N)
        self.redshift_evolution = Redshift(red_evol_parameters,self.cosmo,self.red_weighting)
        self.red_evol_parameters = red_evol_parameters
        if (self.N_requested == 0) and (self.T_obs == 0):
            raise ValueError("Please provide either a T_obs or a number of injections.")


    def MergersPerYearPerGpc3(self,z):

        return self.redshift_evolution.evolution(z)*self.cosmo.volume_z_and_time(z)

    def select_zbins(self):

        self.weights = np.ones(len(self.z))
        self.weights /= np.sum(self.weights)

        if self.lumi_weighting:
            print("Setting up luminosity weights for galaxy selection")
            self.weights *= 10**(-0.4*self.abs_magnitudes)

        z_step = 0.01
        N_z = int((self.zcut-np.min(self.z))/z_step)
        z_sub_set = np.linspace(np.min(self.z),self.zcut,N_z)
        f_mergers = np.zeros(len(z_sub_set)-1,dtype=float)
        mergers = np.zeros(len(z_sub_set)-1,dtype=int)
        for i in range(len(z_sub_set)-1):
            f_mergers_in_bin = quad(self.MergersPerYearPerGpc3,z_sub_set[i],z_sub_set[i+1])[0]
            f_mergers[i] = np.random.poisson(f_mergers_in_bin)
            
        if self.T_obs > 0:
            T_obs = float(self.T_obs)
            f_mergers *= T_obs * self.fraction
            for i in range(len(z_sub_set)-1):
                mergers[i] = int(np.round(f_mergers[i]))
            self.N = int(np.sum(mergers))
            print("For T_obs = {} yr and R0 = {} Gpc^-3 yr^-1 there are in total {} mergers from zmin = {} up to zmax = {}".format(
                T_obs,self.red_evol_parameters['R0'],self.N,round(np.min(self.z),2),round(self.zcut,2)))
            if len(self.z)<self.N:
                raise ValueError("Careful: the number of mergers is higher than the number of galaxies... You should check your input universe or cosmology.")        
        else:
            N_obs = np.sum(f_mergers)
            f_mergers *= (self.N_requested*100./N_obs) # draw 1e2 more mergers than requested, hoping we'll have ~ N_requested events above SNR threshold
            for i in range(len(z_sub_set)-1):
                mergers[i] = int(np.round(f_mergers[i]))
            self.N = int(np.sum(mergers))
        
        if self.N <= 0:
            raise ValueError("No mergers expected in this universe. Exiting")
            
        indexes = np.full(self.N,-1,dtype=int)
        print("Selecting merger hosts...")
        n = 0
        for i in range(len(z_sub_set)-1):
            idx = np.where((self.z>=z_sub_set[i])&(self.z<z_sub_set[i+1]))[0]
            print("#gal in zbin [{} ; {}]: {}".format(z_sub_set[i],z_sub_set[i+1],len(idx)))

        for i in range(len(z_sub_set)-1):
            if mergers[i]<= 0: continue
            idx = np.where((self.z>=z_sub_set[i])&(self.z<z_sub_set[i+1]))[0]
            print("bin with merger: #gal in zbin [{} ; {}]: {}".format(z_sub_set[i],z_sub_set[i+1],len(idx)))
            if len(idx)<mergers[i]:
                print("Problem: ask for {} mergers but {} galaxies are availabe... Force the number of mergers to the number of galaxies.".format(mergers[i],len(idx)))
                mergers[i] = len(idx)
            if len(idx)==0:
                print("No galaxies in redshift bin [{} ; {}]... skip z-bin".format(z_sub_set[i],z_sub_set[i+1]))
                continue
            random_galaxies = np.random.choice(len(idx),size=mergers[i],replace=True,p=self.weights[idx]/np.sum(self.weights[idx]))
            indexes[n:n+mergers[i]] = idx[random_galaxies]
            n += mergers[i]
                
        if n != self.N:
            print("Warning: {} host galaxies were randomly selected instead of {}. Maybe a problem with the galaxy catalog or input parameters".format(n,self.N))
            self.N = n
        ww = np.where(indexes >= 0)[0]
        indexes = indexes[ww]
            
        np.random.shuffle(indexes)

        if self.z_err is None:
            ras, decs, zs, abs_magns = self.ra[indexes], self.dec[indexes], self.z[indexes], self.abs_magnitudes[indexes]
        else:
            ras, decs, abs_magns = self.ra[indexes], self.dec[indexes], self.abs_magnitudes[indexes]
            zs = np.random.normal(self.z[indexes], self.z_err[indexes])



        ras = np.array(ras,dtype=np.float64)
        decs = np.array(decs,dtype=np.float64)
        zs =  np.array(zs,dtype=np.float64)
        abs_magns = np.array(abs_magns,dtype=np.float64)
        dls = self.cosmo.dl_zH0(zs)
        galaxies_parameteters = dict(ras=ras,decs=decs,zs=zs,abs_magns=abs_magns,dls=dls)
        return galaxies_parameteters

    def select_fake_universe(self):

        self.N = np.random.poisson(quad(self.MergersPerYearPerGpc3,0,self.zcut)[0])
        print("Nmergers/year/Gpc^3: {}".format(self.N))
        if self.T_obs > 0:
            self.N = round(self.N*self.T_obs)
            print("For T_obs = {} yr and R0 = {} Gpc^-3 yr^-1 there are in total {} mergers from zmin = {} up to zmax = {}".format(
                self.T_obs,self.red_evol_parameters['R0'],self.N,round(np.min(self.z),2),round(self.zcut,2)))                    
        else:
            self.N = round(self.N_requested*100) # draw 1e2 more mergers than requested, hoping we'll have ~ N_requested events above SNR threshold
                
        self.weights = self.redshift_evolution.evolution(self.z)/(1+self.z)
        if self.lumi_weighting:
            print("Setting up luminosity weights for galaxy selection")
            self.weights *= 10**(-0.4*self.abs_magnitudes)
        self.weights /= np.sum(self.weights)
        indexes = np.random.choice(len(self.z), size=self.N, replace=True, p=self.weights)

        print("{} host galaxies were randomly selected".format(self.N))

        if self.z_err is None:
            ras, decs, zs, abs_magns = self.ra[indexes], self.dec[indexes], self.z[indexes], self.abs_magnitudes[indexes]
        else:
            ras, decs, abs_magns = self.ra[indexes], self.dec[indexes], self.abs_magnitudes[indexes]
            zs = np.random.normal(self.z[indexes], self.z_err[indexes])

        ras = np.array(ras,dtype=np.float64)
        decs = np.array(decs,dtype=np.float64)
        zs =  np.array(zs,dtype=np.float64)
        abs_magns = np.array(abs_magns,dtype=np.float64)
        dls = self.cosmo.dl_zH0(zs)
        galaxies_parameteters = dict(ras=ras,decs=decs,zs=zs,abs_magns=abs_magns,dls=dls)
        return galaxies_parameteters
    
class Select_events(object):
    def __init__(self,pop_parameters,galaxies_parameters,det_parameters,pools=1):

        self.N = det_parameters['N']
        self.Ntot_mergers = pop_parameters['N']
        self.user_request = False
        if self.N > self.Ntot_mergers:
            print("Warning: you asked for more events than the numbers of mergers. Setting N=Nmergers.")
            self.N = self.Ntot_mergers
        if self.N < self.Ntot_mergers:
            # self.N is the number of events above snr_threshold
            self.user_request = True
            self.Counter = Counter(0,self.N) # min, max
            print("init counter: {} -> {}".format(self.Counter.value(),self.Counter.max_value()))
        else:
            # self.N is the total number of mergers
            self.Counter = Counter(0,self.Ntot_mergers) # min, max
            print("init counter: {} -> {}".format(self.Counter.value(),self.Counter.max_value()))
        q = np.random.rand(self.Ntot_mergers)
        self.incs = np.arccos(2.0*q - 1.0)
        self.psis = np.random.rand(self.Ntot_mergers)*2.0*np.pi
        self.phis = np.random.rand(self.Ntot_mergers)*2.0*np.pi
        self.f_min = 20
        self.sampling_frequency = 4096
        self.f_max = self.sampling_frequency/2
        self.ras = galaxies_parameters['ras']
        self.decs = galaxies_parameters['decs']
        self.T_obs = det_parameters['T_obs']
        self.m1s = copy.deepcopy(pop_parameters['m1s'])
        self.m2s = copy.deepcopy(pop_parameters['m2s'])
        self.chi_1 = pop_parameters['chi_1']
        self.chi_2 = pop_parameters['chi_2']
        self.theta_1 = pop_parameters['theta_1']
        self.theta_2 = pop_parameters['theta_2']
        self.zs = galaxies_parameters['zs']
        self.m1d = self.m1s*(1+self.zs)
        self.m2d = self.m2s*(1+self.zs)
        self.dls = galaxies_parameters['dls']
        self.asd = det_parameters['asd']
        self.detectors = det_parameters['detectors']
        self.det_combination = det_parameters['det_combination']
        self.snr_threshold = det_parameters['snr_network']
        self.snr_single_threshold = det_parameters['snr_single']
        self.geocent_time = np.random.rand(self.Ntot_mergers)*86400.
        self.pools = pools
        self.manager = mp.Manager()

        if (self.asd == 'O1') and ('V1' in self.detectors): self.detectors.remove('V1')

        duty_factors = det_parameters['duty_factors']
        days_of_run = det_parameters['days_of_runs']

        if self.asd == None:
            self.duty_factor = {'O4':{},'O3':{},'O2':{},'O1':{}}
        else:
            self.duty_factor={self.asd:{}}

        self.days_of_runs = {}
        for psd in self.duty_factor:
            self.days_of_runs[psd] = days_of_run[psd]

        if self.det_combination==True:
            for p in self.duty_factor:
                for d in duty_factors[p]:
                    if d in self.detectors:
                        self.duty_factor[p][d] = duty_factors[p][d]
                    else:
                        self.duty_factor[p][d] = -1.0

        else:
            for p in self.duty_factor:
                for d in duty_factors[p]:
                    if d in self.detectors:
                        self.duty_factor[p][d] = 1.0
                    else:
                        self.duty_factor[p][d] = -1.0
                    if p=='O1' and d=='V1':
                        self.duty_factor[p][d] = -1.0

        total_days = 0
        for key in self.days_of_runs:
            total_days+=self.days_of_runs[key]
        self.prob_of_run = {}
        for key in self.days_of_runs:
            self.prob_of_run[key] = self.days_of_runs[key]/total_days
        self.psds = []
        self.dets = []

        p = np.random.rand(self.Ntot_mergers)

        if self.asd == None:
            self.asds = {'O4':{},'O3':{},'O2':{},'O1':{}}
            self.ASD_data = {'O4':{},'O3':{},'O2':{},'O1':{}}

            for i in range(self.Ntot_mergers):
                if 0<=p[i]<=self.prob_of_run['O1']:
                    psd = 'O1'
                elif self.prob_of_run['O1']<p[i]<=self.prob_of_run['O1']+self.prob_of_run['O2']:
                    psd = 'O2'
                elif self.prob_of_run['O1']+self.prob_of_run['O2']<p[i]<=self.prob_of_run['O1']+self.prob_of_run['O2']+self.prob_of_run['O3']:
                    psd = 'O3'
                else:
                    psd = 'O4'
                self.psds.append(psd)
        else:
            self.asds = {self.asd:{}}
            self.ASD_data = {self.asd:{}}

            for i in range(self.Ntot_mergers):
                self.psds.append(self.asd)


        for i in range(self.Ntot_mergers):
            d = []
            h = np.random.rand()
            l = np.random.rand()
            v = np.random.rand()
            if (h<=self.duty_factor[self.psds[i]]['H1']) and ('H1' in self.detectors):
                d.append('H1')
            if (l<=self.duty_factor[self.psds[i]]['L1']) and ('L1' in self.detectors):
                d.append('L1')
            if (v<=self.duty_factor[self.psds[i]]['V1']) and ('V1' in self.detectors):
                d.append('V1')
            self.dets.append(d)

        self.data_path = pkg_resources.resource_filename('GWSim', 'data/')
        
        self.do_injections = self.do_injection_bilby
        for run in self.asds:
            for det in self.duty_factor[run]:
                if self.duty_factor[run][det]>0:
                    asdfile = self.data_path+det+'_'+run+'_strain.txt'
                    print("Reading ASD file {}".format(asdfile))
                    data = np.genfromtxt(asdfile)
                    self.asds[run][det]={}
                    self.asds[run][det]['frequency']=data[:,0]
                    self.asds[run][det]['psd']=data[:,1]**2

        self.dets = np.array(self.dets,dtype=object)
        self.psds = np.array(self.psds,dtype=object)

        print("Calculating SNR of mergers to find the detected events")


    def select(self):

        # arrays for storing the results
        self.idx_single = []
        self.snrs_single = []
        self.dets_pe = []
        self.durations = []
        self.snrs = []
        self.seeds = []

        indexes = np.arange(self.Ntot_mergers)
        wok = np.where(self.m1s > 0)[0] # check for potential problems in the mass sampling
        if len(wok)<self.pools:
            print("Warning: the number of mergers is smaller than the requested cpus. Fixing #cpus=#mergers.")
            self.pools = len(wok)
        process_indexes = np.array_split(indexes[wok],self.pools) # divide the work between subprocesses
        results = self.manager.list([[0,0,0,0]]*self.Ntot_mergers) # prepare room for the results
        can_quit = mp.Event() # create event, to terminate the children
        events_ok = mp.Event() # create event, if number of mergers above SNR is OK (user requested) or all mergers are simulated
        procs = [mp.Process(target=self.do_injections,args=(process_indexes[i],results,can_quit,events_ok)) for i in range(self.pools)]
        for p in procs: p.start()
        events_ok.wait() # tells the system to wait until the Event events_ok is set by any of the children
        can_quit.set() # if events_ok is set then set the can_quit Event

        time.sleep(1)
        print("Main thread, dealing with results...")
        # once all children have terminated: work on the results

        for res in results:
            if res[0] >= self.snr_threshold:
                idx_snr = np.where(np.sqrt(res[2])>=self.snr_single_threshold)[0]
                if len(idx_snr)>0:
                    self.snrs.append(res[0])
                    self.idx_single.append(res[1])
                    self.snrs_single.append(np.sqrt(res[2])[idx_snr])
                    self.dets_pe.append(np.array(self.dets[res[1]])[idx_snr])
                    self.durations.append(res[3])
                    self.seeds.append(res[4])

        self.idx_single = np.array(self.idx_single,dtype=int)
        
        print("{} events passed the single detector SNR threshold of {} in at least one of the detectors".format(len(self.idx_single),self.snr_single_threshold))

    def do_injection_bilby(self,n,results,can_quit,events_ok):

        print("child process pid: {}, niter: {}".format(os.getpid(),len(n)))
        logging.disable(logging.INFO)
#        mptest = True
        mptest = False
        niter = 0
        for idx_n in n:
            niter += 1
#            print("TEST counter ... {}, pid: {}, maxevts: {}".format(self.Counter.value(),os.getpid(),self.N))
            if self.Counter.value() >= self.Counter.max_value():
                events_ok.set()
            if can_quit.is_set():
                print("child: breaking 'for' loop...")
                break
            if not mptest: result_sim = self.run_bilby_sim(idx_n)
            else: result_sim = list([os.getpid(),idx_n,
                                     self.snr_single_threshold**2 * np.ones(len(self.dets[idx_n])),4]) # write dummy data
            lr = results[idx_n]
            lr = result_sim
            results[idx_n] = lr
            if self.user_request:
                if lr[0] >= self.snr_threshold:
                    self.Counter.increment() # update counter
                    #                print("Thread {}: counter updated... idx_n {}, {}, maxevts: {}".format(os.getpid(),idx_n,self.Counter.value(),self.Ntot_mergers))
            else:
                self.Counter.increment() # update counter
#                print("Thread {}: counter updated... idx_n {}, {}, maxevts: {}".format(os.getpid(),idx_n,self.Counter.value(),self.Ntot_mergers))

        # if the 'for' loop is finished before activating the Event events_ok, check again
        if self.Counter.value() >= self.Counter.max_value():
            events_ok.set()
        print("child {}: thread terminated... iter {}/{}".format(os.getpid(),niter,len(n)))
        return


    def run_bilby_sim(self,idx_n):

        detectors = self.dets[idx_n]
        if len(detectors)==0:
            return([0,0,0,0,0]) # no data for this event
        seed = int(np.random.rand(1)*100000)
        
        psd = self.psds[idx_n]
        duration = np.ceil(self.get_length(self.f_min,self.m1d[idx_n],self.m2d[idx_n]))
        if duration<1: duration=1
        psds = {}
        for det in detectors:
            psds[det] = bl.gw.detector.PowerSpectralDensity(frequency_array=self.asds[psd][det]['frequency'],
                                                            psd_array=self.asds[psd][det]['psd'])

        injection_parameters = dict(a_1=self.chi_1[idx_n],
                                    a_2=self.chi_2[idx_n],
                                    tilt_1=self.theta_1[idx_n],
                                    tilt_2=self.theta_2[idx_n],
                                    geocent_time=self.geocent_time[idx_n],
                                    mass_1=self.m1d[idx_n],
                                    mass_2=self.m2d[idx_n],
                                    luminosity_distance=self.dls[idx_n],
                                    ra=self.ras[idx_n],
                                    dec=self.decs[idx_n],
                                    theta_jn=self.incs[idx_n],
                                    psi=self.psis[idx_n],
                                    phase=self.phis[idx_n])
#        print("in bilby duration: {}, params: {}".format(duration,injection_parameters))

        waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',reference_frequency=self.f_min,minimum_frequency=self.f_min)
        waveform_generator = bl.gw.waveform_generator.WaveformGenerator(
            sampling_frequency=self.sampling_frequency, duration=duration+1.5,
            frequency_domain_source_model=bl.gw.source.lal_binary_black_hole,
            parameter_conversion=bl.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments)

        ifos_list = list(detectors)
        ifos = bl.gw.detector.InterferometerList(ifos_list)
        for j in range(len(ifos)):
            ifos[j].power_spectral_density = psds[ifos_list[j]]
            ifos[j].minimum_frequency = self.f_min
        np.random.seed(seed)
        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency, duration=duration+1.5,
            start_time=injection_parameters['geocent_time']-duration)
        
        try:
            ifos.inject_signal(waveform_generator=waveform_generator,
                               parameters=injection_parameters)
        except:
            print("Could not inject signal with injections:"+str(injection_parameters))
            return([0,0,0,0,0]) # no data for this event
        #            raise ValueError("Something went wrong with the injections: "+str(injection_parameters))

        det_SNR = 0
        SNRs = []
        for ifo_string in ifos_list:
            mfSNR = np.real(ifos.meta_data[ifo_string]['matched_filter_SNR'])**2
            SNRs.append(mfSNR)
            det_SNR += mfSNR
        SNRs = np.array(SNRs)
        det_SNR = np.sqrt(det_SNR)
        return([det_SNR,idx_n,SNRs,duration,seed])

    def get_length(self,fmin,m1,m2):

        return gwutils.calculate_time_to_merger(frequency=fmin,mass_1=m1,mass_2=m2)

   
