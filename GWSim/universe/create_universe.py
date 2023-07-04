import os
import numpy as np
from optparse import Option, OptionParser, OptionGroup
from scipy.interpolate import interp1d
from GWSim.utils import Rejection_Sampling, Inverse_Cumulative_Sampling, Interpolate_function
from GWSim.universe.distributions import Redshift,SchechterMagFunction

class Universe(object):
    def __init__(self,redshift_parameters,cosmo,LF_parameters,precision,cosmo_parameters,sampling_method):
        self.redshift_parameters = redshift_parameters
        self.cosmology = cosmo
        self.LF_parameters = LF_parameters
        self.N = self.redshift_parameters['N']
        self.precision = precision
        self.cosmo_parameters = cosmo_parameters
        self.sampling_method = sampling_method
        
            
    def create(self):
        print("Creating universe with {} galaxies.".format(self.N))
        self.z = self.sample_redshifts()
        self.z = np.array(self.z, dtype=self.precision)
        self.abs_magn = self.sample_LF()
        self.abs_magn = np.array(self.abs_magn, dtype=self.precision)
        self.ra = np.random.rand(self.N)*2.0*np.pi
        self.ra = np.array(self.ra, dtype=self.precision)
        self.dec = np.arcsin(2.0*np.random.rand(self.N) - 1.0)
        self.dec = np.array(self.dec,dtype=self.precision)

    def sample_redshifts(self):
        print("Sampling the redshifts of galaxies")
        redshift_distribution = Redshift(self.redshift_parameters,self.cosmology)        
        self.redshift_parameters['distribution'] = redshift_distribution.evaluate()
        self.redshift_parameters['interpolation'] = Interpolate_function(self.redshift_parameters)
        if self.sampling_method=='rejection':             
            Sampling = Rejection_Sampling(self.redshift_parameters)
            redshifts = Sampling.Sample()
        else:
            Sampling = Inverse_Cumulative_Sampling(self.redshift_parameters)
            redshifts = Sampling.Sample()
               
        return redshifts 

    def sample_LF(self):
        print("Sampling the magnitudes of galaxies")
        magnitude_distribution = SchechterMagFunction(self.LF_parameters,self.cosmology)
        self.LF_parameters['distribution'] = magnitude_distribution.evaluate()
        self.LF_parameters['interpolation'] = Interpolate_function(self.LF_parameters)
        if self.sampling_method=='rejection':            
            Sampling = Rejection_Sampling(self.LF_parameters)
            magnitudes = Sampling.Sample()
        else:
            Sampling = Inverse_Cumulative_Sampling(self.LF_parameters)
            magnitudes = Sampling.Sample()

        return magnitudes
