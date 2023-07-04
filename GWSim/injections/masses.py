import os
import numpy as np
from optparse import Option, OptionParser, OptionGroup
from scipy.interpolate import interp1d
from GWSim.utils import Rejection_Sampling, Interpolate_function
from GWSim.injections.distributions import Mass_redshift
from GWSim.injections.time_delay import Time_delay
from GWSim.random.priors import mass_prior
import copy
from scipy import integrate


class Masses(object):
    def __init__(self,pop_parameters,time_delay,red_depend):

        self.pop_parameters = pop_parameters
        self.time_delay = time_delay
        self.red_depend = red_depend
        self.mass_values = np.linspace(self.pop_parameters['mmin'],self.pop_parameters['mmax'],1000)
        self.pop_parameters['masses'] = self.mass_values

        if (self.pop_parameters['mu_g']<self.pop_parameters['mu_g_low']) and (self.pop_parameters=='powerlaw-double-gaussian'):

            print('Mu_low is higher than mu, inverting the values')
            x = copy.deepcopy(self.pop_parameters['mu_g'])
            self.pop_parameters['mu_g'] = copy.deepcopy(self.pop_parameters['mu_g_low'])
            self.pop_parameters['mu_g_low'] = x


        self.hyper_params_dict = copy.deepcopy(self.pop_parameters) # same keys for the dictionaries
        # check 'b' field for gwcosmo
        # gwsim asks b in solar masses, gwcosmo wants fraction of interval
        # but do not modify the pop_parameters b value!
        self.hyper_params_dict['b'] = (pop_parameters['b']-pop_parameters['mmin'])/(pop_parameters['mmax']-pop_parameters['mmin'])

    def sample(self,z,cosmo,pools):
        if self.red_depend == True:
            if self.time_delay == True:
                print('Sampling {} masses taking into account time delay.'.format(self.pop_parameters['N']))
                td = Time_delay(self.hyper_params_dict,z,cosmo,pools)
                self.pop_parameters['m1s'], self.pop_parameters['m2s'], self.pop_parameters['m1s_eff'], self.pop_parameters['m2s_eff'] = td.sample(self.mass_values)
            else:
              print('Sampling {} masses taking into account redshift dependance.'.format(self.pop_parameters['N']))
              mass_distribution = Mass_redshift(self.hyper_params_dict,z,pools)
              self.pop_parameters['m1s'], self.pop_parameters['m2s'] = mass_distribution.sample(self.mass_values)
              self.pop_parameters['m1s_eff'] = mass_distribution.m1_eff
              self.pop_parameters['m2s_eff'] = mass_distribution.m2_eff

        else:
            print("Sampling {} values of mass 1, mass 2".format(self.pop_parameters['N']))
            # don't need the Mass class
            mp = mass_prior(self.pop_parameters['model'],self.hyper_params_dict)
            self.pop_parameters['m1s'], self.pop_parameters['m2s'], self.pop_parameters['m1s_eff'], self.pop_parameters['m2s_eff'] = mp.sample(self.pop_parameters['N'],self.mass_values)

        # normalize the m1s_eff and m2s_eff distributions
        norm = integrate.simpson(self.pop_parameters['m1s_eff'], self.mass_values)
        if norm > 0:
            self.pop_parameters['m1s_eff'] /= norm
        else:
            print("Anomaly: m1-pdf integral is {}".format(norm))

        norm = integrate.simpson(self.pop_parameters['m2s_eff'],self.mass_values)
        if norm > 0:
            self.pop_parameters['m2s_eff'] /= norm
        else:
            print(self.pop_parameters['m2s_eff'])
            print("Anomaly: m2-pdf integral is {}".format(norm))
