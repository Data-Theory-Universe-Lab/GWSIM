import numpy as np
from GWSim.utils import Interpolate_function,Integrate_1d

class Catalog(object):
    def __init__(self,Universe,mth,red_error,red_survey):
        self.Universe = Universe
        self.mth = mth
        self.red_error = red_error
        self.red_survey = red_survey
        self.z_real = -1 * np.ones(self.Universe.redshift_parameters['N'],dtype=self.Universe.precision)
        self.app_magn = np.empty(self.Universe.redshift_parameters['N'],dtype=self.Universe.precision)
        self.ra = np.empty(self.Universe.redshift_parameters['N'],dtype=self.Universe.precision)
        self.dec = np.empty(self.Universe.redshift_parameters['N'],dtype=self.Universe.precision)
        self.z = np.empty(self.Universe.redshift_parameters['N'],dtype=self.Universe.precision)
        self.abs_magn = np.empty(self.Universe.redshift_parameters['N'],dtype=self.Universe.precision)
        parameters = dict(minimum=self.Universe.redshift_parameters['minimum'],
                          maximum=self.Universe.redshift_parameters['maximum'])
        z_sub_set = np.linspace(self.Universe.redshift_parameters['minimum'],
                                self.Universe.redshift_parameters['maximum'], 1000)
        parameters['distribution'] = self.Universe.cosmology.dl_zH0(z_sub_set)
        self.interpolation = Interpolate_function(parameters)

    def create(self):

        print("Creating galaxy catalog with mth = {}".format(self.mth))
        ms = self.m_Mdl(self.Universe.abs_magn,self.Universe.z)
        idx = np.where(ms<=self.mth)[0]
        self.app_magn = ms[idx]
        self.ra = self.Universe.ra[idx]
        self.dec = self.Universe.dec[idx]
        self.z = self.Universe.z[idx]
        self.abs_magn = self.Universe.abs_magn[idx]
        self.z_real = self.Universe.z[idx]

        idx = np.where(self.z_real == -1)[0]
        self.z = np.delete(self.z, idx)
        self.app_magn = np.delete(self.app_magn, idx)
        self.ra = np.delete(self.ra, idx)
        self.dec = np.delete(self.dec, idx)
        self.abs_magn = np.delete(self.abs_magn, idx)
        self.z_real = np.delete(self.z_real, idx)

        if self.red_survey=='photoz':
            delta_z = 0.02
        elif self.red_survey=='specz':
            delta_z = 0.001
        else:
            raise ValueError("The type of galaxy survey in not implemented. Select between specz or photoz")

        if self.red_error==True:
            print("Drawing random values from normal distributions for the redshifts")
            self.sigmaz = delta_z*(1+self.z)
            self.z = np.random.normal(self.z,self.sigmaz)
        else:
            self.sigmaz = delta_z*(1+self.z)

    def m_Mdl(self,M, z):

        dl = self.interpolation(z)
        return M + self.DistanceModulus(dl) #+ Kcorr

    def DistanceModulus(self,dL):

        return 5*np.log10(dL)+25
