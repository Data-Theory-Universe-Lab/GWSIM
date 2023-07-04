import astropy as ap
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import FlatwCDM
from astropy.cosmology import Planck18
import numpy as np
import lal
import scipy.integrate as integrate
from scipy import interpolate

c = lal.C_SI/1000. #in km/s

class Cosmology(object):
    def __init__(self,cosmo_parameters,zmax=10):
        self.H0 = cosmo_parameters['H0']
        self.Omega_m = cosmo_parameters['Omega_m']
        self.w0 = cosmo_parameters['w0']
        self.cosmo = cosmo_parameters['cosmo_model']
        self.Omega_Lambda = 1-self.Omega_m
        self.zmax = zmax
        self.vfactor = 1e-9*4*np.pi*(c/self.H0)**3 # 1e-9 to have Gpc^3 per redshift bin units in dVc/dz
        self.ap_cosmo = Planck18 # default
        myz = np.linspace(0,zmax,10000)
        mydcomob_true = 0*myz
        if self.w0 == -1:
            self.ap_cosmo = FlatLambdaCDM(H0=self.H0,Om0=self.Omega_m)
        else:
            self.ap_cosmo = FlatwCDM(name='FlatwCDM',H0=self.H0,Om0=self.Omega_m,w0=self.w0)
#        self.inv_h = ap_cosmo.inv_efunc
        mydcomob_true = self.ap_cosmo.comoving_distance(myz)*self.H0/c # to get the function dcH0overc
        self.interp_dcomob = interpolate.interp1d(myz,mydcomob_true,kind='cubic')

    def dcH0overc(self,z):

        return self.interp_dcomob(z)

    def volume_z(self,z):
        # differential comoving volume dVc/dz, in Gpc^3
        return self.vfactor*(self.dcH0overc(z)**2)*self.ap_cosmo.inv_efunc(z)

    def volume_z_and_time(self,z):

        return self.volume_z(z)/(1.0+z)

    def dl_zH0(self,z):

        return self.dLH0overc(z)*c/self.H0

    def dLH0overc(self,z):

        return (1+z)*self.dcH0overc(z)

    def M_mdL(self, m, z):
        dl = self.dl_zH0(z)
        return m - 5*np.log10(dl)+25
