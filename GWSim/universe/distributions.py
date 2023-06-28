import numpy as np

class Redshift(object):
    def __init__(self,redshift_parameters,cosmology):
        self.minimum = redshift_parameters['minimum']
        self.maximum = redshift_parameters['maximum']
        self.model = redshift_parameters['redshift_model']
        self.n = redshift_parameters['n']
        self.z = np.linspace(self.minimum,self.maximum,self.n)
        self.cosmology = cosmology
        print("Setting up {} redshift distribution with zmin = {} and zmax = {}".format(self.model,self.minimum,self.maximum))

    def evaluate(self):
        if self.model=='Uniform_comoving_volume':
            return self.cosmology.volume_z(self.z)
        else:
            raise ValueError('Redshift model not yet implemented.')



class SchechterMagFunction(object):
    def __init__(self, LF_parameters, cosmo):

        self.Mstar_obs = LF_parameters['Mstar_obs']
        self.minimum = LF_parameters['minimum']
        self.maximum = LF_parameters['maximum']
        self.phistar = LF_parameters['phistar']
        self.alpha = LF_parameters['alpha']
        self.H0 = cosmo.H0
        self.n = LF_parameters['n']
        self.phistar_h0 = self.phistar*(self.H0/100)**3.
        self.minimum_h0 = self.M_Mobs(self.minimum)
        self.maximum_h0 = self.M_Mobs(self.maximum)
        self.Mstar_obs_h0 = self.M_Mobs(self.Mstar_obs)
        self.Ms = np.linspace(self.minimum_h0,self.maximum_h0,self.n)
        self.constant = 0.4*np.log(10.0)*self.phistar_h0
        print("Setting up magnitude distribution with Mmin = {}, Mmax = {}, Mstar = {}, phistar = {}, and alpha = {} for H0 = {}"
                .format(self.minimum_h0,self.maximum_h0,self.Mstar_obs_h0,self.phistar_h0,self.alpha,self.H0))

    def evaluate(self):

        dM = self.Ms-self.Mstar_obs_h0
        return self.constant * np.power(10.0, -0.4*(self.alpha+1.0)*dM) \
               * np.exp(-np.power(10, -0.4*dM))


    def M_Mobs(self,M_obs):

        return M_obs + 5.*np.log10(self.H0/100.)
