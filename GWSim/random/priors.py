from __future__ import absolute_import

import numpy as np

from scipy.interpolate import interp1d
from scipy import integrate
import copy as _copy
import sys as _sys
from . import custom_math_priors as _cmp
import pickle

def pH0(H0, prior='log'):
    """
    Returns p(H0)
    The prior probability of H0

    Parameters
    ----------
    H0 : float or array_like
        Hubble constant value(s) in kms-1Mpc-1
    prior : str, optional
        The choice of prior (default='log')
        if 'log' uses uniform in log prior
        if 'uniform' uses uniform prior

    Returns
    -------
    float or array_like
        p(H0)
    """
    if prior == 'uniform':
        return np.ones(len(H0))
    if prior == 'log':
        return 1./H0

class distance_distribution(object):
    def __init__(self, name):
        self.name = name

        if self.name == 'powerlaw':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        if self.name == 'BNS':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'NSBH':
            dist = PriorDict(conversion_function=constrain_m1m2)
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=1000)

        if self.name == 'BBH-constant':
            dist = PriorDict()
            dist['luminosity_distance'] = PowerLaw(alpha=2, minimum=1, maximum=15000)

        self.dist = dist

    def sample(self, N_samples):
        samples = self.dist.sample(N_samples)
        return samples['luminosity_distance']

    def prob(self, samples):
        return self.dist['luminosity_distance'].prob(samples)


class mass_prior(object):
    """
    Wrapper for for managing the priors on source frame masses.
    The prior is factorized as :math:`p(m_1,m_2) \\propto p(m_1)p(m_2|m_1)`

    Parameters
    ----------
    name: str
        Name of the model that you want. Available 'BBH-powerlaw', 'BBH-powerlaw-gaussian'
        'BBH-broken-powerlaw', 'NSBH-powerlaw', 'NSBH-powerlaw-gaussian', 'NSBH-broken-powerlaw', 'BNS'.
    hyper_params_dict: dict
        Dictionary of hyperparameters for the prior model you want to use. See code lines for more details
    """

    def __init__(self, name, hyper_params_dict):
        
        self.name = name
        self.hyper_params_dict=_copy.deepcopy(hyper_params_dict)
        dist = {}
        
        if self.name == 'powerlaw' or self.name == 'NSBH-powerlaw':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']
            if self.name == 'powerlaw':
                dist={'mass_1':_cmp.PowerLaw_math(alpha=-alpha,min_pl=mmin,max_pl=mmax),
                     'mass_2':_cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=mmax)}
            else:
                dist={'mass_1':_cmp.PowerLaw_math(alpha=-alpha,min_pl=mmin,max_pl=mmax),
                     'mass_2':_cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)}

            self.mmin = mmin
            self.mmax = mmax
            
        elif self.name == 'powerlaw-double-gaussian' or self.name == 'NSBH-powerlaw-double-gaussian':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']
            
            mu_g = hyper_params_dict['mu_g']
            sigma_g = hyper_params_dict['sigma_g']
            lambda_peak = hyper_params_dict['lambda_peak']
            
            mu_g_low = hyper_params_dict['mu_g_low']
            sigma_g_low = hyper_params_dict['sigma_g_low']
            lambda_peak_low = hyper_params_dict['lambda_peak_low']
            
            delta_m = hyper_params_dict['delta_m']
            
            if self.name == 'powerlaw-double-gaussian':
                
                m1pr = _cmp.PowerLawDoubleGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_peak, lambda_g_low=lambda_peak_low
                                                        ,mean_g_low=mu_g_low,sigma_g_low=sigma_g_low,mean_g_high=mu_g,sigma_g_high=sigma_g
                                                        , min_g=mmin,max_g=mu_g+5*np.max([sigma_g_low,sigma_g]))
                m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=np.max([mu_g+5*sigma_g,mmax]))
                
                # Smooth the lower end of these distributions
                dist = self.CheckSmoothing(mmin,mmax,delta_m,m1pr,m2pr)
#                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
#                      'mass_2':_cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}

            else:
                m1pr = _cmp.PowerLawDoubleGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_peak, lambda_g_low=lambda_peak_low
                        ,mean_g_low=mu_g_low,sigma_g_low=sigma_g_low,mean_g_high=mu_g,sigma_g_high=sigma_g
                        , min_g=mmin,max_g=mu_g+5*np.max([sigma_g_low,sigma_g]))
                m2pr = _cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)
                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                      'mass_2':m2pr}

            # TODO Add a check on the mu_g - 5 sigma of the gaussian to not overlap with mmin, print a warning
            if (mu_g_low - 5*sigma_g_low)<=mmin:
                print('Warning, your mean (minuse 5 sigma) of the gaussian component is too close to the minimum mass')

            self.mmin = mmin
            self.mmax = dist['mass_1'].maximum

        elif self.name == 'powerlaw-gaussian' or self.name == 'NSBH-powerlaw-gaussian':
            alpha = hyper_params_dict['alpha']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']

            mu_g = hyper_params_dict['mu_g']
            sigma_g = hyper_params_dict['sigma_g']
            lambda_peak = hyper_params_dict['lambda_peak']

            delta_m = hyper_params_dict['delta_m']

            if self.name == 'powerlaw-gaussian':
                m1pr = _cmp.PowerLawGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_peak
                    ,mean_g=mu_g,sigma_g=sigma_g,min_g=mmin,max_g=mu_g+5*sigma_g)

                # The max of the secondary mass is adapted to the primary mass maximum which is desided byt the Gaussian and PL
                m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=np.max([mu_g+5*sigma_g,mmax]))
                dist = self.CheckSmoothing(mmin,mmax,delta_m,m1pr,m2pr)
                

            else:
                m1pr = _cmp.PowerLawGaussian_math(alpha=-alpha,min_pl=mmin,max_pl=mmax,lambda_g=lambda_peak
                    ,mean_g=mu_g,sigma_g=sigma_g,min_g=mmin,max_g=mu_g+5*sigma_g)
                m2pr = _cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)
                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                      'mass_2':m2pr}

            # TODO Add a check on the mu_g - 5 sigma of the gaussian to not overlap with mmin, print a warning
            #            if (mu_g - 5*sigma_g)<=mmin:
            #                print('Warning, your mean (minuse 5 sigma) of the gaussian component is too close to the minimum mass')

            self.mmin = mmin
            self.mmax = dist['mass_1'].maximum

        elif self.name == 'broken-powerlaw' or self.name == 'NSBH-broken-powerlaw':
            alpha_1 = hyper_params_dict['alpha']
            alpha_2 = hyper_params_dict['alpha_2']
            beta = hyper_params_dict['beta']
            mmin = hyper_params_dict['mmin']
            mmax = hyper_params_dict['mmax']
            b =  hyper_params_dict['b']

            delta_m = hyper_params_dict['delta_m']

            if self.name == 'broken-powerlaw':
                m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-alpha_1,alpha_2=-alpha_2,min_pl=mmin,max_pl=mmax,b=b)
                m2pr = _cmp.PowerLaw_math(alpha=beta,min_pl=mmin,max_pl=mmax)
                dist = self.CheckSmoothing(mmin,mmax,delta_m,m1pr,m2pr)
                #                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                #                      'mass_2':_cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}
            else:
                m1pr = _cmp.BrokenPowerLaw_math(alpha_1=-alpha_1,alpha_2=-alpha_2,min_pl=mmin,max_pl=mmax,b=b)
                m2pr = _cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)

                dist={'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                      'mass_2':m2pr}

            self.mmin = mmin
            self.mmax = mmax

        elif self.name == 'BNS':

            dist={'mass_1':_cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0),
                  'mass_2':_cmp.PowerLaw_math(alpha=0.0,min_pl=1.0,max_pl=3.0)}

            self.mmin = 1.0
            self.mmax = 3.0

        else:
            print('Name not known, aborting')
            _sys.exit()

        self.dist = dist

        if self.name.startswith('NSBH'):
            self.mmin=1.0

    def CheckSmoothing(self,mmin,mmax,delta_m,m1pr,m2pr):
        """
        This function checks the values of the smoothing function in the mass interval
        of interest. If the max is nan or 0, we skip the smoothing and return a 
        dictionnary of m1, m2 pdf without smooting
        """
        xmass = np.linspace(mmin,mmax,100)
        sfactor = _cmp._S_factor(xmass,mmin,delta_m)
        if not np.isnan(np.max(sfactor)) and (np.max(sfactor)>0):
            dist = {'mass_1': _cmp.SmoothedProb(origin_prob=m1pr,bottom=mmin,bottom_smooth=delta_m),
                    'mass_2': _cmp.SmoothedProb(origin_prob=m2pr,bottom=mmin,bottom_smooth=delta_m)}
        else:
            dist = {'mass_1': m1pr,
                    'mass_2': m2pr}
        return dist
        
            
    def joint_prob(self, ms1, ms2):
        """
        This method returns the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        ms1: np.array(matrix)
            mass one in solar masses
        ms2: dict
            mass two in solar masses
        """

        if self.name == 'BNS':
            to_ret =self.dist['mass_1'].prob(ms1)*self.dist['mass_2'].prob(ms2)
        else:
            to_ret =self.dist['mass_1'].prob(ms1)*self.dist['mass_2'].conditioned_prob(ms2,self.mmin*np.ones_like(ms1),ms1)

        return to_ret

    def sample(self, Nsample, mass_values=None):
        """
        This method samples from the joint probability :math:`p(m_1,m_2)`

        Parameters
        ----------
        Nsample: int
            Number of samples you want
        input: vector of solar masses
        return: m1_samples, m2_sample, m1dist, m2dist
        m1dist and m2dist are the pdf values at points mass_values
        """

        vals_m1 = np.random.rand(Nsample)
        vals_m2 = np.random.rand(Nsample)

        m1_trials = np.logspace(np.log10(self.dist['mass_1'].minimum),np.log10(self.dist['mass_1'].maximum),10000)
        m2_trials = np.logspace(np.log10(self.dist['mass_2'].minimum),np.log10(self.dist['mass_2'].maximum),10000)

        cdf_m1_trials = self.dist['mass_1'].cdf(m1_trials)
        cdf_m2_trials = self.dist['mass_2'].cdf(m2_trials)

        m1_trials = np.log10(m1_trials)
        m2_trials = np.log10(m2_trials)

        _,indxm1 = np.unique(cdf_m1_trials,return_index=True)
        _,indxm2 = np.unique(cdf_m2_trials,return_index=True)

        interpo_icdf_m1 = interp1d(cdf_m1_trials[indxm1],m1_trials[indxm1],bounds_error=False,fill_value=(m1_trials[0],m1_trials[-1]))
        interpo_icdf_m2 = interp1d(cdf_m2_trials[indxm2],m2_trials[indxm2],bounds_error=False,fill_value=(m2_trials[0],m2_trials[-1]))

        mass_1_samples = 10**interpo_icdf_m1(vals_m1)

        if self.name == 'BNS':
            mass_2_samples = 10**interpo_icdf_m2(vals_m2)
            indx = np.where(mass_2_samples>mass_1_samples)[0]
            mass_1_samples[indx],mass_2_samples[indx] = mass_2_samples[indx],mass_1_samples[indx]
        else:
            mass_2_samples = 10**interpo_icdf_m2(vals_m2*self.dist['mass_2'].cdf(mass_1_samples))

        m1d = 0
        m2d = 0
        if mass_values is not None:
            m1d = self.dist['mass_1'].prob(mass_values) # easy for m1
            m2d = 0*m1d
            for m1 in mass_1_samples: # have to adapt to each m1 value...
                wm1 = np.where((mass_values >= self.dist['mass_2'].minimum) & (mass_values <= m1))[0]
                if len(wm1)>1: # at least 2 values of the mass_values are concerned
                    m2 = self.dist['mass_2'].prob(mass_values[wm1])
                    integ = integrate.trapz(m2,mass_values[wm1])
                    if integ > 0: m2d[wm1] += m2/integ  # renormalize to 1 in [mmin;m1]
                    else:
                        print("integral == 0! nm2v: {}, m1 = {}, mv = {}, integ = {}, pm2 = {}, min_m2_pl: {}, max_m2_pl: {}, beta: {}"
                                .format(len(wm1),m1,mass_values[wm1],integ,m2,self.dist['mass_2'].minimum,self.dist['mass_2'].maximum,self.hyper_params_dict['beta']))
                        file = open( "debug.p", "wb" )
                        pickle.dump([self.dist['mass_1'],self.dist['mass_2'],mass_values,m1], file)
                else: # m1 is so close to mmin that the mass_values has 0 or 1 point in [mmin;m1]. I set the pdf to 1 in the closest point of mass_values
                    dm = np.abs(mass_values-self.dist['mass_2'].minimum)
                    wm1 = np.where(dm == np.min(dm))[0]
                    m2d[wm1] = 1
        return mass_1_samples, mass_2_samples, m1d, m2d
