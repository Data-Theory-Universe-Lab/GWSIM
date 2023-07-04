import numpy as np
from GWSim.utils import Interpolate_function, Integrate_1d
from scipy.interpolate import interp1d
from scipy.integrate import simps
from scipy.special import erf
import numpy.random as rand
from astropy import units as u
import multiprocess as mp
import os
import time
import GWSim.random.custom_math_priors as cmp
import copy
from GWSim.random.priors import mass_prior

def CheckEvolutionParams(zdict):

    isok = True
    if (zdict['lambda_peak'] < 0) or (zdict['lambda_peak'] > 1):
        isok = False
        print("invalid lambda: {}".format(zdict['lambda_peak']))

    return isok

class Mass_redshift(object):

    def __init__(self,pop_parameters,z,pools=1):

        self.pools = pools
        self.manager = mp.Manager()
        self.z = z
        self.z_max = np.max(self.z)
        # the pop_parameters dict must contains the same fields than the gwcosmo dict for mass sampling
        # the values correspond to z=0
        self.dict_z0 = copy.deepcopy(pop_parameters)
        # store the mass distributions
        self.m1_eff = 0
        self.m2_eff = 0


    def z_dict(self,z): # modify values with redshift evolution

        dict_z = copy.deepcopy(self.dict_z0)
        dict_z['alpha'] += dict_z['alpha']*z
        dict_z['alpha_2'] += dict_z['alpha_2']*z
        dict_z['mmin'] += dict_z['epsilon_Mmin']*z
        dict_z['mmax'] += dict_z['epsilon_Mmax']*z
        dict_z['beta'] += dict_z['epsilon_beta']*z
        dict_z['sigma_g'] += dict_z['epsilon_sigma']*z
        dict_z['lambda_peak'] += dict_z['epsilon_Lambda']*z
        dict_z['mu_g'] += dict_z['epsilon_mu']*z
        dict_z['sigma_g_low'] += dict_z['epsilon_sigma_low']*z
        dict_z['lambda_peak_low'] += dict_z['epsilon_Lambda_low']*z
        dict_z['mu_g_low'] += dict_z['epsilon_mu_low']*z
        dict_z['delta_m'] += dict_z['epsilon_delta_m']*z
        dict_z['b'] += dict_z['epsilon_b']*z
        return dict_z

    def sample(self,mass_values):
        '''
        mass_values is a vector of masses, regularly spaced for instance, where we want to compute the effective mass distribution
        '''
        self.mass_values = mass_values
        m1 = np.zeros(len(self.z))
        m2 = np.zeros(len(self.z))

        indexes = np.arange(len(self.z))
        if len(self.z)<self.pools:
            print("Warning: the number of mergers is smaller than the requested cpus. Fixing #cpus=#mergers.")
            self.pools = len(self.z)
        process_indexes = np.array_split(indexes,self.pools)
        results = self.manager.list([[0,0]]*len(self.z))
        m1ds = self.manager.list([np.zeros(len(self.mass_values),dtype=float)]*self.pools)
        m2ds = self.manager.list([np.zeros(len(self.mass_values),dtype=float)]*self.pools)
        procs = [mp.Process(target=self.sampling,args=(i,process_indexes[i],results,m1ds,m2ds)) for i in range(self.pools)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        time.sleep(1)
        print("Back to Mass_redshit main thread, gathering results...")
        for i in range(len(self.z)):
            m1[i] = results[i][0]
            m2[i] = results[i][1]
        self.m1_eff = np.zeros(len(self.mass_values),dtype=float)
        self.m2_eff = np.zeros(len(self.mass_values),dtype=float)
        for i in range(self.pools):
            self.m1_eff += m1ds[i]
            self.m2_eff += m2ds[i]

        return m1,m2

    def sampling(self,i,n,results,m1dists,m2dists):

        print("child process pid: {}, thread_idx: {}, niter: {}".format(os.getpid(),i,len(n)))

        for idx_n in n:
            m1, m2, m1s_dist, m2s_dist = self.masses(idx_n)
            lr = results[idx_n] # get address
            lr = [m1,m2] # update data
            results[idx_n] = lr  # write data

            lr = m1dists[i] # get address
            lr += m1s_dist # update data
            m1dists[i] = lr # write data
            lr = m2dists[i]
            lr += m2s_dist
            m2dists[i] = lr

        return

    def masses(self,idx_n):

        z = self.z[idx_n]
        new_dict = self.z_dict(z)
        if not CheckEvolutionParams(new_dict):
            print("Inconsistencies detected in the z-evolution parameters in the z-range [{};{}]. Returning masses [-1,-1].".format(np.min(self.z),self.z_max))
            return -1,-1,0,0 # keep last 2 values set to 0 as it is the m1 and m2 pdfs

        mp = mass_prior(new_dict['model'],new_dict)
        m1, m2, m1s_dist, m2s_dist = mp.sample(1,self.mass_values)
        return m1, m2, m1s_dist, m2s_dist


class Spin(object):

    def __init__(self,parameters,hosts_redshift):

        self.parameters = parameters
        self.spin_model = parameters['spin_model']

        if self.spin_model.casefold() == 'None'.casefold():
            self.chi = self.zeros
        elif self.spin_model.casefold() == 'Uniform'.casefold():
            self.chi = self.uniform
        elif self.spin_model.casefold() == 'Gaussian'.casefold():
            self.chi = self.gaussian
        elif self.spin_model.casefold() == 'zGaussian'.casefold():
            self.chi = self.zgaussian
        elif self.spin_model.casefold() == 'Heavy_mass'.casefold():
            self.chi = self.heavy_mass
        elif self.spin_model.casefold() == 'correlated'.casefold():
            self.chi = self.correlated
        else:
            raise ValueError("Spin model not understood. Availables are (case insensitive): None, Uniform, Gaussian, zGaussian, Heavy_mass, correlated.")

        self.qratios = self.parameters['m2s']/self.parameters['m1s']
        self.z = hosts_redshift


    def chi_eff_to_chis(self,N):

        if self.parameters['aligned_spins'] == True: # then (theta1=theta2=0) OR (theta1=theta2=pi)
            t1 = 0*self.chi_eff
            t2 = 0*self.chi_eff
            wm = np.where(self.chi_eff<0)[0]
            if len(wm) != 0:
                t1[wm] = np.pi+0*self.chi_eff[wm]
                t2[wm] = np.pi+0*self.chi_eff[wm]
            qchi = (1+self.qratios)*np.abs(self.chi_eff) # is in [0;1+q], use the absolute value, the sign will be from the cos\theta_i
            chi2 = [np.random.uniform(np.max([0,(qchi[i]-1)/self.qratios[i]]),np.min([1,(1+qchi[i])/self.qratios[i]]),1)[0] for i in range(N)] # is in [0;1]
            chi1 = qchi-self.qratios*chi2 # is in [0;1]
        else: # theta_1 and theta_2 are not 0 or pi
            qchi = (1+self.qratios)*self.chi_eff
            chi2_ct2 = [np.random.uniform(np.max([-1,(qchi[i]-1)/self.qratios[i]]),np.min([1,(1+qchi[i])/self.qratios[i]]),1)[0] for i in range(N)] # is in [-1;1]
            chi1_ct1 = qchi-self.qratios*chi2_ct2 # is in [-1;1]
            # then split the product:
            chi2 = np.abs(chi2_ct2)**np.random.uniform(0,1,N) # is in [0;1]
            ct2 = chi2_ct2/chi2 # is in [-1;1]
            chi1 = np.abs(chi1_ct1)**np.random.uniform(0,1,N) # is in [0;1]
            ct1 = chi1_ct1/chi1 # is in [-1;1]
            t1 = np.arccos(ct1)
            t2 = np.arccos(ct2)

        return np.array(chi1), np.array(chi2), np.array(t1), np.array(t2)

    def chi_eff_from_chis(self,chi1,chi2):
        t1 = 0
        t2 = 0
        N = len(chi1)
        if self.parameters['aligned_spins'] == True: # then (theta1=theta2=0) OR (theta1=theta2=pi)
            ct1 = 2*(np.random.uniform(0,1,N)<0.5)-1 # cos is -1 or 1
            ct2 = ct1
        else:
            ct1 = np.random.uniform(-1,1,N)
            ct2 = np.random.uniform(-1,1,N)

        chi_eff = (chi1*ct1+self.qratios*chi2*ct2)/(1+self.qratios)

        return chi_eff, np.arccos(t1), np.arccos(t2)

    def zeros(self,N):
        self.chi_eff = np.zeros(N)
        return np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)

    def uniform(self,N):
        self.chi_eff = np.random.uniform(-1,1,N)
        return self.chi_eff_to_chis(N)

    def gaussian(self,N):
        self.chi_eff =  np.random.normal(self.parameters['chi_0'], self.parameters['sigma_0'], N)
        return self.chi_eff_to_chis(N)

    def GetProba(self,mu,sigma,a):
        # returns the proba that the random gaussian value is outside of [-a;a]
        fact = 1/(sigma*np.sqrt(2))
        return 1-0.5*(erf((a-mu)*fact)-erf(-(a+mu)*fact))

    def CheckValues(self,mu_z,sigma_z):
        values_are_ok = True
        ww = np.where(sigma_z<0)[0]
        if len(ww)!=0:
            print("Spin-z evol: Some values of the stdev are <0: canceling computation, switching to random values.")
            values_are_ok = False
            return values_are_ok
        probas = self.GetProba(mu_z,sigma_z,1)
        ww = np.where(probas>1-1e-4)[0]
        if len(ww)!=0:
            print("Spin z-evol: Some values of mu/sigma are too far from physical values, switching to random values.")
            values_are_ok = False
            return values_are_ok
        return values_are_ok
    
    def get_zGaussian_params(self):
        user_defined = False
        values_are_ok = True
        if not (self.parameters['spin_zevol_mu'] is None): # all 4 params must be user-defined (this is checked for in GW_injections)
            user_defined = True
            mu_z = self.parameters['spin_zevol_mu'] + self.parameters['spin_zevol_epsilon_mu'] * self.z
            sigma_z = self.parameters['spin_zevol_sigma'] + self.parameters['spin_zevol_epsilon_sigma'] * self.z            
            # test user defined values: sig<0, P(|X|>1)...
            values_are_ok = self.CheckValues(mu_z,sigma_z)
            if not values_are_ok: user_defined = False
                
        if not user_defined or (not values_are_ok):
            # draw all values
            eps_mu_min = -1
            eps_mu_max = 1
            zmax = max(self.z)
            self.parameters['spin_zevol_epsilon_mu'] = np.random.uniform(eps_mu_min,eps_mu_max,1)[0]
            self.parameters['spin_zevol_mu'] = np.random.uniform(max([0,-self.parameters['spin_zevol_epsilon_mu']*zmax]),
                                                                 min([1,1-self.parameters['spin_zevol_epsilon_mu']*zmax]),1)[0]
            eps_sig_min = -1
            eps_sig_max = 1
            sig_max = 1 # quite large spread
            self.parameters['spin_zevol_epsilon_sigma'] = np.random.uniform(eps_sig_min,eps_sig_max,1)[0]
            minval = max([0,-self.parameters['spin_zevol_epsilon_sigma']*zmax])
            self.parameters['spin_zevol_sigma'] = np.random.uniform(minval,minval+sig_max,1)[0]

        # then compute expectation values and stdev with linear variation
        mu_z = self.parameters['spin_zevol_mu'] + self.parameters['spin_zevol_epsilon_mu'] * self.z
        sigma_z = self.parameters['spin_zevol_sigma'] + self.parameters['spin_zevol_epsilon_sigma'] * self.z
        return user_defined, mu_z, sigma_z
    
    def zgaussian(self,N):
        user_defined, mu_z, sigma_z = self.get_zGaussian_params()
        print("Spin-z dependence, user-defined: {}. parameters: mu0: {}, eps_mu: {}, sig0: {}, eps_sig0: {}".format(user_defined,self.parameters['spin_zevol_mu'],
                                                                                                                    self.parameters['spin_zevol_epsilon_mu'],
                                                                                                                    self.parameters['spin_zevol_sigma'],
                                                                                                                    self.parameters['spin_zevol_epsilon_sigma']))
        
        chis = np.array([np.abs(np.random.normal(mu_z[i],sigma_z[i],2)) for i in range(N)])
        # the sign will be given by cos(theta_i)
        ntries_max = 100000 # very very very conservative
        ntries = np.zeros(2)
        for idx in range(2):
            check = True
            while check: # check that chi1 is in [-1;1], can still happen but with low proba
                ww = np.where(chis[:,idx] > 1)[0]
                if len(ww) == 0:
                    break
                else:
                    chis[ww,idx] = np.abs(np.random.normal(mu_z[ww],sigma_z[ww],len(ww))) # redraw values where abs > 1
                    ntries[idx] += 1
                    if ntries[idx]>ntries_max:
                        print("Spin-z-evol: {}, mu_z: {}, sigma_z: {}".format(len(ww),mu_z[ww],sigma_z[ww]))
                        raise ValueError("Something went wrong with the chi_[1,2] simulation in the 'spin-z-evol' case. Exiting.")
        print("Correct (physical) Spin z-evol values for chi_1 and chi_2 obtained after {} and {} iterations, respectively".format(ntries[0],ntries[1]))
        if self.parameters['aligned_spins'] == True: # then (theta1=theta2=0) OR (theta1=theta2=pi)
            ct1 = 2*(np.random.uniform(0,1,N)<0.5)-1 # cos is -1 or 1
            ct2 = ct1
        else:
            ct1 = np.random.uniform(-1,1,N)
            ct2 = np.random.uniform(-1,1,N)
        t1 = np.arccos(ct1)
        t2 = np.arccos(ct2)
        self.chi_eff = (chis[:,0]*ct1+self.qratios*chis[:,1]*ct2)/(1+self.qratios)
        return chis[:,0], chis[:,1], t1, t2

    def heavy_mass(self,N):

        chi1 = np.random.uniform(0,1,N)
        chi2 = np.random.uniform(0,1,N)

        m1 = self.parameters['m1s']
        m2 = self.parameters['m2s']
        wm = np.where(m1<=self.parameters['m_th'])[0]
        if len(wm) != 0:
            chi1[wm] = 0
        wm = np.where(m2<=self.parameters['m_th'])[0]
        if len(wm) != 0:
            chi2[wm] = 0
        self.chi_eff, t1, t2 = self.chi_eff_from_chis(chi1,chi2)

        return chi1, chi2, t1, t2

    def correlated(self,N):

        user_defined = False
        if not (self.parameters['spin_correlation_mu_chi0'] is None):
            mu_chi0 = self.parameters['spin_correlation_mu_chi0']
            sigma_chi0 = self.parameters['spin_correlation_sigma_chi0']
            log10_sigma_chi0 = np.log10(sigma_chi0)
            alpha = self.parameters['spin_correlation_alpha']
            beta = self.parameters['spin_correlation_beta']
            user_defined = True

        if not user_defined:
            print("Drawing random values for spin correlation model...")
            alpha_min = -1 # could be added in the cmdline opts
            alpha_max = 1 # could be added in the cmdline opts
            alpha = np.random.uniform(alpha_min,alpha_max,1)[0]
            beta_min = -2 # could be added in the cmdline opts
            beta_max = 2 # could be added in the cmdline opts
            beta = np.random.uniform(beta_min,beta_max,1)[0]
            mu_chi0_min = -1+abs(alpha)/2
            mu_chi0_max = 1-abs(alpha)/2
            mu_chi0 = np.random.uniform(mu_chi0_min,mu_chi0_max,1)[0] # is in [-1+|alpha|/2;1-|alpha|/2]
            log10_sigma_chi0_min = -2 # could be added in the cmdline opts
            log10_sigma_chi0_max = 0 # could be added in the cmdline opts
            log10_sigma_chi0 = np.random.uniform(log10_sigma_chi0_min,log10_sigma_chi0_max,1)[0]
            sigma_chi0 = 10**log10_sigma_chi0
        else:
            print("Using user-defined values for spin correlation model...")
            if mu_chi0 <= -1+abs(alpha)/2 or mu_chi0 >= 1-abs(alpha)/2:
                print("Warning: user-defined values for spin correlation are strange, non-physical value of Xeff can be obtained.")

        print("spin secret correlation parameters: alpha: {}, beta: {}, mu_chi0: {}, sigma_chi0: {}"
              .format(alpha,beta,mu_chi0,sigma_chi0))
        # record value for the user
        self.parameters['spin_correlation_mu_chi0'] = mu_chi0
        self.parameters['spin_correlation_sigma_chi0'] = sigma_chi0
        self.parameters['spin_correlation_alpha'] = alpha
        self.parameters['spin_correlation_beta'] = beta

        #qratios = self.parameters['m2s']/self.parameters['m1s']
        mu_chi = mu_chi0+alpha*(self.qratios-0.5) # add correlation for expectation value
        log10_sigma_chi = log10_sigma_chi0+beta*(self.qratios-0.5) # add correlation for standard deviation
        sigma_chi = 10**log10_sigma_chi
        ww = np.where(abs(mu_chi)>1+3*sigma_chi)[0] # eventually detect cases where it will be hard to get |chi_eff| <= 1
        if len(ww) != 0: # should never arrive if mu_chi0 is in [-1+|alpha|/2;1-|alpha|/2]
            # force these |mu_chi| to 1, do not touch the sigma
            print("Force {} expectation values ({}%) to 1 in mu_chi".format(len(ww),100*len(ww)/N))
            mu_chi[ww] = np.sign(mu_chi[ww])*1

        # draw chi_eff:
        self.chi_eff = np.random.normal(mu_chi,sigma_chi) # following Callister arxiv:2106.00521
        check = True
        ntries = 0
        ntries_max = 100000 # very very very conservative
        while check: # check that chi_eff is in [-1;1]
            ww = np.where(abs(self.chi_eff) > 1)[0]
            if len(ww) == 0:
                break
            else:
                self.chi_eff[ww] = np.random.normal(mu_chi[ww],sigma_chi[ww],len(ww)) # redraw values where abs > 1
                ntries += 1
                if ntries>ntries_max:
                    print("{}, mu_chi: {}, sigma_chi: {}".format(len(ww),mu_chi[ww],sigma_chi[ww]))
                    raise ValueError("Something went wrong with the chi_eff simulation in the 'correlated' case. Exiting.")

        return self.chi_eff_to_chis(N)

    def sample(self,N):

        return self.chi(N)


class Redshift(object):

    def __init__(self,parameters,cosmo,redshift_weighting):

        self.parameters = parameters
        self.cosmo = cosmo
        if redshift_weighting:
            if self.parameters['model']=='Madau':
                print("Setting up Madau evolution model with zp = {}, alpha = {}, R0 = {} and beta = {}"
                      .format(self.parameters['zp'],self.parameters['madau_alpha'],self.parameters['R0'],self.parameters['madau_beta']))
                self.evolution = self.Madau_evolution
            elif self.parameters['model']=='time_delay':
                self.parameters['madau_alpha'] = 2.7
                self.parameters['madau_beta'] = 2.9
                self.parameters['madau_zp'] = 1.9
                print("Setting up time delay model with d = {}, alpha = {}, t_min = {} and gamma = {}"
                      .format(self.parameters['d'],self.parameters['alpha_delay'],self.parameters['t_min'],self.parameters['gamma_delay']))
                self.evolution = self.time_delay
            
                self.time_delay_interp = self.time_delay_interpolation()
            else:
                raise ValueError("Redshift evolution model not yet implemented.")
        else:
            print("Merger rate independent on z.")
            self.evolution = self.no_evolution
            self.R0 = self.parameters['R0']

    def no_evolution(self,z):
        # no evolution of the merger rate with redshift
        return self.R0
            
    def Madau_evolution(self,z):

        zp = self.parameters['zp']
        alpha = self.parameters['madau_alpha']
        beta = self.parameters['madau_beta']
        R0 = self.parameters['R0']

        C = 1+(1+zp)**(-alpha-beta)
        return C*R0*((1+z)**alpha)/(1+((1+z)/(1+zp))**(alpha+beta)) #Equation 2 in https://arxiv.org/pdf/2003.12152.pdf

    def time_delay(self,z):

        return self.time_delay_interp(z)

    def time_delay_interpolation(self):

        self.t_min = self.parameters['t_min']
        self.d = self.parameters['d']
        self.z_plus = self.parameters['z_plus']
        self.z_f_max = self.parameters['z_f_max']
        z_m = np.linspace(0,self.z_f_max,1000)
        self.t_to_z = interp1d(self.z_to_time(z_m),z_m)
        self.dz = 0.01
        
        self.t_f_min = self.z_to_time(self.z_f_max)
        t_m = self.z_to_time(z_m)
        z_f_min, z_f_max = self.redshifts_limits(z_m,t_m)
        
        idx = np.where((z_f_max<=z_f_max)&(z_f_min<z_f_max))[0]
        R = np.zeros(len(z_m))
        for i in idx:
            R[i] = self.R_z(z_f_min[i],z_f_max[i],t_m[i])
        z_f_min_0, z_f_max_0 = self.redshifts_limits(np.array([0]),np.array([self.z_to_time(0)]))
        R_0 = self.R_z(z_f_min_0[0],z_f_max_0[0],self.z_to_time(0))
            
        return interp1d(z_m,self.parameters['R0']*R/R_0)

    def R_z(self,z_f_min,z_f_max,t_m):

        N = int((z_f_max-z_f_min)/self.dz)
        if N>1:
            zs = np.linspace(z_f_min,z_f_max,N).reshape(-1)
            t_f = self.z_to_time(zs).reshape(-1)
            R_SFR = self.Madau_evolution(zs).reshape(-1)
            P_t = self.P_t(t_m,t_f).reshape(-1)
            dt_dz = self.dt_dz(zs).reshape(-1)

            return simps(P_t*R_SFR*dt_dz,zs)
        else:
            return 0

    def P_t(self,t_m,t_f):
        
        return (t_m-t_f)**(-self.d)

    def dt_dz(self,z):

        Mpc_to_km = 3.086e+19
        sec_to_year = 3.17098e-8
        to_Gyr = Mpc_to_km*sec_to_year*1e-9
        r = 1/(self.cosmo.ap_cosmo.H0.value*self.cosmo.ap_cosmo.efunc(z)*(1+z))

        return r*to_Gyr
    
    def redshifts_limits(self,z_vals,t_m):
        
        t_f_max = t_m - self.t_min
        idx = np.where(t_f_max>self.t_f_min)[0]
        z_f_min, z_f_max = np.zeros(len(z_vals)),np.zeros(len(z_vals))
        z_f_min[idx] = self.time_to_z(t_f_max[idx])
        z_f_max[idx] = z_vals[idx]+self.z_plus
        idx = np.where(z_f_max<z_f_min)[0]
        z_f_min[idx], z_f_max[idx] = 0, 0

        return z_f_min, z_f_max

    def z_to_time(self,z):
        
        return self.cosmo.ap_cosmo.age(z).to(u.Gyr).value

    def time_to_z(self,t):
        
        return self.t_to_z(t)
