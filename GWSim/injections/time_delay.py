import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import numpy.random as rand
from astropy import units as u
import multiprocess as mp
import os
import time
import GWSim.random.custom_math_priors as cmp


class Time_delay(object):

    def __init__(self,hyper_params_dict,z,cosmo,pools=1):

        self.hyper_params_dict = hyper_params_dict
        self.z = z
        self.cosmo = cosmo
        self.z_plus = self.hyper_params_dict['z_plus']
        self.z_f_max = self.hyper_params_dict['z_f_max']
        self.Z_star = self.hyper_params_dict['Z_star']
        self.alpha_delay = self.hyper_params_dict['alpha_delay']
        self.gamma_delay = self.hyper_params_dict['gamma_delay']
        self.d = self.hyper_params_dict['d']
        self.mu_g = self.hyper_params_dict['mu_g']
        self.t_min = self.hyper_params_dict['t_min']
        self.zeta = self.hyper_params_dict['zeta']
        
        z_m = np.linspace(0,self.z_f_max,10000)
        self.t_to_z = interp1d(self.z_to_time(z_m),z_m)
        self.pools = pools
        self.manager = mp.Manager()
        self.dz = 0.01
        self.t_f_min = self.z_to_time(self.z_f_max)
        self.nbins_window = 200
        self.ms = np.linspace(self.hyper_params_dict['mmin'],self.hyper_params_dict['mmax'],self.nbins_window)
        self.dm = self.ms[1]-self.ms[0]
        
        self.t_m = self.z_to_time(z)
        self.z_f_min, self.z_f_max = self.redshifts_limits(z,self.t_m)

    def sample(self,mass_values):

        self.mass_values = mass_values
        m1 = np.zeros(len(self.z))
        m2 = np.zeros(len(self.z))

        indexes = np.arange(len(self.z))
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
        print("Back to TimeDelay main thread, gathering results...")
        for i in range(len(self.z)):
            m1[i] = results[i][0]
            m2[i] = results[i][1]
        self.m1_eff = np.zeros(len(self.mass_values),dtype=float)
        self.m2_eff = np.zeros(len(self.mass_values),dtype=float)
        for i in range(self.pools):
            self.m1_eff += m1ds[i]
            self.m2_eff += m2ds[i]

        return m1,m2,self.m1_eff,self.m2_eff

    def sampling(self,i,n,results,m1dists,m2dists):

        print("child process pid: {}, thread_idx: {}, niter: {}".format(os.getpid(),i,len(n)))

        for idx_n in n:

            m1, m2, m1s_dist, m2s_dist = self.masses(idx_n)
            lr = results[idx_n]
            lr = [m1,m2]
            results[idx_n] = lr

            lr = m1dists[i] # get address
            lr += m1s_dist # update data
            m1dists[i] = lr # write data
            lr = m2dists[i]
            lr += m2s_dist
            m2dists[i] = lr

        return

    def masses(self,idx_n):

        W_td, mu = self.W_td(self.z_f_min[idx_n],self.z_f_max[idx_n],self.t_m[idx_n],self.ms)
                
        # for mass m1
        m1law = cmp.PowerLawGaussian_math(alpha=-self.hyper_params_dict['alpha'], # add the '-' sign for the crazy convention
                                          min_pl=self.hyper_params_dict['mmin'],
                                          max_pl=self.hyper_params_dict['mmax'],
                                          lambda_g=self.hyper_params_dict['lambda_peak'],
                                          mean_g=mu,
                                          sigma_g=self.hyper_params_dict['sigma_g'],
                                          min_g=self.hyper_params_dict['mmin'],
                                          max_g=mu+5*self.hyper_params_dict['sigma_g'])

        # check if smoothing is possible
        sfactor = cmp._S_factor(self.ms,self.hyper_params_dict['mmin'],self.hyper_params_dict['delta_m'])
        if not np.isnan(np.max(sfactor)) and (np.max(sfactor)>0):
            m1law = cmp.SmoothedProb(origin_prob=m1law,
                                   bottom=self.hyper_params_dict['mmin'],
                                   bottom_smooth=self.hyper_params_dict['delta_m'])

        PLG = m1law.prob(self.ms)

        # then, apply window function
        P_M = PLG*W_td
        # finally, draw one realisation of m1
        cdf = np.cumsum(P_M)
        cdf /= np.max(cdf)
        cdf[0] = 0 # start from 0
        icdf = interp1d(cdf,self.ms)
        m1 = icdf(rand.uniform(0,1,1))[0]
        
        # for mass m2, limit to range [mmin;m1]
        m2law = cmp.PowerLaw_math(alpha=self.hyper_params_dict['beta'],
                                  min_pl=self.hyper_params_dict['mmin'],
                                  max_pl=m1)
        ms = np.linspace(self.hyper_params_dict['mmin'],m1,200)
        # check if smoothing is possible
        sfactor = cmp._S_factor(ms,self.hyper_params_dict['mmin'],self.hyper_params_dict['delta_m'])
        if not np.isnan(np.max(sfactor)) and (np.max(sfactor)>0):
            m2law = cmp.SmoothedProb(origin_prob=m2law,
                                     bottom=self.hyper_params_dict['mmin'],
                                     bottom_smooth=self.hyper_params_dict['delta_m'])

        PL2 = m2law.prob(ms)
        cdf = np.cumsum(PL2)
        cdf /= np.max(cdf)
        cdf[0] = 0
        icdf = interp1d(cdf,ms)
        m2 = icdf(rand.uniform(0,1,1))[0]

        # compute effective pdf for m1, m2
        W_td, _ = self.W_td(self.z_f_min[idx_n],self.z_f_max[idx_n],self.t_m[idx_n],self.mass_values)
        m1s_dist = m1law.prob(self.mass_values)*W_td
        m2s_dist = 0*self.mass_values
        wm = np.where( (self.mass_values >= m2law.minimum) & (self.mass_values <= m2law.maximum) )[0]
        if len(wm)>0:
            m2s_dist[wm] = m2law.prob(self.mass_values[wm])
        else:
            print("Anomaly time_delay m2-pdf: no valid mass range.")
        if np.isnan(np.max(m2s_dist)):
            print("Anomaly time_delay m2-pdf: nan!")
            for i,ival in enumerate(self.mass_values):
                val = -1
                print("power: {}, dm: {}, {} {} min:{}, max:{}, {}".format(self.hyper_params_dict['beta'],self.hyper_params_dict['delta_m'],i,self.mass_values[i],m2law.minimum,m2law.maximum,m2s_dist[i]))
#                if (self.mass_values[i]>= m2law.minimum) and (self.mass_values[i]<m2law.maximum): val = m2law.prob(self.mass_values[i])
#                print("{} {} {} : calc: {}".format(i,self.mass_values[i],m2s_dist[i],val))
                #            print(self.mass_values)
            print(m2law.minimum,m2law.maximum)
            print(m1)
            print(m2s_dist)

        return m1,m2,m1s_dist,m2s_dist

    def P_t(self,t_m,t_f):
        
        return (t_m-t_f)**(-self.d)

    def dt_dz(self,z):

        Mpc_to_km = 3.086e+19
        sec_to_year = 3.17098e-8
        to_Gyr = Mpc_to_km*sec_to_year*1e-9
        r = 1/(self.cosmo.ap_cosmo.H0.value*self.cosmo.ap_cosmo.efunc(z)*(1+z))

        return r*to_Gyr

    def W_td(self,z_f_min,z_f_max,t_m,m1):

        N = int((z_f_max-z_f_min)/self.dz)
        if N>1:
            zs = np.linspace(z_f_min,z_f_max,N).reshape(-1)
            w = self.window_function(zs,m1)
            t_f = self.z_to_time(zs).reshape(-1)
            P_t = self.P_t(t_m,t_f).reshape(-1)
            dt_dz = self.dt_dz(zs).reshape(-1)
            W_td = simps(P_t*dt_dz*w,zs)
            norm = simps(W_td,m1)
            W_td/=norm
            if len(np.where(np.diff(W_td)!=0)[0])>0:
                mu = m1[np.min(np.where(np.diff(W_td)!=0)[0])]
            else:
                m1_max = np.linspace(m1[-1],m1[-1]*3,400)
                w = self.window_function(zs,m1_max)
                W_td_max = simps(P_t*dt_dz*w,zs)
                if len(np.where(np.diff(W_td_max)!=0)[0])>0:
                    mu = m1_max[np.min(np.where(np.diff(W_td_max)!=0)[0])]
                else:
                    mu = m1_max[-1]
            return  W_td, mu
        else:
            W_td = np.zeros(len(m1))
            return W_td, 2

    def window_function(self,zs,m1):

        mus = self.mu_g -self.alpha_delay*(self.gamma_delay*zs+self.zeta-np.log10(self.Z_star))
        ms =  np.linspace(np.min(m1)-mus,np.max(m1)-mus,len(m1))
        res = np.heaviside(-ms,1)
        norm = simps(res,m1,axis=0)
        norm[np.where(norm==0)] = 1
        
        return res/norm

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

    
