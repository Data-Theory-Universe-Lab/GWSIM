import os
import numpy as np
from numpy.random import uniform as uni
import math

####################################################

class Masses(object):

    def __init__(self):

        self.DefaultRandomValues()
        self.evolution = False

    def DefaultRandomValues(self):

        self.alpha = uni(1.5,12,1)[0]
        self.beta = uni(-4,12,1)[0]
        self.Mmax = uni(70,200,1)[0]
        self.Mmin = uni(2,10,1)[0]
        self.mu = uni(10,70,1)[0]
        self.sigma = uni(0.4,10,1)[0]
        self.Lambda = uni(0,1,1)[0]
        self.delta_m = uni(0,10,1)[0]

    def Evolution(self,z_max):

        self.evolution = True
        self.z_max = z_max
        self.epsilon_alpha = uni(-self.alpha/self.z_max,self.alpha/self.z_max,1)[0]
        self.epsilon_beta = uni(-self.beta/self.z_max,self.beta/self.z_max,1)[0]
        self.epsilon_Mmax = uni(0,self.Mmax/self.z_max,1)[0]
        self.epsilon_Mmin = uni(0,self.Mmin/self.z_max,1)[0]
        self.epsilon_mu = uni(max(0,self.epsilon_Mmin+(self.Mmin-self.mu)/self.z_max),min(self.mu/self.z_max,self.epsilon_Mmax+(self.Mmax-self.mu)/self.z_max),1)[0]
        self.epsilon_Lambda = uni(-self.Lambda/self.z_max,(1-self.Lambda)/self.z_max,1)[0] # to have lambda+epsilon_lambda*z always in [0;1] for z in [0;z_max]
        self.epsilon_sigma = uni(0,self.sigma/self.z_max,1)[0]
        self.epsilon_delta_m = 0 #uni(0,self.delta_m/self.z_max,1)[0]

    def PrintArgs(self):

        cmd = '--alpha '+str(self.alpha)+' --beta '+str(self.beta)+\
            ' --mu '+str(self.mu)+' --sigma '+str(self.sigma)+\
            ' --Lambda '+str(self.Lambda)+' --Mmax '+str(self.Mmax)+\
            ' --Mmin '+str(self.Mmin)+' --delta_m '+str(self.delta_m)+' '

        if self.evolution:
            cmd += ' --epsilon_alpha '+str(self.epsilon_alpha)+\
                ' --epsilon_beta '+str(self.epsilon_beta)+\
                ' --epsilon_Mmax '+str(self.epsilon_Mmax)+\
                ' --epsilon_Mmin '+str(self.epsilon_Mmin)+\
                ' --epsilon_mu '+str(self.epsilon_mu)+\
                ' --epsilon_Lambda '+str(self.epsilon_Lambda)+\
                ' --epsilon_sigma '+str(self.epsilon_sigma)+\
                ' --epsilon_delta_m '+str(self.epsilon_delta_m)+\
                ' --red_depend 1 '+' --zcut '+str(self.z_max)+' '
        return cmd

####################################################

class MergerRate(object):

    def __init__(self):

        self.DefaultRandomValues()

    def DefaultRandomValues(self):

        self.R0 = uni(0,100,1)[0]
        self.Madau_alpha = uni(0,5,1)[0]
        self.Madau_beta = uni(0,4,1)[0]
        self.Madau_zp = uni(0,4,1)[0]

    def PrintArgs(self):

        cmd = ' --R0 '+str(self.R0)+' --Madau_alpha '+\
            str(self.Madau_alpha)+' --Madau_beta '+\
            str(self.Madau_beta)+\
            ' --Madau_zp '+str(self.Madau_zp)+' '
        return cmd

####################################################

class Cosmo(object):

    def __init__(self):

        self.DefaultRandomValues()

    def DefaultRandomValues(self):

        self.H0 = uni(10,200,1)[0]
        self.w0 = uni(-3,0,1)[0]
        self.Om = uni(0,1,1)[0]

    def PrintArgs(self):

        cmd = ' --cosmo_model flatLCDM --H0 '+str(self.H0)+\
            ' --w0 '+str(self.w0)+' --Om '+str(self.Om)+' '
        return cmd

####################################################

class TimeDelay(object):

    def __init__(self):

        self.DefaultRandomValues()

    def DefaultRandomValues(self):

        self.d = uni(0,4,1)[0]
        self.t_min = uni(0.01,13,1)[0]
        self.alpha = uni(0,3,1)[0]
        self.gamma = uni(-3,0,1)[0]

    def PrintArgs(self):

        cmd = ' --t_min '+str(self.t_min)+\
            ' --alpha_delay '+str(self.alpha)+\
            ' --d '+str(self.d)+\
            ' --gamma_delay '+str(self.gamma)+\
            ' --time_delay 1 --red_depend 1 '
        return cmd

####################################################

class Spin(object):

    def __init__(self,spin_model):

        self.spin_model = spin_model
        self.spin_aligned = 0
        choice = np.random.uniform(0,1,1)[0]<0.5
        if choice: self.spin_aligned = 1

    def PrintArgs(self):

        cmd = ' --aligned_spins '+str(self.spin_aligned)+\
            ' --spin_model '+self.spin_model+' '
        return cmd

####################################################
####################################################
####################################################

class SimuParams(object):

    def __init__(self):

        self.DefaultValues()
        self.time_delay_activated = False
        self.spin_activated = False

    def DefaultValues(self):

        self.masses = Masses()
        self.cosmo = Cosmo()
        self.rate = MergerRate()
        self.time_delay = TimeDelay()
        self.spin = Spin('uniform')
        self.seed = int(uni(0,1000000,1)[0])
        self.snr = 12
        self.Tobs = 10

    def PrintArgs(self):

        cmd = self.masses.PrintArgs()+\
            self.cosmo.PrintArgs()+\
            self.rate.PrintArgs()+\
            ' --seed '+str(self.seed)+\
            ' --snr '+str(self.snr)+\
            ' --T_obs '+str(self.Tobs)+' '
        if self.time_delay_activated:
            cmd += self.time_delay.PrintArgs()
        if self.spin_activated:
            cmd += self.spin.PrintArgs()
        return cmd

####################################################

z_max_micecat = 1.7

def case_1(i,snr,Tobs):

    tmp = SimuParams()
    tmp.snr = snr
    tmp.Tobs = Tobs
    tmp.cosmo.w0 = -1
    tmp.cosmo.Om = 0.3
    tmp.rate.Madau_alpha = 0
    tmp.rate.Madau_beta = 0
    tmp.rate.Madau_zp = 0
    return tmp.PrintArgs()

def case_2(i,snr,Tobs):

    tmp = SimuParams()
    tmp.snr = snr
    tmp.Tobs = Tobs
    tmp.cosmo.w0 = -1
    tmp.cosmo.Om = 0.3
    z_max = z_max_micecat
    tmp.masses.Evolution(z_max)
    return tmp.PrintArgs()

def case_3(i,snr,Tobs):

    tmp = SimuParams()
    tmp.snr = snr
    tmp.Tobs = Tobs
    tmp.cosmo.w0 = -1
    tmp.cosmo.Om = 0.3
    tmp.time_delay_activated = True
    # use the SFR for Madau in this case
    # but will be overwritten in GW_injections in the time_delay case
    tmp.rate.Madau_alpha = 2.7
    tmp.rate.Madau_beta = 2.9
    tmp.rate.Madau_zp = 1.9
    return tmp.PrintArgs()

def case_4(i,snr,Tobs):

    tmp = SimuParams()
    tmp.snr = snr
    tmp.Tobs = Tobs
    z_max = z_max_micecat
    tmp.masses.Evolution(z_max)
    return tmp.PrintArgs()

def case_5(i,snr,Tobs,spin):

    tmp = SimuParams()
    tmp.snr = snr
    tmp.Tobs = Tobs
    tmp.cosmo.w0 = -1
    tmp.cosmo.Om = 0.3
    z_max = z_max_micecat
    tmp.masses.Evolution(z_max)
    tmp.spin_activated = True
    tmp.spin.spin_model = spin # update value
    return tmp.PrintArgs()

################################################
################################################
################################################

path=str(input('Enter saving path (full path): '))
#name=str(input('Enter your albert.einstein name:'))
name = os.getlogin()
ii=int(input('Enter number of runs: '))
Tobs=float(input('Enter Tobs: '))
snr=float(input('Enter snr: '))
case=int(input('Enter case (1,2,3,4 or 5): '))
if case==5: spin=str(input('Enter spins model(heavy_mass, uniform, Gaussian, correlated):'))
folder='Case_'+str(case)+'_SNR_'+str(snr)+'_Tobs_'+str(Tobs)
extra_params = ' --file /home/federico.stachurski/LVK/load_catalog/reduced_catalogs/ --catalog_name MiceCatv2 \
                --z_err_perc 5 --magnitude_band g --magnitude_type absolute --NSIDE 64 --Nth 100 --mth_cat 30 --npools 8 \
                --luminosity_weight 1 --redshift_weight 1 --population_model powerlaw-gaussian --use_cosmo 1 '
p=path+'/'+folder
if not os.path.exists(p):
    os.makedirs(p)

for i in range(1,ii+1):
    if case==1:
        par=case_1(i,snr,Tobs)
    elif case==2:
        par=case_2(i,snr,Tobs)
    elif case==3:
        par=case_3(i,snr,Tobs)
    elif case==4:
        par=case_4(i,snr,Tobs)
    elif case==5:
        par=case_5(i,snr,Tobs,spin)

    params=par+extra_params+' --output Injection_'+str(i)+' '

    #exec_dir = '/home/christos.karathanasis/MDC/gwuniverse/'
    exec_dir = '/home/benoit.revenu/.conda/envs/gwsim-mdc/'
    #exec_dir = '/home/benoit.revenu//gwuniverse/'

    text='universe = vanilla\ngetenv = True\ndir = '+exec_dir+'\nexecutable = $(dir)/bin/GW_injections \
        \narguments =  '+params+'\naccounting_group = ligo.dev.o4.cbc.pe.lalinference\naccounting_group_user = '+name+'\nrequest_memory = 90000 \
        \noutput = '+p+'/'+str(i)+'.out\nrequest_disk=4000\nrequest_cpus=4\nerror = '+p+'/'+str(i)+'.err\nqueue'
    f = open(p+'/'+str(i)+".sub", "w")
    f.write(text)
    f.close()

submit=str(input('Submit jobs?(y/n) '))
if submit=='y':
    for i in range(1,ii+1):
        bashCommand = 'cd '+p+'\ncondor_submit '+str(i)+'.sub'
        os.system(bashCommand)

