import bilby
import numpy as np
import pkg_resources
import json
import os
import h5py
import subprocess

class Posteriors(object):
    def __init__(self,full_pe=False):

        self.full_pe = full_pe

    def get_parameters_from_file(self,injections_parameters,idx):

        event_parameters = dict()
        event_parameters['m1'] = injections_parameters['m1d'][idx]
        event_parameters['m2'] = injections_parameters['m2d'][idx]
        event_parameters['chi_1'] = injections_parameters['chi_1'][idx]
        event_parameters['chi_2'] = injections_parameters['chi_2'][idx]
        event_parameters['theta_1'] = injections_parameters['theta_1'][idx]
        event_parameters['theta_2'] = injections_parameters['theta_2'][idx]
        event_parameters['dl'] = injections_parameters['dls'][idx]
        event_parameters['ra'] = injections_parameters['ras'][idx]
        event_parameters['dec'] = injections_parameters['decs'][idx]
        event_parameters['psd'] = injections_parameters['psds'][idx]
        event_parameters['det'] = injections_parameters['dets_pe'][idx]
        event_parameters['inc'] = injections_parameters['incs'][idx]
        event_parameters['phi'] = injections_parameters['phis'][idx]
        event_parameters['psi'] = injections_parameters['psis'][idx]
        event_parameters['snr'] = injections_parameters['snrs'][idx]
        event_parameters['duration'] = injections_parameters['durations'][idx]
        event_parameters['geocent_time'] = injections_parameters['geocent_time'][idx]
        try:
            event_parameters['seed'] = injections_parameters['seeds'][idx]
        except:
            event_parameters['seed'] = int(np.random.rand(1)*100000) #is here for runs of MDC to be compatible. 

        return event_parameters

    def event_with_idx(self,injections_parameters,output,idx,pe_arguments):

        event_parameters = self.get_parameters_from_file(injections_parameters,idx)
        self.bilby_pe(event_parameters,output,pe_arguments)

    def bilby_pe(self,event_parameters,output,pe_arguments):
        if pe_arguments['fake_posteriors']:
            print("Generating fake posteriors...")
            ra,dec,dl,m1,m2,theta_jn = self.fake_posteriors(event_parameters)
        else:
            psd = event_parameters['psd']
            det = event_parameters['det'].tolist()

            data_path = pkg_resources.resource_filename('GWUniverse', 'data/')
            injection_parameters = dict(a_1=event_parameters['chi_1'],
                                        a_2=event_parameters['chi_2'],
                                        tilt_1=event_parameters['theta_1'],
                                        tilt_2=event_parameters['theta_2'],
                                        geocent_time=event_parameters['geocent_time'],
                                        mass_1=event_parameters['m1'],
                                        phi_12=0.,
                                        phi_jl=0.,
                                        mass_2=event_parameters['m2'],
                                        luminosity_distance=event_parameters['dl'],
                                        ra=event_parameters['ra'],
                                        dec=event_parameters['dec'],
                                        theta_jn=event_parameters['inc'],
                                        psi=event_parameters['psi'],
                                        phase=event_parameters['phi'])
            injection_parameters['chirp_mass'] = self.get_chirp(injection_parameters['mass_1'],injection_parameters['mass_2'])
            duration = event_parameters['duration']
            sampling_frequency = pe_arguments['sampling_frequency']
            waveform_arguments = dict(waveform_approximant=pe_arguments['waveform'],reference_frequency=50.)
            waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(sampling_frequency=sampling_frequency,
                                                                               duration=duration+1.5,
                                                                               frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                                                                               parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                                                                               waveform_arguments=waveform_arguments)

            ifos = bilby.gw.detector.InterferometerList(det)
            for ifo in ifos:
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=data_path +str(ifo.name)+ '_'+ str(psd) + '_strain.txt')
            np.random.seed(event_parameters['seed'])
            ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency,
                                                                duration=duration+1.5,
                                                                start_time=injection_parameters['geocent_time']-duration)
            ifos.inject_signal(waveform_generator=waveform_generator,parameters=injection_parameters)

            fixed_parameters = ['a_1', 'a_2','tilt_1','tilt_2','phi_12', 'phi_jl']
            if self.full_pe: fixed_parameters = []

            priors = self.set_up_priors(injection_parameters,fixed_parameters,output,pe_arguments['wide_priors'],event_parameters['seed'])
            likelihood = bilby.gw.likelihood.GravitationalWaveTransient(interferometers=ifos, waveform_generator=waveform_generator,
                                                                        distance_marginalization=pe_arguments['distance_margi'],priors=priors)


            result = bilby.core.sampler.run_sampler(likelihood=likelihood, priors=priors, sampler='dynesty',npool=pe_arguments['npool'],
                                                    outdir=output+'/PE',nlive=pe_arguments['nlive'],injection_parameters=injection_parameters,
                                                    conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,dlogz=pe_arguments['dlogz'])
            with open(output+'/PE/label_result.json','r') as json_file:
                data = json.loads(json_file.read())
                ra = data['posterior']['content']['ra']
                dec = data['posterior']['content']['dec']
                dl = data['posterior']['content']['luminosity_distance']
                m1 = data['posterior']['content']['mass_1']
                m2 = data['posterior']['content']['mass_2']
                theta_jn = data['posterior']['content']['theta_jn']

        self.save_to_file(ra,dec,dl,m1,m2,theta_jn,output)
        if not pe_arguments['fake_posteriors']:
                self.create_skymap(output)

    def set_up_priors(self,injection_parameters,fixed_parameters,output,wide_priors,seed):

        priors = bilby.gw.prior.BBHPriorDict()
        
        if not wide_priors:
            mass_ratio = injection_parameters['mass_2']/injection_parameters['mass_1']
            q_min = np.array((mass_ratio-0.2,mass_ratio-0.5),dtype=np.float16)
            idx = np.where(q_min>0)[0]
            m = 0.001
            if len(idx)>0: m = np.max((0.001,np.min(q_min[idx])))

            q_max = np.array((mass_ratio+0.2,mass_ratio+0.5),dtype=np.float16)
            idx = np.where(q_max<=1)[0]
            m_m = 1
            if len(idx)>0: m_m = np.min((1,np.max(q_max[idx])))
            priors['mass_ratio'] = bilby.core.prior.Uniform(minimum=m, maximum=m_m, name='mass_ratio', latex_label='$\\mathcal{M}$')

            chirp_mass = self.get_chirp(injection_parameters['mass_1'],injection_parameters['mass_2'])
            chirp_min = np.array((chirp_mass-5,chirp_mass-2,chirp_mass-1),dtype=np.float16)
            idx = np.where(chirp_min>0)[0]
            m = 0.1
            if len(idx)>0: m = np.max((0.1,np.min(chirp_min[idx])))
            priors['chirp_mass'] = bilby.core.prior.Uniform(minimum=m, maximum=injection_parameters['chirp_mass']+5, name='chirp_mass', latex_label='$\\mathcal{M}$')

            m1_min = np.array((injection_parameters['mass_1']-15,injection_parameters['mass_1']-10,injection_parameters['mass_1']-5),dtype=np.float16)
            idx = np.where(m1_min>0)[0]
            m = 1
            if len(idx)>0: m = np.max((1.,np.min(m1_min[idx])))
            priors['mass_1'] = bilby.core.prior.Constraint(minimum=m, maximum=injection_parameters['mass_1']+15,name='mass_1')

            m2_min = np.array((injection_parameters['mass_2']-15,injection_parameters['mass_2']-10,injection_parameters['mass_2']-5),dtype=np.float16)
            idx = np.where(m2_min>0)[0]
            m = 1
            if len(idx)>0: m = np.max((1.,np.min(m2_min[idx])))
            priors['mass_2'] = bilby.core.prior.Constraint(minimum=m, maximum=injection_parameters['mass_2']+15,name='mass_2')

            dl_min = np.array((injection_parameters['luminosity_distance']-2000,injection_parameters['luminosity_distance']-1000,injection_parameters['luminosity_distance']-500),dtype=np.float16)
            idx = np.where(dl_min>0)[0]
            m = 10
            if len(idx)>0: m = np.max((10.,np.min(dl_min[idx])))
            priors['luminosity_distance'] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=m, maximum=injection_parameters['luminosity_distance']+2000, unit='Mpc', latex_label='$d_L$')

        else:
            mass_max = 500
            dist_max = 15000
            if injection_parameters['mass_1']<30: 
                mass_max = 100
                dist_max = 9000
            
            priors['mass_ratio'] = bilby.core.prior.Uniform(minimum=0.01, maximum=1, name='mass_ratio', latex_label='$\\mathcal{M}$')
            priors['mass_1'] = bilby.core.prior.Constraint(minimum=1, maximum=mass_max,name='mass_1')
            priors['mass_2'] = bilby.core.prior.Constraint(minimum=1, maximum=mass_max,name='mass_2')
            
            #m=injection_parameters['chirp_mass']-0.6*injection_parameters['chirp_mass']
            #mm=injection_parameters['chirp_mass']+0.6*injection_parameters['chirp_mass']
            m = 0.1*injection_parameters['chirp_mass']
            mm = 2*injection_parameters['chirp_mass']
            priors['chirp_mass'] = bilby.core.prior.Uniform(minimum=m, maximum=mm, name='chirp_mass', latex_label='$\\mathcal{M}$')
            
            priors['luminosity_distance'] = bilby.core.prior.PowerLaw(alpha=2, name='luminosity_distance', minimum=0, maximum=dist_max, unit='Mpc', latex_label='$d_L$')
                        
            
        priors['geocent_time'] = bilby.core.prior.Uniform(minimum=injection_parameters['geocent_time']-0.1, maximum=injection_parameters['geocent_time']+0.1,name='mass_2')

        for key in fixed_parameters: priors[key] = injection_parameters[key]
        txt = ''
        for key in priors.keys(): txt+= str(key)+' = '+str(priors[key])+'\n'
        txt+= 'seed = '+str(seed)+'\n'
        priors_file = open(output+'/priors.priors','w')
        priors_file.write(txt)
        priors_file.close()

        return priors


    def save_to_file(self,ra,dec,dl,m1,m2,theta_jn,output):

        parameters = ['luminosity_distance','ra','dec','mass_1','mass_2','theta_jn']
        Nsamp = len(dl)
        data = np.concatenate((dl,ra,dec,m1,m2,theta_jn)).reshape(len(parameters),Nsamp).T
        h5py_data = np.array([tuple(i) for i in data], dtype=[tuple([i, 'float64']) for i in parameters])
        file = h5py.File(output+'/posterior.h5', 'w')
        dset = file.create_dataset("C01:IMRPhenomPv2/posterior_samples", data=h5py_data)
        file.close()

    def create_skymap(self,path):

        bashCommand = "ligo-skymap-from-samples --disable-multiresolution --samples {}/posterior.h5 --outdir {}/ --maxpts 5000".format(path,path)
        subprocess.run([bashCommand],shell=True)

    def get_chirp(self,m1,m2):

        return ((m1*m2)**(3/5))/((m1+m2)**(1/5))

    def fake_posteriors(self,event_parameters):

        npts = 10000
        relative_error = 0.01
        vals = np.random.normal(1,relative_error,npts)
        ra = event_parameters['ra']*vals
        vals = np.random.normal(1,relative_error,npts)
        dec = event_parameters['dec']*vals
        vals = np.random.normal(1,relative_error,npts)
        dl = event_parameters['dl']*vals
        vals = np.random.normal(1,relative_error,npts)
        m1 = event_parameters['m1']*vals
        vals = np.random.normal(1,relative_error,npts)
        m2 = event_parameters['m2']*vals
        vals = np.random.normal(1,relative_error,npts)
        theta_jn = event_parameters['inc']*vals
        return ra,dec,dl,m1,m2,theta_jn
