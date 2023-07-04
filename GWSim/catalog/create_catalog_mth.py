from GWSim.utils import Interpolate_function,Integrate_1d
import xpol
import pickle
import h5py
import healpy as hp
import numpy as np
import matplotlib.pylab as plt
from importlib import reload
from scipy import interpolate

def get_power_spectrum(imap,imask,beamsize_deg,lbin):
    # imap and imask are the raw skymap and mask (pixels are 0 or 1, no apodization)
    gmap = imap.copy()
    gmask = imask.copy()
    # remove average of the skymap (visible part, outside of the mask)
    wok = np.where(gmask != 0)
    avgmap = np.mean(gmap[wok])
    gmap[wok] -= avgmap

    # prepare binning for Xpol
    gnside = hp.npix2nside(len(gmap))
    lmax = 3*gnside-1
    binning = xpol.Bins.fromdeltal(2,lmax,lbin)
    apod_mask = xpol.apodization(gmask,1)
    xp = xpol.Xpol(apod_mask,bins=binning,verbose=False)
    Dl = True
    # beam
    fwhm = np.deg2rad(beamsize_deg) # beamsize_deg should be equal to the pixel size
    # power spectrum of the beam
    bell = hp.gauss_beam(fwhm,lmax=lmax)
    # set Q, U maps to 0*gmap and compute Cl
    empty = 0*gmap
    # compute unbiased Cl of the input skymap
    # the Cl values are binned in l (with width lbin), to avoid oscillations
    pcl,cl = xp.get_spectra([gmap,empty,empty],Dl=Dl,bell=bell)
    clinput = cl[0]/((binning.lbin*(binning.lbin+1)/(2*np.pi)))
    return binning,pcl,clinput

def make_similar_map(imap,imask,beamsize_deg,lbin,seed):
    # imap and imask are the raw skymap and mask (pixels are 0 or 1, no apodization)
    gmap = imap.copy()
    gmask = imask.copy()
    # remove average of the skymap (visible part, outside of the mask)
    wok = np.where(gmask != 0)
    avgmap = np.mean(gmap[wok])
    gmap[wok] -= avgmap

    # prepare binning for Xpol
    gnside = hp.npix2nside(len(gmap))
    lmax = 3*gnside-1
    binning = xpol.Bins.fromdeltal(2,lmax,lbin)
    apod_mask = xpol.apodization(gmask,1)
    xp = xpol.Xpol(apod_mask,bins=binning,verbose=False)
    Dl = True
    # beam
    fwhm = np.deg2rad(beamsize_deg) # beamsize_deg should be equal to the pixel size
    # power spectrum of the beam
    bell = hp.gauss_beam(fwhm,lmax=lmax)
    # set Q, U maps to 0*gmap and compute Cl
    empty = 0*gmap
    # compute unbiased Cl of the input skymap
    # the Cl values are binned in l (with width lbin), to avoid oscillations
    pcl,cl = xp.get_spectra([gmap,empty,empty],Dl=Dl,bell=bell)
    clinput = cl[0]/((binning.lbin*(binning.lbin+1)/(2*np.pi)))
    
    # then interpolate the Cls for all l in [2;lmax]
    interp_cl = interpolate.interp1d(binning.lbin,clinput,kind='cubic')
    # synfast expects a vector of Cl for l=0...lmax so have to deal with unset values
    l_min = int(np.ceil(np.min(binning.lbin)))
    l_max = int(np.floor(np.max(binning.lbin)))
    all_l = np.arange(0,l_max+1) # [0;lmax]
    lseen = np.arange(l_min,l_max+1) # [l_min;l_max]
    all_cl_seen = interp_cl(lseen)
    all_cl = np.concatenate((np.zeros(l_min),all_cl_seen),axis=0) # Cl = 0 in [0;l_min-1] U Cl in [l_min;l_max] = [0;l_max]
    plt.plot(binning.lbin,clinput,'o',all_l,all_cl,'+')
    plt.plot(all_l,all_cl,'+')
        
    # then create new map with same Cl corresponding to l values in [0;l_max]
    np.random.seed(seed)
    dTsim = hp.synfast(all_cl,gnside,new=True,pixwin=True,fwhm=fwhm)

    # put back the dipole as Xpol computes the power spectrum for ell >= 2
    gnpix = hp.nside2npix(gnside)
    # compute monopole/dipole taking into account the mask: multiply the map by the mask and set the bad pixels value to 0
    # the mask here must be the original mask (with 1 and 0, not apodized)
    gmap_monodip = hp.fit_dipole(gmap*gmask,bad=0)
    x, y, z = hp.pix2vec(gnside,np.arange(0,gnpix))
    dipole_map = gmap_monodip[1][0]*x + gmap_monodip[1][1]*y + gmap_monodip[1][2]*z
    final_map = dTsim + dipole_map# + gmap_monodip[0]
#    final_map -= final_map.min()
    final_map *= gmask
    # set final_map to 0 averaged outside of the mask
    final_map[wok] -= np.mean(final_map[wok])
    # check: the Cl of final_map should be the same than the input Cl within the cosmic variance
    # the check is done on the 0-averaged final map
    pclfinal,llclfinal = xp.get_spectra([final_map,0*final_map,0*final_map],Dl=Dl,bell=bell)
    clfinal = llclfinal[0]/((binning.lbin*(binning.lbin+1)/(2*np.pi)))
#    fig = plt.figure()
#    plt.plot(binning.lbin,clinput/clfinal,'.')
#    plt.plot(binning.lbin,1+0*binning.lbin,'r')
#    plt.ylim([0, 2])
    cvplus = clinput+np.sqrt(2./(2*binning.lbin+1))/xpol.fsky(gmask)*clinput
    cvminus = clinput-np.sqrt(2./(2*binning.lbin+1))/xpol.fsky(gmask)*clinput
#    plt.savefig("cls.png")
    return apod_mask,clinput,cvplus,cvminus,all_l,all_cl,clfinal,binning,final_map

def rescale_map(map,mask,newmin,newmax):
    wok = np.where(mask != 0)
    oldmin = map[wok].min()
    oldmax = map[wok].max()
    map[wok] = newmin+(newmax-newmin)*(map[wok]-oldmin)/(oldmax-oldmin)
    return map

def build_skymap(ipix,nside):
    # ipix is the list of pixels seen
    umap = np.zeros(hp.nside2npix(nside))
    sipix = np.sort(ipix)
    unique, counts = np.unique(sipix,return_counts=True)
    umap[unique] = counts
    return umap

def select_galaxies(gc,app_mag):
    npix = gc.npix
    nside = gc.nside
    eff_map = np.zeros(npix)
    eff_gal = [0]
    pp = gc.Universe.ra
    tt = (np.pi/2.-gc.Universe.dec)
    ipix = hp.ang2pix(nside,tt,pp)
    app_mag_lim = gc.final_map[ipix]
    selection = np.where( (app_mag < app_mag_lim) & (gc.catalog_mask_skymap[ipix] == 1) & (gc.Universe.z > gc.z_min) & (gc.Universe.z < gc.z_max) )[0]
    eff_map = build_skymap(ipix[selection],gc.nside)
    return eff_map,selection

class GalaxyCatalog(object):
#    def __init__(self,Universe,catalog_skymap,catalog_mask_skymap,mmin,mmax,red_survey,red_error,seed=0,lbin=20):
    def __init__(self,Universe,catalog_skymap,catalog_mask_skymap,cmdline):
        self.Universe = Universe
        self.npix = len(catalog_skymap)
        self.nside = hp.npix2nside(self.npix)
        self.m_min = cmdline.m_min
        self.m_max = cmdline.m_max
        self.z_min = cmdline.z_min
        self.z_max = cmdline.z_max
        self.red_survey = cmdline.red_survey
        self.catalog_skymap = catalog_skymap # number of galaxies per pixel, user defined
        self.catalog_mask_skymap = catalog_mask_skymap # 1 or 0, user defined
        self.cmap = 0
        self.apod_mask = 0 # for the apodized mask
        self.final_map = 0 # the simulated skymap of varying mth with same Cl than the input galaxy catalog
        self.cl_input_map = 0 # the binned Cl of the input map
        self.cl_final_map = 0
        self.cvplus = 0 # the binned Cl of the input map + cosmic variance
        self.cvminus = 0 # the binned Cl of the input map - cosmic variance
        self.binning = 0 # the binned l
        self.z = 0
        self.ra = 0
        self.dec = 0
        self.z = 0
        self.abs_magn = 0
        self.app_magn = 0
        self.sigmaz = 0
        self.z_real = 0
        self.red_error = cmdline.red_error
        # compute monopole/dipole taking into account the mask: multiply the map by the mask and set the bad pixels value to 0
        # the mask here must be the original mask (with 1 and 0, not apodized)
        self.monodip = hp.fit_dipole(self.catalog_skymap*self.catalog_mask_skymap,bad=0)
        self.beamsize_deg = np.sqrt(4*np.pi/self.npix)*(180/np.pi)
        # seed for skymap generation with synfast
        self.seed = cmdline.seed
        # binning for l (for interpolation)
        self.lbin = cmdline.lbin

    def create(self):
        self.cmap = self.catalog_skymap*self.catalog_mask_skymap # just to be sure
        print("Computing power spectrum and generating new map...")
        self.apod_mask,self.cl_input_map,self.cvplus,self.cvminus,all_l,all_cl,self.cl_final_map,self.binning,self.final_map = make_similar_map(self.cmap,self.catalog_mask_skymap,self.beamsize_deg,self.lbin,self.seed)
        
        self.final_map = rescale_map(self.final_map,self.catalog_mask_skymap,self.m_min,self.m_max)

        print("Building skymap of all galaxies in the universe...")
        pp = self.Universe.ra
        tt = np.pi/2.-self.Universe.dec # ra, dec in RADIANS!
        ipix = hp.ang2pix(self.nside,tt,pp)
        umap = build_skymap(ipix,self.nside)
        print("Computing apparent magnitudes...")
        app_mag = 25+self.Universe.abs_magn+5*np.log10(self.Universe.cosmology.dl_zH0(self.Universe.z))
        print("Building simulated skymap and catalog...")
        self.mycatalog,self.list_gal = select_galaxies(self,app_mag)
        self.ra = self.Universe.ra[self.list_gal]
        self.dec = self.Universe.dec[self.list_gal]
        self.z = self.Universe.z[self.list_gal]
        self.z_real = self.Universe.z[self.list_gal]
        self.abs_magn = self.Universe.abs_magn[self.list_gal]
        self.app_magn = app_mag[self.list_gal]

        if self.red_survey=='photoz':
                delta_z = 0.02
        elif self.red_survey=='specz':
                delta_z = 0.001
        else:
                raise ValueError("The type of redshift estimate is not implemented. Select between specz or photoz")

        self.sigmaz = delta_z*(1+self.z)
        if self.red_error==True:
                print("Drawing random values from normal distributions for the redshifts")
                self.z = np.random.normal(self.z,self.sigmaz)
