import os
import pandas as pd
import numpy as np
from optparse import Option, OptionParser, OptionGroup
import sys
import fitsio
import healpy as hp


class RealCatalog(object):
    
    def __init__(self, pathname, catalog_name, zmax, zmin, z_err_perc, magnitude_band, magnitude_type, NSIDE, Nth, mth_cat):
        
        self.pathname = pathname
        self.catalog_name = catalog_name
        self.filepath = self.pathname + self.catalog_name
        self.zmax = zmax
        self.zmin = zmin
        self.z_err = z_err_perc
        self.catalog_data = pd.DataFrame()
        self.magnitude_band = magnitude_band
        self.magnitude_type = magnitude_type
        self.NSIDE = NSIDE
        self.Nth = Nth
        self.mth_cat = mth_cat
 


        
        try:
            if catalog_name =='MiceCatv2':
                print('Loading MiceCatv2 catalog... Takes about 4 minutes \n ')
                #add this if true redshift is wanted 'z_cgal', 
                

                columns =  ['ra_gal', 'dec_gal', 'z_cgal', 'z_cgal_v']
                
                #SDSS magnitudes
                magnitude_columns = ['sdss_u_abs_mag', 'sdss_u_true', 'sdss_g_abs_mag',
                                     'sdss_g_true',  'sdss_r_abs_mag', 'sdss_r_true',
                                    'sdss_i_abs_mag', 'sdss_i_true', 'sdss_z_abs_mag', 'sdss_z_true']
                
                for mag in magnitude_columns:

                    if self.magnitude_type == 'absolute':

                        if mag[-7:] == 'abs_mag':
                            if mag[5] == self.magnitude_band:
                                magnitude = mag


                    elif self.magnitude_type =='apparent': 

                        if mag[-4:] == 'true':
                            if mag[5] == self.magnitude_band:
                                magnitude = mag

                    else: 
                        raise ValueError("magnitude type or magnitude band not supported")
                
                
                columns = columns + [magnitude]
                cat_fname = self.filepath+'/reduced_MiceCatv2_v2.fits'
                npcat = fitsio.read(cat_fname, columns=columns)
                npcat = npcat.byteswap().newbyteorder()
                data_frame = pd.DataFrame.from_records(npcat)
                data_frame = data_frame[columns]
                
                if self.magnitude_type == 'apparent':
                    data_frame = data_frame.rename(columns = {'ra_gal':'ra', 'dec_gal':'dec',
                                                             'z_cgal':'z', 'z_cgal_v':'z_v', magnitude:'app_magn'})
                elif self.magnitude_type == 'absolute':    
                    data_frame = data_frame.rename(columns = {'ra_gal':'ra', 'dec_gal':'dec',
                                             'z_cgal':'z', 'z_cgal_v':'z_v', magnitude:'abs_magn'})    
                
                if float(self.zmax) <  0.07296: 
                    raise ValueError('MiceCatv2 has no galaxies below 0.07296, please give a higher zmax')
                
                if float(self.zmax) < self.zmin:
                    raise ValueError('zmax is lower than zmin')
                    
                if float(self.zmin) < 0.07296:
                    print('zmin lower than minimum redshift of MiceCat, z = 0.07296. Setting zmin = 0.07296.') 
                    self.zmin =  0.07296  
                
                
                
                phis = np.array(data_frame['ra'])
                print("Raw catalog, #galaxies = {}".format(len(phis)))
                data_frame = data_frame[data_frame['z'] <= float(self.zmax)]
                data_frame = data_frame[data_frame['z'] >= float(self.zmin)]
                data_frame = data_frame[data_frame['ra'] >= 0.0]
                phis = np.array(data_frame['ra'])
                print("After cut on z: {} - {} and ra>0 #galaxies: {}".format(self.zmin,self.zmax,len(phis)))
                #print(data_frame.head(4))
                data_frame, fraction_cat = self.masking(data_frame, self.NSIDE, self.Nth)
                phis = np.array(data_frame['ra'])
                print("After masking, #galaxies = {}".format(len(phis)))
                self.fraction_cat = fraction_cat #Micecat 1/8 of Full sky 

                
                
                #Compute sigma_z 
                fraction =  self.z_err/100
                data_frame['sigma_z'] = fraction*data_frame.z
                
                #select only galaxies with app_mag lower than mth_cat
                if self.mth_cat != 0:
                    #print(data_frame)
                    # indicies = np.where(np.array(data_frame['app_magn']) < float(self.mth_cat))[0]
                    data_frame = data_frame[data_frame['app_magn'] < float(self.mth_cat)]
                    phis = np.array(data_frame['ra'])
                    print("After cut on app_magn < {}, #galaxies: {}".format(self.mth_cat,len(phis)))
                    
                    
                self.catalog_data = data_frame
                
                
            else:
                raise ValueError("Catalog file not supported")
        except:
            raise ValueError("Can't find the catalog file") 

            
    def masking(self, dataframe, NSIDE,Nth):
            print('Applying a mask to catalog with NSIDE = {} and Nth ={}'.format(NSIDE,Nth))
            #masking 
            #Spherical Angles
            phis = np.array(np.deg2rad(dataframe['ra']))
            thetas = np.pi/2 - np.array(np.deg2rad(dataframe['dec']))

            #Turn angles into pixel indicies
            pixel_indices = hp.ang2pix(NSIDE, thetas, phis, nest = False)

            #Tot number of pixels 
            Npix = hp.nside2npix(NSIDE)
            
            #count indiceis in bins, (galaxy count per pixel)
            bc = np.bincount(pixel_indices, minlength=Npix)


            #initialise zeros mask
            mask = np.zeros(Npix, dtype = int) #Blank healpix map

            #N galaxies Threshold 
            inx = np.where(bc >= Nth) 
            mask[inx] = 1
            print('Selecting galaxies from mask')
            indicies_ = np.where(mask == 1)[0]
            gal_id = np.where(np.in1d(pixel_indices, indicies_))[0]
            masked_cat = dataframe.iloc[gal_id,:] 
            
            #compute fraction 
            fraction_cat = float(len(inx[0])/len(mask))
            return masked_cat, fraction_cat
