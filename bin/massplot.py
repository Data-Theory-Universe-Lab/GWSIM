import os
import numpy as np
import math
from scipy import integrate
import matplotlib.pylab as plt
from optparse import Option, OptionParser, OptionGroup
import pickle


def mkplot(mf,out):
    file = open(mf,'rb')
    inj = pickle.load(file)
    nbins = 100
    fig, axs = plt.subplots(1,2,figsize=(15,6))
    cm1, bm1 = np.histogram(inj['m1s'],nbins)
    bc1 = (bm1[:-1] + bm1[1:]) / 2
    axs[0].plot(bc1,cm1,'.')
    axs[0].set(xlabel="m1, source frame [$M_{\odot}$]")
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].set_ylim([np.min(cm1[np.where(cm1>0)])/2,np.max(cm1)*2])
    norm = integrate.simpson(cm1,bc1)
    axs[0].plot(inj['masses'],norm*inj['m1s_eff'])
    cm2, bm2 = np.histogram(inj['m2s'],nbins)
    bc2 = (bm2[:-1] + bm2[1:]) / 2
    axs[1].plot(bc2,cm2,'.')
    axs[1].set(xlabel="m2, source frame [$M_{\odot}$]")
    axs[1].set_xscale('log')
    axs[1].set_yscale('log')
    axs[1].set_ylim([np.min(cm2[np.where(cm2>0)])/2,np.max(cm2)*2])
    norm = integrate.simpson(cm2,bc2)
    axs[1].plot(inj['masses'],norm*inj['m2s_eff'])
    plt.savefig(out+'.png')
#    plt.show()

parser = OptionParser(
    description = __doc__,
    usage = "%prog [options]",
    option_list = [
        Option("--file", default='', type=str,
               help="file path (..._masses.p)"),
        Option("--out", default='fig', type=str,
               help="output file name (default=input file +.png)")
        ]
    )
opts, args = parser.parse_args()
print(opts)

mkplot(opts.file,opts.out)
