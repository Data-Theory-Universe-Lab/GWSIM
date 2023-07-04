# GWSim

July 2023.

GWSim is a python code that allows to generate fake gravitational waves (GW) events corresponding to binary black holes (BBHs) systems. First you have to either create a universe or use a galaxy catalog. Some galaxies will be chosen to host GW events, following the merger rate and population model you are using. The universe created by the code is uniform in comobile volume. If you have suggestions or find bugs please contact us: <revenu@in2p3.fr>.

All formulas used in the code are described in the [paper](<https://arxiv.org/abs/2210.05724>), accepted for publication in Astronomy \& Astrophysics.

# How-to install
* Clone the gwcosmo repository with

  `
  git clone https://git.ligo.org/benoit.revenu/gwsim.git
  `

* Set up a conda environment

  `
  conda create -n gwsim python=3.9
  `
  
* Activate your conda environment

  `
  conda activate gwsim
  `

* Enter the cloned gwsim directory

  `cd gwsim`

* Then install gwsim by running (must be executed in the directory where the file `setup.py` is located)

    `
    python -m pip install .
    `

# Typical usage

## Creation of a fake universe

Create a fake universe, uniform in comobile volume, $\Lambda\text{CDM}$ model, with density of galaxies of $10^{-5}~\text{Mpc}^{-3}$, between redshifts $0$ and $5$, with a Hubble constant of $H_0=80~\text{km s}^{-1}\text{Mpc}^{-1}$. The parameters of the Schechter function can be also modified and the default values correspond to $H_0=100~\text{km s}^{-1}\text{Mpc}^{-1}$.

```
./bin/GW_create_universe --zmin 0 --zmax 5 --H0 80 --w0 -1 --log_n -5
```

## Creation of mergers

Then you have to pick up some galaxies, chosen to be mergers hosts. The GW events are randomly generated following the merger rate you ask and the population you want to simulate. The population is described by the masses of the black holes ($m_1$ is the heaviest mass of both and $m_2$ is the mass of the lightest object): you have to choose the probability density function for $m_1: p(m_1)$ that can be a powerlaw, a powerlaw+gaussian peak etc (see the [paper](<https://arxiv.org/abs/2210.05724>)).

```
./bin/GW_injections \
--file Universe.p # file containing the simulated universe
--luminosity_weight 1 # more luminous galaxies have more mergers
--redshift_weight 1 # the merger rate depends on the redshift (Madau-Dickinson law)
--population_model powerlaw-gaussian # probability density function for m1, p(m1)
--alpha 3.4 --mu 35 --sigma 3.88 --Lambda 0.04 --delta_m 4.8 # parameters for p(m1)
--beta 0.8 # parameter for m2, the pdf is a powerlaw with spectral index beta
--snr 10 # SNR threshold to detect a GW event
--output my_GW_injections # output filename
--Madau_alpha 2.7 --Madau_beta 2.9 --Madau_zp 1.9 # Madau-Dickinson law parameters
--T_obs 3 # observation time in years
--R0 20 # merger rate at z=0, in Gpc^{-3} per year
--seed 294757 # value of the random seed
-—npools 8 # number of CPUs for parallel computation
```
*CAREFUL*: for a value `--alpha x` provided in the commandline, it corresponds to a powerlaw with spectral index `-x`!

The details of the command line arguments can always be obtained with the flag `--help`.

After some time, we obtain that there have been 140418 mergers in the universe (between z=0 and z=5) and during the observation time specified. This number can vary as we use a poissonian law to draw it. Then the SNR is computed for all mergers, with a random choice among the interferometers (LIGO/Virgo), according to their duty cycles and according to the user's choice. After this step, you obtain some files:

```
my_GW_injections_galaxies.p
my_GW_injections_masses.p
my_GW_injections_spins.p
my_GW_injections.p
```

The `my_GW_injections_galaxies.p` file contains all parameters of the galaxies selected as BBH mergers hosts (redshit, distance, magnitude, sky coordinates...). This file allows to avoid re-selecting merger hosts in case you want to re-run the injections.

The `my_GW_injections_masses.p` file contains all mass values of the individual mergers: m1, m2 but also the global effective mass distribution.

`my_GW_injections_spins.p` contains the spin values of all mergers.

And finally `my_GW_injections.p` contains all information about selected mergers (i.e. having a SNR larger than the requested threshold). This information must be provided to the code `GW_create_posteriors` to generate the posteriors, corresponding to the final goal of `gwsim`.

## Creation of the posteriors

This is a quite long process. We usually have to use a cluster and the computation for $\sim 200$ events can take days/weeks of computation, depending on the computing power you have access to. `gwsim` is designed to work with the condor job management system. The final result is the posterior for all selected events, in `hdf5` files. Each selected event signal is computed, mixed with realistic noise from the triggered interferometers and the parameter estimation (PE) is done with `Bilby`. The typical command line is:

```
./bin/GW_create_posteriors --file my_GW_injections.p --npools 20 --output fake_GW_events
```

The final product is in the `hdf5` files, where the estimated parameters of the GW sources can be used for many analyses: population studies, cosmology...
