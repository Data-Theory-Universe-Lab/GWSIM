import sys
from setuptools import setup

setup_requires = ['setuptools >= 30.3.0']
if {'pytest', 'test', 'ptr'}.intersection(sys.argv):
    setup_requires.append('pytest-runner')
if {'build_sphinx'}.intersection(sys.argv):
    setup_requires.extend(['recommonmark',
                           'sphinx'])

def readme():
    with open('README.md') as f:
        return f.read()

setup(name='GWSim',
      version='0.1.0',
      description='A package to create datasets of mock data challenges for gravitational-wave codes',
      url='https://git.ligo.org/benoit.revenu/gwsim',
      author=['Christos Karathanasis','Benoit Revenu', 'Suvodip Mukherjee', 'Federico Stachurski'],
      author_email=['ckarathanasis@ifae.es','revenu@in2p3.fr', 'suvodip@tifr.res.in', 'f.stachurski.1@research.gla.ac.uk'],
      license='GNU',
      packages=['GWSim', 'GWSim.universe', 'GWSim.catalog', 'GWSim.injections', 'GWSim.random'],
      package_dir={'GWSim': 'GWSim'},
      scripts=['bin/GW_create_universe', 'bin/GW_create_catalog', 'bin/GW_injections','bin/GW_create_posteriors'],
      include_package_data=True,
      install_requires=[
          'numpy>=1.9',
          'matplotlib>=2.0',
          'pandas',
          'scipy',
          'h5py',
          'lalsuite',
          'fitsio',
          'healpy',
          'multiprocess',
          'bilby'],
      setup_requires=setup_requires,
      zip_safe=False)
