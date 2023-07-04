import numpy as np
from GWSim.injections.distributions import Spin

class Spins(object):

    def __init__(self,pop_parameters,redshifts):

        self.pop_parameters = pop_parameters
        self.z = redshifts

    def sample(self):
        print('Sampling spins for {} mergers with the {} model.'.format(self.pop_parameters['N'],self.pop_parameters['spin_model']))
        spins_distribution = Spin(self.pop_parameters,self.z)
        self.pop_parameters['chi_1'], self.pop_parameters['chi_2'], self.pop_parameters['theta_1'], self.pop_parameters['theta_2'] = spins_distribution.sample(self.pop_parameters['N'])
