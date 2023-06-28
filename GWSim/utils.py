import numpy as np
from scipy.interpolate import interp1d
import numpy.random as rand

def Sampling(parameters):

    try:
        conditional = parameters['conditional']
        samples, conditional_reshaped = Conditional_sampling(parameters)
        return samples, conditional_reshaped
    except:
        i=0
        N, maximum, minimum = parameters['N'], parameters['maximum'], parameters['minimum']
        interpol = parameters['interpolation']
        samples = np.zeros(N)
        x = np.linspace(minimum,maximum,1000)
        function_values_max = np.max(interpol(x))
        function_values_min = 0

        while i<len(samples):
            dim = len(samples)-i
            points = rand.rand(dim)*(maximum-minimum)+minimum
            function_values = rand.rand(dim)*(function_values_max-function_values_min)+function_values_min

            passing_the_criteria = np.where(function_values<=interpol(points))
            samples[i:i+len(passing_the_criteria[0])] = points[passing_the_criteria]

            i+= len(passing_the_criteria[0])
        idx = rand.permutation(len(samples))
        return samples[idx]

def Conditional_sampling(parameters):

    N, maximum, minimum = parameters['N'], parameters['maximum'], parameters['minimum']
    interpol = parameters['interpolation']
    conditional = parameters['conditional']
    samples = np.zeros(N)
    conditional_reshaped = np.zeros(N)
    i=0
    x = np.linspace(minimum,maximum,1000)
    function_values_max = np.max(interpol(x))
    function_values_min = 0

    while i<len(samples):
        dim = len(samples)-i
        temp = (maximum+1)*np.ones(dim)
        points = rand.rand(dim)*(maximum-minimum)+minimum
        function_values = rand.rand(dim)*(function_values_max-function_values_min)+function_values_min

        passing_the_criteria = np.where(function_values<=interpol(points))
        temp[passing_the_criteria] = points[passing_the_criteria]

        passing_the_condition = np.where(temp<=conditional)
        samples[i:i+len(passing_the_condition[0])] = temp[passing_the_condition]
        conditional_reshaped[i:i+len(passing_the_condition[0])] = conditional[passing_the_condition]
        conditional = np.delete(conditional,passing_the_condition[0])
        i+= len(passing_the_condition[0])
    idx = rand.permutation(len(samples))
    return samples[idx],conditional_reshaped[idx]

def Interpolate_function(parameters):

    maximum, minimum = parameters['maximum'], parameters['minimum']
    function_values = parameters['distribution']
    x = np.linspace(minimum,maximum,len(function_values))
    return interp1d(x,function_values)
'''
def Integrate(minimum,maximum,N,function):

    z_grid = np.linspace(minimum,maximum,N).T
    dz = z_grid[:,1:]-z_grid[:,:-1]
    values = function(z_grid)
    return np.sum(0.5*(values[:,:-1]+values[:,1:])*dz,axis=1)
'''
def Integrate_1d(minimum,maximum,N,function):

    z_grid = np.linspace(minimum,maximum,N)
    dz = z_grid[1]-z_grid[0]
    values = function(z_grid)
    return np.sum(0.5*(values[:-1]+values[1:])*dz)


class Rejection_Sampling(object):

    def __init__(self,parameters):
        self.parameters = parameters

    def Sample(self):

        try:
            conditional = self.parameters['conditional']
            samples, conditional_reshaped = Conditional_sampling(self.parameters)
            return samples, conditional_reshaped
        except:
            i=0
            N, maximum, minimum = self.parameters['N'], self.parameters['maximum'], self.parameters['minimum']
            interpol = self.parameters['interpolation']
            samples = np.zeros(N)
            x = np.linspace(minimum,maximum,1000)
            function_values_max = np.max(interpol(x))
            function_values_min = 0

            while i<len(samples):
                dim = len(samples)-i
                points = rand.rand(dim)*(maximum-minimum)+minimum
                function_values = rand.rand(dim)*(function_values_max-function_values_min)+function_values_min

                passing_the_criteria = np.where(function_values<=interpol(points))
                samples[i:i+len(passing_the_criteria[0])] = points[passing_the_criteria]

                i+= len(passing_the_criteria[0])
            idx = rand.permutation(len(samples))
            return samples[idx]

    def Conditional_sampling(self):

        N, maximum, minimum = self.parameters['N'], self.parameters['maximum'], self.parameters['minimum']
        interpol = self.parameters['interpolation']
        conditional = self.parameters['conditional']
        samples = np.zeros(N)
        conditional_reshaped = np.zeros(N)
        i=0
        x = np.linspace(minimum,maximum,1000)
        function_values_max = np.max(interpol(x))
        function_values_min = 0

        while i<len(samples):
            dim = len(samples)-i
            temp = (maximum+1)*np.ones(dim)
            points = rand.rand(dim)*(maximum-minimum)+minimum
            function_values = rand.rand(dim)*(function_values_max-function_values_min)+function_values_min

            passing_the_criteria = np.where(function_values<=interpol(points))
            temp[passing_the_criteria] = points[passing_the_criteria]

            passing_the_condition = np.where(temp<=conditional)
            samples[i:i+len(passing_the_condition[0])] = temp[passing_the_condition]
            conditional_reshaped[i:i+len(passing_the_condition[0])] = conditional[passing_the_condition]
            conditional = np.delete(conditional,passing_the_condition[0])
            i+= len(passing_the_condition[0])
        idx = rand.permutation(len(samples))
        return samples[idx],conditional_reshaped[idx]


class Inverse_Cumulative_Sampling(object):

    def __init__(self,parameters):

        self.parameters = parameters

    def Sample(self):

        cdf = np.cumsum(self.parameters['distribution'])
        cdf /= np.max(cdf)
        cdf[0] = 0 # start from 0
        icdf = interp1d(cdf,np.linspace(self.parameters['minimum'],self.parameters['maximum'],len(cdf)))
        return icdf(rand.uniform(0,1,self.parameters['N']))
