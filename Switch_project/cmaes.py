from ccmaesframework.configurablecmaes import ConfigurableCMAES
from algorithm import Algorithm
import numpy as np

class CMAES(Algorithm):
    """ Wrapper class for the configurable CMA-ES framework.
    
    Parameters:
    --------------
    None
    
    Attributes:
    --------------
    configcmaes : object-type
        An object of type ConfigurableCMAES.
    
    Methods:
    --------------
    
    set_params(parameters):
        Sets algorithm parameters for warm-start.
        
    get_params(parameters):
        Transfers internal parameters to parameters dictionary.
        
    run():
        Runs the CMA-ES algorithm.

    Parent:
    ------------

    """
    __doc__ += Algorithm.__doc__

    
    def __init__(self, func):
        Algorithm.__init__(self, func)
        self.configcmaes = ConfigurableCMAES(self.func, self.dim)
        
    
    def set_params(self, parameters):
        self.budget = parameters.budget
        
        """ Warm start routine """
        
        # Initialize step size
        
        if 'stepsize' in parameters.internal_dict:
            self.configcmaes.parameters.sigma = parameters.internal_dict['stepsize']
        
        # Initialize Covariance matrix
        
        # Initialize population
        
        if 'x_opt' in parameters.internal_dict:
            self.configcmaes.parameters.m = np.asarray(parameters.internal_dict['x_opt'])[:, np.newaxis]
        
    def get_params(self, parameters):
        parameters.internal_dict['x_opt'] = self.func.best_so_far_variables
        parameters.internal_dict['stepsize'] = self.configcmaes.parameters.sigma
        parameters.internal_dict['C'] = self.configcmaes.parameters.C
        
        return parameters

    def run(self):
        print(f'CMA-ES started')

        self.configcmaes.parameters.budget = self.budget
        self.configcmaes.break_conditions = self.stop
        self.configcmaes.run()

        print(f'CMA-ES complete')

        return self.func.best_so_far_variables, self.func.best_so_far_fvalue

