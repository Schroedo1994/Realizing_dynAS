import scipy_differentialevolution
import numpy as np
from algorithm import Algorithm
import numpy as np
import copy
from scipy.optimize import Bounds
from scipy._lib._util import check_random_state

class DE(Algorithm):
    """DE algorithm
    """
    __doc__ += Algorithm.__doc__
    
    def __init__(self, func):
        Algorithm.__init__(self, func)
        bounds = Bounds(self.func.lowerbound, self.func.upperbound)
        self.de_wrapper = scipy_differentialevolution.DifferentialEvolutionSolver(self.func, bounds = bounds)
        self.random_number_generator = check_random_state(seed=None)

    def set_params(self, parameters):
        self.budget = parameters.budget
        
        # Warmstarting
        # Warm-start population around x_opt
  
        if 'x_opt' in parameters.internal_dict:
            x_opt = parameters.internal_dict['x_opt']
            eta = 0.1
            scale_arg = 10   # important for scaling variables to interval between 0 and 1
            rng = self.random_number_generator
            
            for i in range(0, self.de_wrapper.num_population_members):
                for j in range(0, self.func.number_of_variables):
                    self.de_wrapper.population[i][j] = (x_opt[j] + rng.uniform(low=-eta, high=eta)) / scale_arg + 0.5

            # reset population energies
            self.de_wrapper.population_energies = np.full(self.de_wrapper.num_population_members,
                                           np.inf)
            
            #print(f'init pop: {self.de_wrapper.population}')

        
    def get_params(self, parameters):
        
        parameters.internal_dict['x_opt'] = self.func.best_so_far_variables
        parameters.internal_dict['de_x_opt'] = self.func.best_so_far_variables
        parameters.internal_dict['de_x_hist'] = self.de_wrapper.a2_x_hist
        parameters.internal_dict['de_f_hist'] = self.de_wrapper.f_hist
        parameters.internal_dict['de_gen_counter'] = self.de_wrapper.generation_counter

        return parameters
    
    def run(self):
        print(f'DE started')

        self.de_wrapper.maxfun = self.budget
        self.de_wrapper.stop = self.stop
        self.de_wrapper.solve()
        

        print(f'DE complete')
        print(f'evals: {self.func.evaluations} prec: {self.func.best_so_far_precision}')

        return self.func.best_so_far_variables, self.func.best_so_far_fvalue


