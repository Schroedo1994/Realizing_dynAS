from ccmaesframework.configurablecmaes import ConfigurableCMAES
from algorithm import Algorithm
import numpy as np
from ccmaesframework.utils import _correct_bounds
import copy
from regression_warmstart import cma_es_warm_starting

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

        # Load parameters object
        #if 'Hessian' in parameters.internal_dict:
         #   self.configcmaes.parameters = self.configcmaes.parameters.load('parameters.pkl')

        # Initialize population
        if 'x_opt' in parameters.internal_dict:
            m = np.clip(parameters.internal_dict['x_opt'], self.func.lowerbound, self.func.upperbound)
            self.configcmaes.parameters.m = np.asarray(m)[:, np.newaxis]

        # Use BFGS's inverse Hessian matrix to initialize covariance matrix
        beta = 1 # scaling factor
        if 'invHessian' in parameters.internal_dict:
            self.configcmaes.parameters.C = beta * parameters.internal_dict['invHessian']
            self.configcmaes.parameters.perform_eigendecomposition()
            
            # Plot initial covariance matrix
            #self.configcmaes.parameters.lambda_ = 10000    

        # Initialize step size with bfgs_x_hist
        if 'bfgs_x_hist' in parameters.internal_dict:
            # Init sigma
            bfgs_x_hist = parameters.internal_dict['bfgs_x_hist']    # bfgs_x_hist only contains xk points of BFGS
            number_points = 2
            cumul_dist = 0
            for i in range(1, number_points+1):
                a = bfgs_x_hist[len(bfgs_x_hist)-i]
                b = bfgs_x_hist[len(bfgs_x_hist)-i-1]
                cumul_dist += np.linalg.norm(a-b)
            
            # average euclidean distance between last n points
            predicted_sigma = cumul_dist / number_points

            self.configcmaes.parameters.sigma = predicted_sigma
        
        if 'C' in parameters.internal_dict:
            self.configcmaes.parameters.C = parameters.internal_dict['C']
            print(f'C changed to {self.configcmaes.parameters.C}')
            self.configcmaes.parameters.sigma = parameters.internal_dict['stepsize']
            self.configcmaes.parameters.perform_eigendecomposition()
            #self.configcmaes.parameters.lambda_ = 10000
        
        """
        # Gaussian regression model to predict C and sigma
        if ('mlsl_x_hist' in parameters.internal_dict and 'mlsl_f_hist' in parameters.internal_dict):

            X = np.asarray(parameters.internal_dict['mlsl_x_hist'])
            y = np.asarray(parameters.internal_dict['mlsl_f_hist'])
            sigma0, inv_H = cma_es_warm_starting(X, y)
            self.configcmaes.parameters.C = inv_H 
            self.configcmaes.parameters.perform_eigendecomposition()
            self.configcmaes.parameters.sigma = sigma0
            
            # Plot initial covariance matrix
            #self.configcmaes.parameters.lambda_ = 10000
        """
        
        """
        # Hand-over saved C, m and sigma to plot covariance matrix and distribution
        if 'C' in parameters.internal_dict:
            self.configcmaes.parameters.C = parameters.internal_dict['C']
            self.configcmaes.parameters.perform_eigendecomposition()
            self.configcmaes.parameters.sigma = parameters.internal_dict['stepsize']
            self.configcmaes.parameters.m = parameters.internal_dict['m']
            
            # Plot initial covariance matrix
            #self.configcmaes.parameters.lambda_ = 10000

        """

        """
        # Initialize step size with clipping
        
        if 'stepsize' in parameters.internal_dict:
            if parameters.internal_dict['stepsize'] <= 0.01:
                self.configcmaes.parameters.sigma = parameters.internal_dict['stepsize']
            else:
                self.configcmaes.parameters.sigma = 0.01
        """

        # save number of evaluations and stepsizes to create plot
        if 'stepsizes' in parameters.internal_dict:
            self.configcmaes.stepsizes = parameters.internal_dict['stepsizes']
            
        if 'evalcount' in parameters.internal_dict:
            self.configcmaes.eval_count = parameters.internal_dict['evalcount']
            
        if 'matrix_norm' in parameters.internal_dict:
            self.configcmaes.matrix_norm = parameters.internal_dict['matrix_norm']


    def get_params(self, parameters):
        parameters.internal_dict['x_opt'] = self.func.best_so_far_variables
        parameters.internal_dict['stepsize'] = self.configcmaes.parameters.sigma
        parameters.internal_dict['C'] = self.configcmaes.parameters.C
        parameters.internal_dict['invC'] = self.configcmaes.parameters.invC
        parameters.internal_dict['m'] = self.configcmaes.parameters.m

        # save number of evaluations and stepsizes to create plot
        parameters.internal_dict['stepsizes'] = self.configcmaes.stepsizes
        parameters.internal_dict['evalcount'] = self.configcmaes.eval_count
        parameters.internal_dict['matrix_norm'] = self.configcmaes.matrix_norm
        parameters.internal_dict['cmaes_x_hist'] = self.configcmaes.x_hist
        parameters.internal_dict['cmaes_f_hist'] = self.configcmaes.f_hist
        parameters.internal_dict['cmaes_x_opt'] = self.func.best_so_far_variables
        parameters.internal_dict['cmaes_gen_counter'] = self.configcmaes.generation_counter
        parameters.internal_dict['evals_splitpoint'] = self.func.evaluations

        """
        # Save parameters object
        if 'C' in parameters.internal_dict:
            #self.configcmaes.parameters.save(filename = 'parameters-inst1-f8d2.pkl')
        """

        return parameters

    def run(self):
        print(f'CMA-ES started')
        #print(f'sigma: {self.configcmaes.parameters.sigma} C: {self.configcmaes.parameters.C}')

        self.configcmaes.parameters.budget = self.budget
        self.configcmaes.break_conditions = self.stop
        self.configcmaes.run()

        print(f'CMA-ES complete')
        print(f'evals: {self.func.evaluations} prec: {self.func.best_so_far_precision}')
        #print(f'C: {self.configcmaes.parameters.C}')

        return self.func.best_so_far_variables, self.func.best_so_far_fvalue
