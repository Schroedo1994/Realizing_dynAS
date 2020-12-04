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
        
        """
        # Use BFGS's Hessian matrix to initialize covariance matrix
        beta = 1e-1 # scaling factor
        if 'Hessian' in parameters.internal_dict:
            self.configcmaes.parameters.C = beta * parameters.internal_dict['Hessian']
            self.configcmaes.parameters.perform_eigendecomposition()
            
            # Plot initial covariance matrix
            #self.configcmaes.parameters.lambda_ = 10000    

        # Initialize step size with a1_x_hist
        if 'a1_x_hist' in parameters.internal_dict:
            # Init sigma
            a1_xhist = parameters.internal_dict['a1_x_hist']    # a1_x_hist only contains xk points of BFGS
            number_points = 2
            cumul_dist = 0
            for i in range(1, number_points+1):
                a = a1_xhist[len(a1_xhist)-i]
                b = a1_xhist[len(a1_xhist)-i-1]
                cumul_dist += np.linalg.norm(a-b)
            
            # average euclidean distance between last n points
            predicted_sigma = cumul_dist / number_points

            self.configcmaes.parameters.sigma = predicted_sigma
        """

        
        # Gaussian regression model to predict C and sigma
        if ('a1_x_hist' in parameters.internal_dict and 'a1_f_hist' in parameters.internal_dict):
            ub = len(np.asarray(parameters.internal_dict['x_hist']))
            lb = ub - int(1 * ub)

            X = np.asarray(parameters.internal_dict['a1_x_hist'])[lb:ub]
            y = np.asarray(parameters.internal_dict['a1_f_hist'])[lb:ub]
            sigma0, inv_H = cma_es_warm_starting(X, y)
            #print(f'inv_H: {inv_H}')
            self.configcmaes.parameters.C = inv_H 
            self.configcmaes.parameters.perform_eigendecomposition()
            self.configcmaes.parameters.sigma = sigma0
            #self.configcmaes.parameters.lambda_ = 10000
        

        """   
        # Hand-over previous C, m and sigma to plot covariance matrix and distribution
        if 'C' in parameters.internal_dict:
            self.configcmaes.parameters.C = parameters.internal_dict['C']
            self.configcmaes.parameters.perform_eigendecomposition()
            self.configcmaes.parameters.sigma = parameters.internal_dict['stepsize']
            self.configcmaes.parameters.m = parameters.internal_dict['m']
            self.configcmaes.parameters.lambda_ = 10000
        """

        """
            # Init ps, pc and dm
            old_m = a1_xhist[len(a1_xhist)-2]
            self.configcmaes.parameters.dm = (self.configcmaes.parameters.m - old_m) / self.configcmaes.parameters.sigma
            
            cparams =  self.configcmaes.parameters
            self.configcmaes.parameters.ps = (np.sqrt(cparams.cs * (2 - cparams.cs)-cparams.mueff) * 
                                             cparams.invC @ cparams.dm) * cparams.ps_factor
            
            hs = (np.linalg.norm(self.configcmaes.parameters.ps) /
                np.sqrt(1 - np.power(1 - cparams.cs, 2 *
                                 (self.func.evaluations / cparams.lambda_)))
                        ) < (1.4 + (2 / (self.dim + 1))) * cparams.chiN
            
            self.configcmaes.parameters.pc = (hs * np.sqrt(cparams.cc * (2 - cparams.cc) * cparams.mueff )) * cparams.dm
            
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
        parameters.internal_dict['m'] = self.configcmaes.parameters.m

        # save number of evaluations and stepsizes to create plot
        parameters.internal_dict['stepsizes'] = self.configcmaes.stepsizes
        parameters.internal_dict['evalcount'] = self.configcmaes.eval_count
        parameters.internal_dict['matrix_norm'] = self.configcmaes.matrix_norm
        parameters.internal_dict['cmaes_x_hist'] = self.configcmaes.x_hist
        parameters.internal_dict['cmaes_f_hist'] = self.configcmaes.f_hist
        parameters.internal_dict['cmaes_x_opt'] = self.func.best_so_far_variables

        """
        # Save parameters object
        if 'C' in parameters.internal_dict:
            #self.configcmaes.parameters.save(filename = 'parameters-inst1-f8d2.pkl')
        """

        return parameters

    def run(self):
        print(f'CMA-ES started')

        # Implementation validation prints
        #print(f'init_sigma: {self.configcmaes.parameters.sigma}')
        #print(f'init C: {self.configcmaes.parameters.C}')

        self.configcmaes.parameters.budget = self.budget
        self.configcmaes.break_conditions = self.stop
        self.configcmaes.run()

        print(f'CMA-ES complete')
        print(f'x_opt: {self.func.best_so_far_variables}')

        return self.func.best_so_far_variables, self.func.best_so_far_fvalue
