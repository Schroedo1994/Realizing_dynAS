# imports
import numpy as np
import random as rd
import math
import shutil
import datetime
from scipy.special import gamma
from scipy.optimize import minimize, Bounds

from algorithm import Algorithm

# setting up numpy random
np.random.seed()
generator = np.random.default_rng()

class MLSL(Algorithm):
    """ Multi-level single linkage algorithm.

    Parameters:
    -----------------
    func : object, callable
        The function to be optimized.

    Attributes:
    --------------

    pop : array
        Matrix holding all current solution candidates or points.

    gamma: float
        Factor determining the size of the reduced sample.

    k : int
        The current iteration number.

    zeta : float
        The scaling factor for updating the critical distance.

    xr : array
        Matrix holding the reduced sample points.

    fr : array
        Array holding the fitness values for points in the reduced sample.

    rk : float
        The critical distance rk.

    lebesgue : float
        The Lebesgue measure of distance.
        
    Methods:
    --------------
    
    set_params(parameters):
        Sets algorithm parameters for warm-start.
        
    get_params(parameters):
        Transfers internal parameters to parameters dictionary.
        
    run():
        Runs the MLSL algorithm.
        
    Parent:
    ------------

    """
    __doc__ += Algorithm.__doc__

    def __init__(self, func):
        Algorithm.__init__(self, func)
        self.pop = None
        self.f = []
        self.gamma = 0.1
        self.k = 1
        self.zeta = 2.0
        self.xr = None
        self.fr = None
        self.rk = 0
        self.lebesgue = math.sqrt(100 * self.dim)
        

    def set_params(self, parameters):
        self.budget = parameters.budget
        
        """Warm start routine"""
        
        if 'pop' in parameters.internal_dict:
            self.pop = parameters.internal_dict['pop']
        else:
            self.pop = []


    def get_params(self, parameters):
        parameters.internal_dict['rk'] = self.rk
        parameters.internal_dict['iteration'] = self.k
        parameters.internal_dict['x_opt'] = self.func.best_so_far_variables

        return parameters


    def calc_rk(self):
        """ Calculates the critical distance depending on current iteration and population.

        Parameters:
        -------------
        None

        Returns:
        -------------
        rk : float
             The critical distance rk

        """
        kN = self.k * len(self.pop)
        rk = (1 / math.sqrt(np.pi)) * math.pow((gamma(1 + (self.dim / 2))
                                    * self.lebesgue * (self.zeta * math.log1p(kN)) / kN), (1 / self.dim))

        return rk

    def run(self):
        """ Run the MLSL algorithm.

        Parameters:
        ------------
        None

        Returns:
        ------------
        best_so_far_variables : array
                The best found solution.

        best_so_far_fvaluet: float
               The fitness value for the best found solution.

        """

        print(f' MLSL started')

        # Set parameters depending on function characteristics
        local_budget = 0.1 * (self.budget - self.func.evaluations)        
        bounds = Bounds(self.func.lowerbound, self.func.upperbound)
        self.popsize = 50 * self.dim    # 50 points according to original BBOB submission

        # Initialize reduced sample and (re)set iteration counter to 1
        x_star = []
        f_star = []
        self.k = 1

        # Start iteration
        while not self.stop():

            # Sample new points
            for i in range(0, self.popsize):
                new_point = np.zeros(self.dim)
                for j in range(0, self.dim):
                    new_point[j] = generator.uniform(low=-5, high=5)
                self.pop.append(new_point)
                self.f.append(self.func(new_point))


            # Extract reduced sample xr
            self.xr = np.zeros((math.ceil(self.gamma * self.k * self.popsize), self.dim))
            m = np.hstack((np.asarray(self.pop), np.expand_dims(np.asarray(self.f), axis=1)))
            sorted_m = m[np.argsort(m[:, self.dim])]
            self.xr = sorted_m[0:len(self.xr), 0:self.dim]
            self.fr = sorted_m[0:len(self.xr), self.dim]
 
            # Update rk
            self.rk = self.calc_rk()

            # Check critical distance and fitness differences in xr
            for i in range(0, len(self.xr)):
                cond = False
                for j in range(0, len(self.xr)):
                    if j == i:
                        continue
                    if self.fr[j] < self.fr[i]:
                        cond = np.linalg.norm(self.xr[j] - self.xr[i]) < self.rk
                    if cond:
                        break

                # If there is no point with better fitness in critical distance, start local search
                if not cond:
                    solution = minimize(self.func, self.xr[i], method='Powell', bounds=bounds,
                                        options={'ftol': 1e-8, 'maxfev': local_budget})
                    x_star.append(solution.x)
                    f_star.append(solution.fun)

                    local_budget = local_budget - solution.nfev
                    if local_budget < 0:
                        local_budget = 0

            self.k = self.k+1

        print(f' MLSL complete')

        return self.func.best_so_far_variables, self.func.best_so_far_fvalue