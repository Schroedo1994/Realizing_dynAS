# imports
import numpy as np
import random as rd
import math
import shutil
import datetime
from scipy.special import gamma
from scipy.optimize import minimize, Bounds

# setting up numpy random
np.random.seed()
generator = np.random.default_rng()

class MLSL():
    """ Multi-level single linkage algorithm.

    Parameters:
    --------------
    budget : int
             The budget for function evaluations.

    Attributes:
    --------------
    d : int
        Number of problem variables, dimensionality.

    x_opt : array
            The best found solution.

    f_opt : float
            Function value for the best found solution.

    pop : array
          Matrix holding all current solution candidates or points.

    n_points : int
               The number of new solutions per iteration.

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

    """

    def __init__(self, budget, order):
        self.budget = budget
        self.d = 1
        self.x_opt = None
        self.f_opt = np.inf
        self.pop = None
        self.n_points = 1
        self.gamma = 0.1
        self.k = 1
        self.zeta = 2.0
        self.xr = None
        self.fr = None
        self.rk = 0
        self.lebesgue = 0
        self.order = order

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
        rk = (1 / math.sqrt(np.pi)) * math.pow((gamma(1 + (self.d / 2))
                                    * self.lebesgue * (self.zeta * math.log1p(kN)) / kN), (1 / self.d))

        return rk

    def __call__(self, func, tau, target):
        """ Run the MLSL algorithm.

        Parameters:
        ------------
        func : object
               Function to be optimized.

        Returns:
        ------------
        x_opt : array
                The best found solution.

        f_opt: float
               The fitness value for the best found solution.

        """

        # Set parameters depending on function characteristics
        self.d = func.number_of_variables
        eval_budget = self.budget * self.d
        local_budget = 0.1 * eval_budget     # factor 0.1 according to original BBOB submission
        bounds = Bounds(func.lowerbound, func.upperbound)
        self.n_points = 50 * self.d    # 50 points according to original BBOB submission
        self.lebesgue = math.sqrt(100 * self.d)

        # Initialize reduced sample and (re)set iteration counter to 1
        self.pop = []
        f = []
        x_star = []
        f_star = []
        self.k = 1

        # Start iteration
        while func.evaluations < eval_budget and not func.final_target_hit:

            # Sample new points
            for i in range(0, self.n_points):
                new_point = np.zeros(self.d)
                for j in range(0, self.d):
                    new_point[j] = generator.uniform(low=-5, high=5)
                self.pop.append(new_point)
                f.append(func(new_point))
                
            print(f' Init-MLSL prec: {func.best_so_far_precision} evals: {func.evaluations}')

            # Extract reduced sample xr
            self.xr = np.zeros((math.ceil(self.gamma * self.k * self.n_points), self.d))
            m = np.hstack((np.asarray(self.pop), np.expand_dims(np.asarray(f), axis=1)))
            sorted_m = m[np.argsort(m[:, self.d])]
            self.xr = sorted_m[0:len(self.xr), 0:self.d]
            self.fr = sorted_m[0:len(self.xr), self.d]

            # Update rk
            self.rk = self.calc_rk()
            print(f' rk update, new rk: {self.rk}')

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
                    solution = minimize(func, self.xr[i], method='Powell', bounds=bounds,
                                        options={'ftol': 1e-8, 'maxfev': local_budget})
                    x_star.append(solution.x)
                    f_star.append(solution.fun)
                    local_budget = local_budget - solution.nfev
                    if local_budget < 0:
                        local_budget = 0
                
                print(f' For-MLSL prec: {func.best_so_far_precision} evals: {func.evaluations}')
                if self.order == 'a1' and (func.best_so_far_precision <= tau):
                    break
                if self.order == 'a2' and (func.best_so_far_precision <= target):
                    break

            print(f' MLSL prec: {func.best_so_far_precision} evals: {func.evaluations}')

            if self.order == 'a1' and (func.best_so_far_precision <= tau):
                break
            if self.order == 'a2' and (func.best_so_far_precision <= target):
                break

            self.k = self.k+1

        # Get best point from x_star
        n = np.hstack((np.asarray(x_star), np.expand_dims(np.asarray(f_star), axis=1)))
        sorted_n = n[np.argsort(n[:, self.d])]
        self.x_opt = sorted_n[0, 0:self.d]
        self.f_opt = sorted_n[0, self.d]

        print(f' MLSL complete, x_opt: {self.x_opt} eval: {func.evaluations}')
        return self.x_opt, self.f_opt
