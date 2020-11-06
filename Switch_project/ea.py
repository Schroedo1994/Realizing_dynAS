# Imports
import numpy as np
import random as rd
import math
from algorithm import Algorithm

# setting up numpy random
np.random.seed()
generator = np.random.default_rng()

class EA(Algorithm):
    """ A simple evolutionary algorithm.

    Parameters:
    -----------------
    func : object, callable
        The function to be optimized.

    Attributes:
    -----------------

    lamda : int
        The number of offspring individuals per iteration.

    pc : float
        The cross-over probability, must be 0>= pc >=1

    pop : array-like
        Matrix holding the current population.

    sigma : array-like, float
        Array containing the individual step-sizes.

    global_tau : float
        Constant for step size update in individual stepsize mutation.

    local_tau : float
        Constant for step size update in individual stepsize mutation. 
        
    Methods:
    --------------
    
    set_params(parameters):
        Sets algorithm parameters for warm-start.
        
    get_params(parameters):
        Transfers internal parameters to parameters dictionary.
        
    run():
        Runs the EA algorithm.

    Parent:
    ------------

    """
    __doc__ += Algorithm.__doc__

    def __init__(self, func):
        Algorithm.__init__(self, func)
        self.lamda = 100
        self.pc = 0.75
        self.pop = None
        self.f = []
        self.sigma = None
        self.global_tau = 1.0
        self.local_tau = 1.0
        
    
    def set_params(self, parameters):
        self.budget = parameters.budget

        """Warm start routine"""


    def get_params(self, parameters):
        parameters.internal_dict['sigma_vec'] = self.sigma
        parameters.internal_dict['x_opt'] = self.func.best_so_far_variables

        return parameters


    def initialize_population(self):
        """ Creates a random population.

        Parameters:
        -------------
        None

        Returns:
        -------------
        population : array-like
            An array of mu individuals

        """
        population = []
        for i in range(0, self.popsize):
            new_ind = np.zeros(self.dim)
            for j in range(0, self.dim):
                new_ind[j] = generator.uniform(low=-5, high=5)
            population.append(new_ind)
        
        return population

    def global_discrete_recombine(self):
        offspring = np.zeros((self.dim))
        for i in range(0, self.dim):
            random_parent = self.pop[rd.randrange(0, len(self.pop))]
            offspring[i] = random_parent[i]

        return offspring

    def update_sigma(self):
        updated_sigma = np.zeros(self.dim)
        for i in range(0, self.dim):
            updated_sigma[i] = self.sigma[i] * math.exp(
                self.global_tau * np.random.normal(0, 1)
                + self.local_tau * np.random.normal(0, 1))

        return updated_sigma

    def mutate(self, ind):
        mutant = np.zeros(len(ind))
        for j in range(0, len(ind)):
            mutant[j] = ind[j] + self.sigma[j] * np.random.normal(0, 1)
        mutant = np.clip(mutant, self.func.lowerbound, self.func.upperbound)

        return mutant

    def run(self):
        """ Runs the evolutionary algorithm.

        Parameters:
        --------------
        None

        Returns:
        --------------
        best_so_far_variables : array
                The best found solution.

        best_so_far_fvaluet: float
               The fitness value for the best found solution.

        """
        print(f' EA started')
        
        # Initialize parameters depending on problem
        self.popsize = 15
        self.global_tau = 1 / math.sqrt(2*self.dim)
        self.local_tau = 1 / math.sqrt(2*math.sqrt(self.dim))

        # Initialization
        self.pop = self.initialize_population()    # parent population
        xo = []    # offspring population
        fo = []   # offspring fitness
        self.sigma = np.zeros(self.dim)
        for s in range(0, self.dim):
            self.sigma[s] = 10 / 6

        # Evaluate initial parent population
        for i in range(0, self.popsize):
            self.f.append(self.func(self.pop[i]))

        # Evaluation loop
        while not self.stop():
            self.sigma = self.update_sigma()

            # Create new offspring generation
            for i in range(0, self.lamda):
                offspring = self.global_discrete_recombine()
                xo.append(self.mutate(offspring))
                fo.append(self.func(xo[i]))

            # Create joint matrix of individuals and their fitness values
            m = np.hstack((np.asarray(xo), np.expand_dims(np.asarray(fo), axis=1)))

            # Sort based on fitness (lowest values highest ranking)
            sorted_m = m[np.argsort(m[:, self.dim])]

            # Copy best individuals and remove fitness values
            xpnew = sorted_m[0:self.popsize, 0:self.dim]

            # Set new population for next generation
            self.pop = []
            for i in range (0, len(xpnew)):
                self.pop.append(xpnew[i])

            # Copy fitness values of new generation
            self.f = []
            mf = sorted_m[0:self.popsize, self.dim]
            for i in range (0, len(mf)):
                self.f.append(mf[i])
        
        print(f' EA complete')

        return self.func.best_so_far_variables, self.func.best_so_far_fvalue