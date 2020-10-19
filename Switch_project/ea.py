import numpy as np
import random as rd
import math
import IOHexperimenter as IOH
from IOHexperimenter import IOH_function
from shutil import make_archive
import datetime
np.random.seed()

class EA:
    """ A simple evolutionary algorithm object.

    Parameters:
    -----------------
    budget: int
        The budget for function evaluations.

    Attributes:
    -----------------
    d : int
        The number of dimensions, problem variables.

    mu : int
        The population size, number of parents.

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

    tau : float
         Constant for step size update in single stepsize mutation.

    """

    def __init__(self, budget, order):
        self.budget = budget
        self.d = 1
        self.mu = 15
        self.lamda = 100
        self.pc = 0.75
        self.pop = None
        self.sigma = None
        self.global_tau = 1.0
        self.local_tau = 1.0
        self.tau = 1.0
        self.order = order

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
        population = np.zeros((self.mu, self.d))
        for i in range(0, self.d):
            for k in range(0, self.mu):
                population[k][i] = rd.uniform(-5, 5)
            return population

    def global_discrete_recombine(self):
        offspring = np.zeros((self.d))
        for i in range(0, self.d):
            random_parent = self.pop[rd.randrange(0, len(self.pop))]
            offspring[i] = random_parent[i]

        return offspring

    def update_sigma(self):
        new_sigma = self.sigma * math.exp(self.tau * np.random.normal(0, 1))
        return new_sigma

    def singlestep_mutate(self, ind, func):
        mutant = np.zeros(len(ind))
        for j in range(0, len(ind)):
            mutant[j] = ind[j] + self.sigma * np.random.normal(0, 1)

        mutant = np.clip(mutant, func.lowerbound, func.upperbound)

        return mutant

    def update_sigma_vec(self):
        updated_sigma = np.zeros(self.d)
        for i in range(0, self.d):
            updated_sigma[i] = self.sigma[i] * math.exp(
                self.global_tau * np.random.normal(0, 1)
                + self.local_tau * np.random.normal(0, 1))

        return updated_sigma

    def indstep_mutate(self, ind, func):
        mutant = np.zeros(len(ind))
        for j in range(0, len(ind)):
            mutant[j] = ind[j] + self.sigma[j] * np.random.normal(0, 1)
        mutant = np.clip(mutant, func.lowerbound, func.upperbound)

        return mutant

    def __call__(self, func, tau, target):
        """ Runs the evolutionary algorithm.

        Parameters:
        --------------
        func : object-like
            The function to be optimized.

        Returns:
        --------------
        x_opt : array
                The best found solution.

        f_opt: float
               The fitness value for the best found solution.

        """

        # Initialize parameters depending on problem
        self.d = func.number_of_variables
        eval_budget = self.budget * self.d
        self.global_tau = 1 / math.sqrt(2*self.d)
        self.local_tau = 1 / math.sqrt(2*math.sqrt(self.d))

        """ For single step mutation, use:

        sigma = 10/6
        self.tau = 1 / math.sqrt(self.d)

        """

        # Initialization
        self.pop = self.initialize_population()    # parent population
        f = np.zeros((self.mu, 1))    # parents fitness
        xo = np.zeros((self.lamda, self.d))    # offspring population
        fo = np.zeros((self.lamda, 1))    # offspring fitness
        x_opt = None
        f_opt = np.inf
        self.sigma = np.zeros(self.d)
        for s in range(0, self.d):
            self.sigma[s] = 10 / 6

        # Evaluate initial parent population
        for i in range(0, self.mu):
            f[i] = func(self.pop[i])

        running_condition = True
        
        # Evaluation loop
        while func.evaluations < eval_budget and running_condition:
            # Update sigma
            self.sigma = self.update_sigma_vec()

            # Create new offspring generation
            for i in range(0, self.lamda):
                offspring = self.global_discrete_recombine()
                xo[i] = self.indstep_mutate(offspring, func)
                fo[i] = func(xo[i])

            # Selection
            # Create joint matrix of individuals and their fitness values
            m = np.concatenate((xo, fo), axis=1)
            # Sort based on fitness (lowest values highest ranking)
            sorted_m = m[np.argsort(m[:, self.d])]
            # Copy best individuals and remove fitness values
            xpnew = sorted_m[0:self.mu, 0:self.d]
            # Set new population for next generation
            self.pop = xpnew
            # Copy fitness values of new generation
            f = sorted_m[0:self.mu, self.d]

            # Check if new best individual is better than previous best
            if f[0] < f_opt:
                x_opt = self.pop[0]
                f_opt = f[0]
                
            if self.order == 'a1': 
                running_condition = func.best_so_far_precision > tau
            elif self.order == 'a2':
                running_condition = func.best_so_far_precision > target

        return x_opt, f_opt