import random as rd
import math
import numpy as np
import copy
from algorithm import Algorithm

class Particle:
    """ Create one particle for the swarm.

    Parameters:
    --------------
    dim : int
        Number of problem variables, dimensionality.

    Attributes:
    --------------
    position : array-like
        The current position of the particle.

    velocity : array-like
        The current velocity of the particle.

    fitness : float
        The current fitness value of the particle.

    pos_best : array-like
        The all-time best position found by the particle.

    best_fitness : float
        The fitness value at the all-time best position.

    """

    def __init__(self, dim):
        self.dim = dim
        self.position = np.zeros(self.dim)
        self.velocity = np.zeros(self.dim)
        self.fitness = np.inf
        self.pos_best = np.zeros(self.dim)
        self.best_fitness = np.inf

        # initialize particle with uniform random position and velocities
        for i in range(0, self.dim):
            self.velocity[i] = rd.uniform(-5, 5)
            self.position[i] = rd.uniform(-5, 5)

    def update_velocity(self, pos_best_g, w):
        """ Update the particle's velocity.

        Parameters:
        --------------
        pos_best_g : array-like
            The best position found by the swarm.
            
        w : float
            Inertia weight to control velocity update.

        """
        c1 = 1.4944     # social constant
        c2 = 1.4944     # cognitive constant
        vel_lowerbound = -5   # lower bound for velocity
        vel_upperbound = 5    # upper bound for velocity

        # create uniform random vectors between 0 and c1 / c2
        u1 = np.random.uniform(0, c1, self.dim)
        u2 = np.random.uniform(0, c2, self.dim)

        """ update velocity with previous velocity and component-wise multiplication of u-vectors
        with differences between current solution, previous best and global best"""
        for i in range(0, self.dim):
            self.velocity[i] = w * self.velocity[i] + u1[i] * (self.pos_best[i]
                                                    - self.position[i]) + u2[i] * (pos_best_g[i] - self.position[i])

        self.velocity = np.clip(self.velocity, vel_lowerbound, vel_upperbound)

    def update_position(self, func):
        """ Update a particle's position.

        """
        for i in range(0, self.dim):
            result = self.position[i] + self.velocity[i]

            # check boundary conditions, reset velocity to zero if particle is on boundary
            if result < -5:
                self.position[i] = -5
                self.velocity[i] = 0
            elif result > 5:
                self.position[i] = 5
                self.velocity[i] = 0
            else:
                self.position[i] = result

class PSO(Algorithm):
    """ Class for the PSO algorithm.

    Parameters:
    ---------------
    None

    Attributes:
    ---------------
    g_best : array-type
        Best solution found so far.
        
    g_best_f : float
        Fitness value at best solution so far.
    
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
        self.g_best = None
        self.g_best_f = np.inf
        self.popsize = 40
        
    def set_params(self, parameters):
        self.budget = parameters.budget

        """Warm start routine"""

    def get_params(self, parameters):
        parameters.internal_dict['x_opt'] = self.func.best_so_far_variables

        return parameters


    def run(self):
        """ Runs the PSO algorithm.

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

        print(f' PSO started')

        # establish the swarm
        swarm = []
        for i in range(0, self.popsize):
            swarm.append(Particle(self.dim))   # append objects of class Particle
            swarm[i].fitness = self.func(swarm[i].position)   # evaluate fitness for each particle
            swarm[i].pos_best = swarm[i].position.copy()   # set personal_best to current position, use .copy() to avoid link
            swarm[i].best_fitness = swarm[i].fitness   # set best_fitness to current particle's fitness

            # update global best if particle is best solution so far
            if swarm[i].fitness < self.g_best_f:
                self.g_best = swarm[i].position.copy()
                self.g_best_f = swarm[i].fitness

        # evaluation loop
        while not self.stop():
            # Update inertia weight
            w = 0.9 - 0.8 * self.func.evaluations / self.budget

            # Iterate through particle swarm
            for k in range(0, self.popsize):
                swarm[k].update_velocity(self.g_best, w)    # update velocity based on global_best and inertia weight
                swarm[k].update_position(self.func)    # update position with new velocity
                swarm[k].fitness = self.func(swarm[k].position)    # evaluate the particle's fitness

                # check if new particle position is best-so-far for this particle
                if swarm[k].fitness < swarm[k].best_fitness:
                    swarm[k].pos_best = swarm[k].position.copy()
                    swarm[k].best_fitness = swarm[k].fitness

                # check if new particle position is best-so-far globally
                if swarm[k].fitness < self.g_best_f:
                    self.g_best = swarm[k].position.copy()
                    self.g_best_f = swarm[k].fitness
        
        print(f' PSO complete')

        # return best global solution including its fitness value
        return self.func.best_so_far_variables, self.func.best_so_far_fvalue