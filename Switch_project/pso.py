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
    None

    Parent:
    ------------

    """
    __doc__ += Algorithm.__doc__

    def __init__(self, func):
        Algorithm.__init__(self, func)
        self.popsize = 40

    def run(self):

        # establish the swarm
        swarm = []
        for i in range(0, self.popsize):
            swarm.append(Particle(self.dim))   # append objects of class Particle
            swarm[i].fitness = self.func(swarm[i].position)   # evaluate fitness for each particle
            swarm[i].pos_best = swarm[i].position.copy()   # set personal_best to current position, use .copy() to avoid link
            swarm[i].best_fitness = swarm[i].fitness   # set best_fitness to current particle's fitness

            # update global best if particle is best solution so far
            if swarm[i].fitness < self.f_opt:
                self.x_opt = swarm[i].position.copy()
                self.f_opt = swarm[i].fitness

        # evaluation loop
        while not self.stop:
            # Update inertia weight
            w = 0.9 - 0.8 * self.func.evaluations / self.budget

            # Iterate through particle swarm
            for k in range(0, self.popsize):
                swarm[k].update_velocity(self.x_opt, w)    # update velocity based on global_best and inertia weight
                swarm[k].update_position(self.func)    # update position with new velocity
                swarm[k].fitness = self.func(swarm[k].position)    # evaluate the particle's fitness

                # check if new particle position is best-so-far for this particle
                if swarm[k].fitness < swarm[k].best_fitness:
                    swarm[k].pos_best = swarm[k].position.copy()
                    swarm[k].best_fitness = swarm[k].fitness

                # check if new particle position is best-so-far globally
                if swarm[k].fitness < self.f_opt:
                    self.x_opt = swarm[k].position.copy()
                    self.f_opt = swarm[k].fitness

        # return best global solution including its fitness value
        return self.x_opt, self.f_opt