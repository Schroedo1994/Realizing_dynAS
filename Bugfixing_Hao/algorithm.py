from parameters import Parameters
import numpy as np

class Algorithm:
    """ Generic algorithm class.

    Parameters:
    --------------
    func : object, callable
        The function to be optimized.
    
    Attributes:
    --------------
    dim : int
        The number of dimenions, problem variables.

    budget : int
        The available budget for function evaluations.

    popsize : int
        The number of individuals in the population.

    """

    def __init__(self, func):
        self.dim = func.number_of_variables
        self.func = func
        self.budget = 0
        self.popsize = 1
        self.x_hist = []
        self.f_hist = []

    def stop(self):
        return False

    def get_params(self):
        pass

        return parameters

    def set_params(self, parameters):
        pass

    def set_stopping_criteria(self, stopping_criteria):
        self.stop = stopping_criteria

    def run(self):
        pass
    
        return self