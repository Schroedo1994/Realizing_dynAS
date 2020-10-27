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

    x_opt : array-type
        The best found solution.

    f_opt : float
        Function value for the best found solution.

    popsize : int
        The number of individuals in the population.

    """

    def __init__(self, func):
        self.dim = func.number_of_variables
        self.func = func
        self.budget = 0
        self.x_opt = None
        self.f_opt = np.inf
        self.popsize = 1

    def stop(self):
        return False

    def get_params(self):
        parameters = Parameters()
        parameters.budget = self.budget
        parameters.x_opt = self.x_opt

        return parameters

    def set_params(self, parameters):
        self.budget = parameters.budget

        if hasattr(self, 'x0'):
            self.x0 = parameters.x_opt

    def set_stopping_criteria(self, stopping_criteria):
        self.stop = stopping_criteria

    def run(self):
        pass