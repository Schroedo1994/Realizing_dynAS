# Import algorithm classes
from bfgs import BFGS
from mlsl import MLSL
from pso import PSO
from cmaes import CMAES
from de import DE

def create_dict():
    """ Manually creates dictionary.

    Returns:
    ------------
    data : dict
        A dictionary holding tau as keys, algorithms as values.

    """
    data = {
      1e-2: CMAES, 
      1e-8: BFGS
    }

    return data