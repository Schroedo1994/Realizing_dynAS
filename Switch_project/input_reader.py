# TODO: implement csv reader to create dict from data source

# Import algorithm classes
from bfgs import BFGS
from mlsl import MLSL
from ea import EA
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
      25.1: CMAES, 
      1e-8: BFGS 
    }

    return data