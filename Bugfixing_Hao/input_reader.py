# TODO: implement csv reader to create dict from data source

# Import algorithm classes
from bfgs import BFGS
from cmaes import CMAES

def create_dict():
    """ Manually creates dictionary.

    Returns:
    ------------
    data : dict
        A dictionary holding tau as keys, algorithms as values.

    """
    data = {
      2.51e-5: BFGS, 
      1e-8: CMAES  
    }

    return data