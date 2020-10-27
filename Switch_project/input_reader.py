# TODO: implement csv reader to create dict from data source

# Import algorithm classes
from bfgs import BFGS
from mlsl import MLSL

def create_dict():
    """ Manually creates dictionary.

    Returns:
    ------------
    data : dict
        A dictionary holding tau as keys, algorithms as values.

    """
    data = {
      0.63095: MLSL,
      1e-8: BFGS
    }

    return data