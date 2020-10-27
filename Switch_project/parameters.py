class Parameters:
    """ Class to define parameters object attributes.

    Parameters:
    --------------
    None

    Attributes:
    --------------

    budget : int
        The number of available function evaluations.

    """

    def __init__(self):
        self.budget = 0
        self.x_opt = None

def init_params(budget):
    """ Initialize parameter settings for first algorithm.

    Parameters:
    --------------
    budget : int
        The number of available function evaluations.

    Returns:
    --------------
    parameters : object-type
        Parameters object that holds different params.

    """
    parameters = Parameters()
    parameters.budget = budget

    return parameters

