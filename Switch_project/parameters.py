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
        self.pop = None
        self.f = None
        self.internal_dict = {}
        
#     def set_history
#         self.info_available += 'hist'
#         #x0, full history x, corresponding f
        
    hist_dict = {}
    
    def set_info(self, name, dict_):
        self.internal_dict[name] = dict_
        
        if 'vel' in self.internal_dict:
            
    

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

