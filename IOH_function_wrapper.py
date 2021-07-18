from IOHexperimenter import IOH_function

class IOH_function_logger(IOH_function):

    def __init__(self, func_id, dim, inst, *args, **kwargs):
        IOH_function.__init__(self, func_id, dim, inst, *args, **kwargs)
        self.x_hist = []
        self.f_hist = []
        
    def __call__(self, x):
        self.x_hist.append(x)
        f = IOH_function.__call__(self, x)
        self.f_hist.append(f)
        
        return f
