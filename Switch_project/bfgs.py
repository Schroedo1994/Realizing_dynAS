import numpy as np
import random as rd
import math
import datetime
from shutil import make_archive

from scipy.optimize.optimize import _prepare_scalar_function
from scipy.optimize.optimize import vecnorm
from scipy.optimize.optimize import _line_search_wolfe12
from scipy.optimize.optimize import _LineSearchError
from scipy.optimize.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message
from scipy.optimize._differentiable_functions import ScalarFunction, FD_METHODS
from scipy.optimize.linesearch import (line_search_wolfe1, line_search_wolfe2,
                         line_search_wolfe2 as line_search,
                         LineSearchWarning)

from algorithm import Algorithm

class BFGS(Algorithm):

    """ Optimizer class for BFGS algorithm.

    Parameters:
    -----------------
    func : object, callable
        The function to be optimized.

    Attributes:
    -----------------

    gtol : float
        Value for gradient tolerance.

    norm : float

    eps : float

    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.

    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x0) * max(1, abs(x0))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
        
    x0 : array-type
        The initial point for the algorithm routine.
        
    Hk : array-type
        Hessian approximation matrix at iteration k
        
    alpha_k : float
        Step size after iteration k
        
    Methods:
    --------------
    
    set_params(parameters):
        Sets algorithm parameters for warm-start.
        
    get_params(parameters):
        Transfers internal parameters to parameters dictionary.
        
    run():
        Runs the BFGS algorithm.
        
    Parent:
    ------------

    """
    __doc__ += Algorithm.__doc__

    def __init__(self, func):
        Algorithm.__init__(self, func)
        self.gtol = 1e-10
        self.norm = np.inf
        self.eps = math.sqrt(np.finfo(float).eps)
        self.return_all = False
        self.jac = None
        self.finite_diff_rel_step = None
        self.x0 = None
        self.Hk = np.eye(self.dim, dtype=int)   # B0 = identity
        self.alpha_k = 0

    def set_params(self, parameters):
        self.budget = parameters.budget

        """Warm start routine"""

        # Initialize first point x0
        
        if 'x_opt' in parameters.internal_dict:
            self.x0 = parameters.internal_dict['x_opt']
        
        # Initialize stepsize alpha_k
        if 'stepsize' in parameters.internal_dict:
            self.alpha_k = parameters.internal_dict['stepsize']

        # Initialize Hk
  
        # if 'C' in parameters.internal_dict:
        #     do something nice
        #     finally set Hk to something


    def get_params(self, parameters):
        parameters.internal_dict['Hk'] = self.Hk
        parameters.internal_dict['stepsize'] = self.alpha_k
        parameters.internal_dict['x_opt'] = self.func.best_so_far_variables

        return parameters


    def run(self):
        """ Runs the BFGS algorithm.

        Parameters:
        --------------
        None

        Returns:
        --------------
        best_so_far_variables : array
                The best found solution.

        best_so_far_fvaluet: float
               The fitness value for the best found solution.

        """

        print(f' BFGS started')

        # Initialization 
        I = np.eye(self.dim, dtype=int)    # identity matrix
        k = 0

        # Initialize first point x0 at random
        if self.x0 is None:
            self.x0 = np.zeros(self.dim)
            for i in range(0, self.dim):
                self.x0[i] = rd.uniform(-5, 5)

        # Prepare scalar function object and derive function and gradient function
        sf = _prepare_scalar_function(self.func, self.x0, self.jac, epsilon=self.eps,
                              finite_diff_rel_step=self.finite_diff_rel_step)
        f = sf.fun    # function object to evaluate function
        gradient = sf.grad    # function object to evaluate gradient

        old_fval = f(self.x0)    # evaluate x0
        gfk = gradient(self.x0)   # gradient at x0

        # Sets the initial step guess to dx ~ 1
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        xk = self.x0

        if self.return_all:
            allvecs = [self.x0]

        # Calculate initial gradient norm
        gnorm = vecnorm(gfk, ord = self.norm)
        
        # Algorithm loop
        while not self.stop():
            pk = -np.dot(self.Hk, gfk)    # derive direction pk from HK and gradient at x0 (gfk)
            """Derive alpha_k with Wolfe conditions.
            
            alpha_k : step size
            fc : count of function evaluations, gc: count of gradient evaluations
            old_fval : function value of new point xkp1 (xk + ak * pk)
            old_old_fval: function value of start point xk
            gfkp1 : gradient at new point xkp1
            """
            try:
                self.alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, gradient, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100, amax=1e100)
                
            except _LineSearchError:
                #print('break because of line search error')
                break

            # calculate xk+1 with alpha_k and pk
            xkp1 = xk + self.alpha_k * pk
            if self.return_all:
                allvecs.append(xkp1)
            sk = xkp1 - xk    # step sk is difference between xk+1 and xk
            xk = xkp1    # make xk+1 new xk for next iteration
            # Calculate gradient of xk+1 if not already found by Wolfe search
            if gfkp1 is None:
                gfkp1 = gradient(xkp1)
            yk = gfkp1 - gfk    # gradient difference
            gfk = gfkp1    # copy gradient to gfk for new iteration
            k += 1

            if not np.isfinite(old_fval):
                #print('break because of np.isfinite')
                break

            # Check if gnorm is already smaller than tolerance
            gnorm = vecnorm(gfk, ord=self.norm)
            if (gnorm <= self.gtol):
                #print('break because of gnorm')
                break

            # Calculate rhok factor for Hessian approximation matrix update
            try:
                rhok = 1.0 / (np.dot(yk, sk))
            except ZeroDivisionError:
                rhok = 1000.0
            if np.isinf(rhok):  # this is patch for NumPy
                rhok = 1000.0

            # Hessian approximation matrix Hk (Bk in papers) update
            A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
            A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
            self.Hk = np.dot(A1, np.dot(self.Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                     sk[np.newaxis, :])

        # Store found fitness value in fval for result
        fval = old_fval

        # Create OptimizeResult object based on found point and value
        result = OptimizeResult(fun=fval, jac=gfk, hess_inv=self.Hk, nfev=sf.nfev,
                        njev=sf.ngev, x=xk,
                        nit=k)

        if self.return_all:
            result['allvecs'] = allvecs

        print(f' BFGS complete')

        return self.func.best_so_far_variables, self.func.best_so_far_fvalue