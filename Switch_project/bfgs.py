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

class BFGS:

    """ Optimizer class for BFGS algorithm.

    Parameters:
    -----------------
    budget : int
        Budget for function evaluations.

    Attributes:
    -----------------
    d : int
        The number of dimensions, problem variables.

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

    """

    def __init__(self, budget, order):
        self.budget = budget
        self.d = 1
        self.gtol = 1e-10
        self.norm = np.inf
        self.eps = math.sqrt(np.finfo(float).eps)
        self.return_all = False
        self.jac = None
        self.finite_diff_rel_step = None
        self.order = order
        self.x0 = None

    def __call__(self, func, tau, target):
        """ Runs the BFGS algorithm.

        Parameters:
        --------------
        func : object-like
            The function to be optimized.

        Returns:
        --------------
        x_opt : array
                The best found solution.

        f_opt: float
               The fitness value for the best found solution.

        """

        # Initialization
        self.d = func.number_of_variables
        eval_budget = self.budget * self.d
        x_opt = None
        f_opt = np.inf
        retall = self.return_all
        I = np.eye(self.d, dtype=int)    # identity matrix
        Hk = np.eye(self.d, dtype=int)   # B0 = identity
        k = 0

        # Initialize first point x0 at random
        if self.x0 is None:
            self.x0 = np.zeros(self.d)
            for i in range(0, self.d):
                self.x0[i] = rd.uniform(-5, 5)
        
        print(f' BFGS started, x0 = {self.x0}')

        # Prepare scalar function object and derive function and gradient function
        sf = _prepare_scalar_function(func, self.x0, self.jac, epsilon=self.eps,
                              finite_diff_rel_step=self.finite_diff_rel_step)
        f = sf.fun    # function object to evaluate function
        gradient = sf.grad    # function object to evaluate gradient

        old_fval = f(self.x0)    # evaluate x0
        gfk = gradient(self.x0)   # gradient at x0

        # Sets the initial step guess to dx ~ 1
        old_old_fval = old_fval + np.linalg.norm(gfk) / 2

        xk = self.x0

        if retall:
            allvecs = [self.x0]

        # Calculate initial gradient norm
        gnorm = vecnorm(gfk, ord=self.norm)
        
        # Algorithm loop
        while (func.evaluations < eval_budget) and not func.final_target_hit:
            pk = -np.dot(Hk, gfk)    # derive direction pk from HK and gradient at x0 (gfk)
            # Derive alpha_k with Wolfe conditions
            try:
                alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     _line_search_wolfe12(f, gradient, xk, pk, gfk,
                                          old_fval, old_old_fval, amin=1e-100, amax=1e100)
            except _LineSearchError:
                print('break because of line search error')
                break

            # calculate xk+1 with alpha_k and pk
            xkp1 = xk + alpha_k * pk
            if retall:
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
                print('break because of np.isfinite')
                break

            # Check if gnorm is already smaller than tolerance
            gnorm = vecnorm(gfk, ord=self.norm)
            if (gnorm <= self.gtol):
                print('break because of gnorm')
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
            Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                     sk[np.newaxis, :])
            
            print(f' BFGS prec: {func.best_so_far_precision} evals: {func.evaluations}')
            
            if self.order == 'a1' and (func.best_so_far_precision <= tau):
                break
            if self.order == 'a2' and (func.best_so_far_precision <= target):
                break

        # Store found fitness value in fval for result
        fval = old_fval

        # Create OptimizeResult object based on found point and value
        result = OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                        njev=sf.ngev, x=xk,
                        nit=k)

        # Store in x_opt and f_opt
        x_opt = result.x
        f_opt = result.fun
        print(f' BFGS done, x: {x_opt} f: {f_opt}')

        if retall:
            result['allvecs'] = allvecs

        return x_opt, f_opt