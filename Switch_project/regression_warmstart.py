import numpy as np
from bayes_optim.Surrogate import GaussianProcess, trend

def cma_es_warm_starting(
    X, # should be an numpy 2-array of shape (n_sample, n_dim)
    y, # should be an numpy array of shape (n_sample, )
    **kwargs
    ):
    dim = X.shape[1]
    fopt = np.min(y)
    xopt = X[np.where(y == fopt)[0][0]]

    mean = trend.constant_trend(dim, beta=None)  # Simple Kriging
    thetaL = 1e-10 * np.ones(dim)
    thetaU = 10 * np.ones(dim)
    theta0 = np.random.rand(dim) * (thetaU - thetaL) + thetaL

    model = GaussianProcess(
        mean=mean, corr='squared_exponential',
        theta0=theta0, thetaL=thetaL, thetaU=thetaU,
        nugget=1e-6, noise_estim=False,
        optimizer='BFGS', wait_iter=5, random_start=5 * dim,
        eval_budget=100 * dim
    )
    model.fit(X, y)

    # obtain the Hessian and gradient from the GP mean surface
    H = model.Hessian(xopt)
    g = model.gradient(xopt)[0]

    w, B = np.linalg.eigh(H)
    w[w <= 0] = 1e-6     # replace the negative eigenvalues by a very small value
    w_min, w_max = np.min(w), np.max(w)

    # to avoid the conditional number gets too high
    cond_upper = 1e3
    delta = (cond_upper * w_min - w_max) / (1 - cond_upper)
    w += delta

    # compute the upper bound for step-size
    M = np.diag(1 / np.sqrt(w)).dot(B.T)
    H_inv = B.dot(np.diag(1 / w)).dot(B.T)
    p = -1 * H_inv.dot(g).ravel()
    alpha = np.linalg.norm(p)

    if np.isnan(alpha):
        alpha = 1
        H_inv = np.eye(dim)

    # use a backtracking line search to determine the initial step-size
    tau, c = 0.9, 1e-4
    slope = np.inner(g.ravel(), p.ravel())

    if slope > 0:  # this should not happen..
        p *= -1
        slope *= -1

    f = lambda x: model.predict(x)
    while True:
        _x = (xopt + alpha * p).reshape(1, -1)
        if f(_x) <= f(xopt.reshape(1, -1)) + c * alpha * slope:
            break
        alpha *= tau

    sigma0 = np.linalg.norm(M.dot(alpha * p)) / np.sqrt(dim - 0.5)
    
    return sigma0, H_inv