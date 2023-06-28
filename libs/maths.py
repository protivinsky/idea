import numpy as np
import pandas as pd


def logistic(x):
    return 1 / (1 + np.exp(-x))


def nanaverage(x, weights=None):    
    if weights is None:
        if len(x.shape) == 1:
            return np.nanmean(x)
        else:
            res = np.nanmean(x, axis=0)
            return pd.Series(res, x.columns) if isinstance(x, pd.DataFrame) else res
    else:
        w = x[weights].fillna(0)
        x = x.drop(columns=[weights])
        mask = np.isnan(x)
        xm = np.ma.masked_array(x, mask=mask)
        if len(x.shape) == 1:
            return np.ma.average(xm, weights=w)
        else:
            res = np.ma.average(xm, weights=w, axis=0)
            return pd.Series(res, x.columns) if isinstance(x, pd.DataFrame) else res            


def bradley_terry(X, tol=1e-6, max_iter=1000, verbose=True):
    """ Estimates the Bradley-Terry model via maximum likelihood. """
    X = np.asarray(X)
    n = X.shape[0]
    p_hat = np.zeros(shape=n, dtype=np.float_)
    for i in range(max_iter):
        p_hat_old = p_hat.copy()
        logit_diffs = np.subtract.outer(p_hat, p_hat)
        P_hat = np.exp(logit_diffs) / (1 + np.exp(logit_diffs))
        ll = -np.sum(X * np.log(P_hat))
        if verbose:
            print(f'Step {i}, log-likelihood {ll:.3g}, params {p_hat}.')
        H = (X + X.T) * P_hat * P_hat.T - np.diag(np.sum((X + X.T) * P_hat * P_hat.T, axis=1))
        grad = np.sum(X - (X + X.T) * P_hat, axis=1)
        p_hat[1:] = p_hat[1:] - np.linalg.inv(H[1:, 1:]) @ grad[1:]

        if np.max(np.abs(p_hat_old - p_hat)) < tol:
            break

    logit_diffs = np.subtract.outer(p_hat, p_hat)
    P_hat = np.exp(logit_diffs) / (1 + np.exp(logit_diffs))
    H = (X + X.T) * P_hat * P_hat.T - np.diag(np.sum((X + X.T) * P_hat * P_hat.T, axis=1))
    p_var = np.zeros(shape=(n, n), dtype=np.float_)
    p_var[1:, 1:] = -np.linalg.inv(H[1:, 1:])
    return p_hat, p_var
