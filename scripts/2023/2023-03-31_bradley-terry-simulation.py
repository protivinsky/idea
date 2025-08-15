import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize
from libs.maths import bradley_terry

rng = default_rng()

num_items = 5
min_matches = 10
max_matches = 100
# num_simulations = 1
num_simulations = 1000

p = rng.normal(size=num_items)
p[0] = 0.

num_matches = rng.integers(low=min_matches, high=max_matches, size=(num_items, num_items))
num_matches[np.triu_indices_from(num_matches)] = 0

logit_diffs = np.subtract.outer(p, p)
P = np.exp(logit_diffs) / (1 + np.exp(logit_diffs))

sim_data = rng.binomial(n=num_matches, p=P, size=(num_simulations, num_items, num_items))

X = sim_data + (num_matches - sim_data).transpose(0, 2, 1)

# use the function I put into maths
p_hats = np.zeros(shape=(num_simulations, num_items), dtype=np.float_)
for i in range(num_simulations):
    p_hats[i] = bradley_terry(X[i], verbose=False)[0]

H = (num_matches + num_matches.T) * P * P.T - np.diag(np.sum((num_matches + num_matches.T) * P * P.T, axis=1))
p_var = np.zeros(shape=(num_items, num_items), dtype=np.float_)
p_var[1:, 1:] = -np.linalg.inv(H[1:, 1:])

p  # true parameters
np.sqrt(np.diag(p_var))  # true variance of estimated parameters

# estimated parameters
p_hats.mean(axis=0)
p_hats.std(axis=0)

# all is very close, good
# p  # true parameters
# Out[24]: array([ 0.        , -1.00420429, -0.05370208, -0.21473623, -0.9279807 ])
# np.sqrt(np.diag(p_var))  # true variance of estimated parameters
# Out[25]: array([0.        , 0.17645611, 0.18496682, 0.1738499 , 0.19649127])
# p_hats.mean(axis=0)
# Out[26]: array([ 0.        , -1.00782826, -0.05070654, -0.21245722, -0.92948572])
# p_hats.std(axis=0)
# Out[27]: array([0.        , 0.17377976, 0.18212923, 0.17601547, 0.20259833])


p
p_var_hat = bradley_terry(X[0], verbose=False)[1]
np.sqrt(np.diag(p_var_hat))





X_plus_Xt = X + X.transpose(0, 2, 1)
X = X[0]

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
    return p, p_var


max_iter = 1000
tol = 1e-6
p_hat = np.zeros(shape=(num_simulations, num_items), dtype=np.float_)

for i in range(max_iter):
    p_hat_old = p_hat.copy()
    logit_diffs_hat = p_hat[:, :, None] - p_hat[:, None, :]
    P_hat = np.exp(logit_diffs_hat) / (1 + np.exp(logit_diffs_hat))
    grad = np.sum(X - X_plus_Xt * P_hat, axis=2)
    H = X_plus_Xt * P_hat * (1 - P_hat) - (np.sum(X_plus_Xt * P_hat * (1 - P_hat), axis=2)[:, :, None]
                                           * np.eye(num_items))
    H_inv = np.linalg.inv(H[:, 1:, 1:])
    p_hat[:, 1:] = p_hat[:, 1:] - np.einsum('ijk,ik->ij', H_inv, grad[:, 1:])
    if np.max(np.abs(p_hat_old - p_hat)) < tol:
        break


H = (num_matches + num_matches.T) * P * (1 - P) - np.diag(np.sum((num_matches + num_matches.T) * P * (1 - P), axis=1))
p_var = np.linalg.inv(H[1:, 1:])
p_std = np.sqrt(-np.diag(p_var))

p
p_std

p_hat.mean(axis=0)
p_hat.std(axis=0)

foo = np.sum(X_plus_Xt * P_hat * (1 - P_hat), axis=2)[:3, :]
foo.shape

# np.einsum('ij->ijj', foo)




p_hat_mean = p_hat.mean(axis=0)
logit_diffs_hat_mean = np.subtract.outer(p_hat_mean, p_hat_mean)
P_hat_mean = np.exp(logit_diffs_hat_mean) / (1 + np.exp(logit_diffs_hat_mean))
P
(num_matches + num_matches.T)

X.mean(axis=0) / (num_matches + num_matches.T)
P

for i in range(num_simulations):
    np.linalg.inv(H[i, 1:, 1:])

np.linalg.inv(H[2, 1:, 1:])
X[2]
np.linalg.inv(X[2, 1:, 1:])

H[2]

grad.shape


p_hat2 = np.zeros(shape=(num_simulations, num_items), dtype=np.float_)
for i in range(num_simulations):
    p_hat2[i] = bradley_terry(X[i], verbose=False)

p_hat2.mean(axis=0)
p_hat2.std(axis=0)

H = X_plus_Xt * P_hat * (1 - P_hat) - (np.sum(X_plus_Xt * P_hat * (1 - P_hat), axis=2)[:, :, None]
                                       * np.eye(num_items))

H_inv = np.linalg.inv(H[:, 1:, 1:])
np.einsum('ijk,ik->ij', H_inv, grad[:, 1:])

H_inv.shape
grad[:, 1:]



# diags = np.sum(X_plus_Xt * P_hat * (1 - P_hat), axis=2)
# diags[:, :, None] * np.eye(5)



X[1]
bradley_terry(X[0])
p




