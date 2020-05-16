import numpy as np
import json

from feature_scaling import normalize_scale
from cost_grad import cost_func, gradient
from normal_eqn import compute_theta_normal

X_list, y_list = [], []

filename = "training_set.data"
with open(filename) as f:
    for line in f:
        a = np.array(line.rstrip().split(), dtype = np.float)
        X_list.append(a[:-1].tolist())
        y_list.append(a[-1].tolist())

m = len(y_list) # number of training examples

X = np.hstack((np.ones((m, 1)), np.array(X_list))) # input training examples
y = np.array(np.array(y_list)) # output training examples

mu_X = np.mean(X, axis = 0)
sigma_X = np.std(X, axis = 0)
for i in range(np.size(sigma_X)):
    if sigma_X[i] == 0:
        sigma_X[i] = X[0][i]
        mu_X[i] = 0

# mean normalization and feature scaling
X = normalize_scale(X, mu_X, sigma_X)

theta = np.zeros((np.size(X, axis = 1))) # parameter vector

NITER = 1000 # number of iterations for batch gradient descent
ALPHA = 0.01 # learning rate

J = [] # list of cost functions evaluated over NITER passes

# gradient descent
for i in range(NITER):
    J.append(cost_func(X, y, theta) / (2 * m))
    theta -= ALPHA / m * gradient(X, y, theta)

filename = "cost.json"
with open(filename, "w") as f:
    json.dump(J, f)

results = {
    "mu_X": mu_X.tolist(),
    "sigma_X": sigma_X.tolist(), 
    "theta": theta.tolist()
}

# calculating theta with normal equation
results["theta_n"] = compute_theta_normal(X, y).tolist()

filename = "hyp_data.json"
with open(filename, "w") as f:
    json.dump(results, f)
