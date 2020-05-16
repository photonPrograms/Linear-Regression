# the constant terms like 1/m are included at the final stage

import numpy as np

def cost_func(X, y, theta):
    """compute the cost function for linear regression
    X is the training examples input matrix, each row containing one example
    y is the row vector containing output examples
    theta is the row vector for parameters"""

    J = np.sum((theta @ np.transpose(X) - y) ** 2)
    return J

def gradient(X, y, theta):
    """compute the gradient for gradient descent"""

    G = (theta @ np.transpose(X) - y) @ X
    return G
