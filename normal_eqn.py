import numpy as np

def compute_theta_normal(X, y):
    """using normal equation method to obtain the parameter vector theta"""
    return y @ np.transpose(np.linalg.inv(np.transpose(X) @ X) @ np.transpose(X))
