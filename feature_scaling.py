def normalize_scale(X, mu_X, sigma_X):
    """mean normalization and feature scaling with standard deviation
    of the input features"""
    X = (X - mu_X) / sigma_X
    return X
