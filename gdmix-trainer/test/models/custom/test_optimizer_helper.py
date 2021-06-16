import numpy as np
import scipy
import statsmodels.api as sm

from gdmix.util import constants


def compute_coefficients_and_variance(X, y, weights=None, offsets=None,
                                      variance_mode=constants.SIMPLE,
                                      lambda_l2=0.0, has_intercept=True):
    """
    compute coefficients and variance for logistic regression model
    :param X: num_samples x num_features matrix
    :param y: num_samples binary labels (0 or 1)
    :param weights: num_samples floats, weights of each sample.
    :param offsets: num_samples floats, offset of each sample
    :param variance_mode: full or simple
    :param lambda_l2: L2 regularization coefficient
    :param has_intercept: whether to include intercept
    :return: (mean, variance) tuple
    """
    if scipy.sparse.issparse(X):
        X = X.toarray()
    X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X)) if has_intercept else X
    lr_model = sm.GLM(y, X_with_intercept, family=sm.families.Binomial(),
                      offset=offsets, freq_weights=weights)
    if lambda_l2 != 0.0:
        raise ValueError("This function uses statsmodels to compute LR coefficients and its variance. "
                         "However, as of version 0.12.2, the coefficients when non-zero L2 regularization"
                         " is applied are not correct. So we can only check L2=0.")
    lr_results = lr_model.fit_regularized(alpha=lambda_l2, maxiterint=500,
                                          cnvrg_tol=1e-12, L1_wt=0.0)
    mean = lr_results.params
    hessian = lr_model.hessian(mean)
    if variance_mode == constants.SIMPLE:
        variance = -1.0 / np.diagonal(hessian)
    elif variance_mode == constants.FULL:
        variance = -np.diagonal(np.linalg.inv(hessian))
    return mean, variance
