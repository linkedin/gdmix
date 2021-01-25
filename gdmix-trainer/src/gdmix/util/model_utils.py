import numpy as np


def threshold_coefficients(coefficients, threshold_value):
    """
    Set coefficients whose absolute values less than or equal to the threshold to 0.0.

    :param coefficients: a list of floats, usually coefficients from a trained model.
    :param threshold_value: a positive float used as the threshold value.
    :return a numpy array, the zeroed coefficients according to the threshold.
    """
    return np.array([0.0 if abs(x) <= threshold_value else x for x in coefficients])
