import numpy as np
import tensorflow as tf

from gdmix.util.model_utils import threshold_coefficients


class TestModelUtils(tf.test.TestCase):
    """
    Test Model Utils
    """
    def testThresholdCoefficients(self):
        coefficients = np.array([1e-5, -1e-4, -0.1, -1e-5, 2.2, 3.3])
        expected_coefficients = np.array([0.0, 0.0, -0.1, 0.0, 2.2, 3.3])
        actual_coefficients = threshold_coefficients(coefficients, 1e-4)
        self.assertAllEqual(actual_coefficients, expected_coefficients)
