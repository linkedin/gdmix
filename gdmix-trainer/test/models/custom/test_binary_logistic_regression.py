import numpy as np
import os
import pickle
import tensorflow as tf
from scipy import sparse
from sklearn.model_selection import train_test_split

from gdmix.models.custom.binary_logistic_regression import BinaryLogisticRegressionTrainer
from gdmix.util import constants
from models.custom.test_optimizer_helper import compute_coefficients_and_variance

sample_dataset_path = os.path.join(os.getcwd(), "test/resources/custom")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

_TOLERANCE = 1.0e-5


class TestBinaryLogisticRegressionTrainer(tf.test.TestCase):
    """
    Test binary logistic regression trainer
    """

    def setUp(self):
        # Since grid machines may or may not have access to internet,
        # using a pickled instance of popular open-source breast cancer dataset for testing
        sample_dataset = pickle.load(open(sample_dataset_path + "/sklearn_data.p", "rb"))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(sample_dataset.data,
                                                                                sample_dataset.target,
                                                                                test_size=0.25,
                                                                                random_state=0)

        self.binary_lr_trainer = BinaryLogisticRegressionTrainer(max_iter=1000)
        self.custom_weights = self.binary_lr_trainer.fit(X=self.x_train,
                                                         y=self.y_train,
                                                         weights=None,
                                                         offsets=None)[0][0]

    def test_on_dense_dataset(self):
        """
        Test training on a dense dataset
        """
        # Train on sample data
        self.binary_lr_trainer.fit(X=self.x_train,
                                   y=self.y_train,
                                   weights=None,
                                   offsets=None)

        # Get predictions and metrics on the training data
        training_pred = self.binary_lr_trainer.predict_proba(X=self.x_train,
                                                             offsets=None)
        training_metrics = self.binary_lr_trainer.compute_metrics(X=self.x_train,
                                                                  y=self.y_train,
                                                                  offsets=None)
        # Assert prediction shape matches expectation, and training metrics are within expected range
        assert (0.0 <= training_metrics['auc'] <= 1.0)
        assert (training_pred.shape[0] == self.x_train.shape[0])

    def test_on_sparse_dataset(self):
        """
        Test training on a sparse dataset
        """
        # Train on sparsified sample data
        self.binary_lr_trainer.fit(X=sparse.csr_matrix(self.x_train),
                                   y=self.y_train,
                                   weights=None,
                                   offsets=None)

        # Get predictions and metrics on the training data
        training_pred = self.binary_lr_trainer.predict_proba(X=sparse.csr_matrix(self.x_train),
                                                             offsets=None)
        training_metrics = self.binary_lr_trainer.compute_metrics(X=sparse.csr_matrix(self.x_train),
                                                                  y=self.y_train,
                                                                  offsets=None)
        # Assert prediction shape matches expectation, and training metrics are within expected range
        assert (0.0 <= training_metrics['auc'] <= 1.0)
        assert (training_pred.shape[0] == self.x_train.shape[0])

    def test_scoring_on_validation_data(self):
        """
        Test inference and metrics computation
        """
        # Train on sample data
        self.binary_lr_trainer.fit(X=sparse.csr_matrix(self.x_train),
                                   y=self.y_train,
                                   weights=None,
                                   offsets=None)

        # Get predictions and metrics on the test data
        validation_pred = self.binary_lr_trainer.predict_proba(X=self.x_test,
                                                               offsets=None)
        validation_metrics = self.binary_lr_trainer.compute_metrics(X=self.x_test,
                                                                    y=self.y_test,
                                                                    offsets=None)

        # Assert prediction shape matches expectation, and training metrics are within expected range
        assert (0.0 <= validation_metrics['auc'] <= 1.0)
        assert (validation_pred.shape[0] == self.x_test.shape[0])

    def test_scoring_should_fail_if_not_trained(self):
        """
        Inference should fail on untrained model
        """
        # Reset trainer object
        self.binary_lr_trainer = BinaryLogisticRegressionTrainer()
        with self.assertRaises(Exception):
            self.binary_lr_trainer.predict_proba(X=self.x_test,
                                                 offsets=None)

    def test_scoring_should_fail_if_custom_weights_not_of_known_type(self):
        """
        Inference should fail if custom weights are neither Numpy ndarray or Scipy sparse amtrix
        """
        # Reset trainer object
        self.binary_lr_trainer = BinaryLogisticRegressionTrainer()
        # Run inference using a Python list, which is neither a numpy ndarray nor a scipy matrix
        with self.assertRaises(Exception):
            self.binary_lr_trainer.predict_proba(X=self.x_test,
                                                 offsets=None,
                                                 custom_theta=self.custom_weights.tolist())

    def test_metrics_computation_should_fail_if_model_not_trained(self):
        """
        Metrics computation should fail on untrained model
        """
        # Reset trainer object
        self.binary_lr_trainer = BinaryLogisticRegressionTrainer()
        with self.assertRaises(Exception):
            self.binary_lr_trainer.compute_metrics(X=self.x_test,
                                                   y=self.y_test,
                                                   offsets=None)

    def test_scoring_should_succeed_if_custom_weights_provided(self):
        """
        Inference should succeed on untrained model if custom weights provided
        """
        # Reset trainer object
        self.binary_lr_trainer = BinaryLogisticRegressionTrainer()
        validation_pred = self.binary_lr_trainer.predict_proba(X=self.x_test,
                                                               offsets=None,
                                                               custom_theta=self.custom_weights)
        assert (validation_pred.shape[0] == self.x_test.shape[0])

    def test_metrics_computation_should_succeed_if_custom_weights_provided(self):
        """
        Metrics computation should succeed on untrained model if custom weights provided
        """
        # Reset trainer object
        self.binary_lr_trainer = BinaryLogisticRegressionTrainer()
        validation_metrics = self.binary_lr_trainer.compute_metrics(X=self.x_test,
                                                                    y=self.y_test,
                                                                    offsets=None,
                                                                    custom_theta=self.custom_weights)
        assert (0.0 <= validation_metrics['auc'] <= 1.0)

    def test_training_with_warm_start(self):
        """
        Training with a user provided model for warm start.
        """
        # Get trainer object, but only train 1 L-BFGS step.
        binary_lr_trainer = BinaryLogisticRegressionTrainer(lambda_l2=0.0, max_iter=1)
        coefficients_warm_start = binary_lr_trainer.fit(X=self.x_train,
                                                        y=self.y_train,
                                                        weights=None,
                                                        offsets=None,
                                                        theta_initial=self.custom_weights)[0][0]
        # Warm start.
        # The trained model should be close to initial value
        # since the solution should have already converged.
        self.assertAllClose(coefficients_warm_start, self.custom_weights,
                            rtol=_TOLERANCE, atol=_TOLERANCE, msg='models mismatch')

        coefficients_cold_start = binary_lr_trainer.fit(X=self.x_train,
                                                        y=self.y_train,
                                                        weights=None,
                                                        offsets=None,
                                                        theta_initial=None)[0][0]
        # Cold start
        # The trained model should be far from initial value since we only train 1 step,
        # while the initial model was trained for 100 steps.
        self.assertNotAllClose(coefficients_cold_start, self.custom_weights, msg='models are too close')

    def test_fit_with_variance_computation(self):
        """
        Test fit when the variance computation is required
        """
        # Generate the dataset
        num_features = 10
        num_samples = 100
        X = np.random.randn(num_samples, num_features)
        y = np.random.randint(2, size=num_samples)
        weights = np.random.rand(num_samples)
        offsets = np.random.randn(num_samples)
        lambda_l2 = 0.0
        binary_lr_trainer = BinaryLogisticRegressionTrainer(lambda_l2=lambda_l2,
                                                            max_iter=1000, regularize_bias=True)
        expected_simple = compute_coefficients_and_variance(X=X, y=y, weights=weights, offsets=offsets,
                                                            variance_mode=constants.SIMPLE,
                                                            lambda_l2=lambda_l2)

        expected_full = compute_coefficients_and_variance(X=X, y=y, weights=weights, offsets=offsets,
                                                          variance_mode=constants.FULL,
                                                          lambda_l2=lambda_l2)

        actual_simple = binary_lr_trainer.fit(X=sparse.csr_matrix(X),
                                              y=y,
                                              weights=weights,
                                              offsets=offsets,
                                              variance_mode=constants.SIMPLE)

        actual_full = binary_lr_trainer.fit(X=sparse.csr_matrix(X),
                                            y=y,
                                            weights=weights,
                                            offsets=offsets,
                                            variance_mode=constants.FULL)
        self.assertAllClose(expected_simple[0], actual_simple[0][0], rtol=1e-02, atol=1e-02,
                            msg='simple mean mismatch')
        self.assertAllClose(expected_simple[1], actual_simple[1], rtol=1e-02, atol=1e-02,
                            msg='simple variance mismatch')
        self.assertAllClose(expected_full[0], actual_full[0][0], rtol=1e-02, atol=1e-02,
                            msg='full mean mismatch')
        self.assertAllClose(expected_full[1], actual_full[1], rtol=1e-02, atol=1e-02,
                            msg='full variance mismatch')
