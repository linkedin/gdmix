import numpy as np
import scipy.sparse
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import expit
from sklearn import metrics

from gdmix.util import constants


class BinaryLogisticRegressionTrainer:
    """
    Class to train l2-regularized  binary logistic regression. Supports the following:

    [1] Training
    [2] Inference/scoring
    [3] Metrics computation

    This implementation assumes that the binary label setting is 0/1. It also automatically adds an intercept
    term during optimization.

    The loss and the gradient assume l2-regularization. Users can use the "regularize_bias" switch to regularize the
    intercept term or not.

    In this implementation, we assume the first element of the model vector is the intercept.
    """

    def __init__(self, lambda_l2=0.0, solver="lbfgs", precision=10, num_lbfgs_corrections=10,
                 max_iter=100, regularize_bias=False, has_intercept=True):
        self.lambda_l2 = lambda_l2
        assert solver in ("lbfgs",)
        self.solver = solver

        # Iterations stop when (f^k - f^{k+1})/max{|f^k|,|f^{k+1}|,1} <= precision * eps
        self.precision = precision

        self.num_lbfgs_corrections = num_lbfgs_corrections
        self.max_iter = max_iter
        self.regularize_bias = regularize_bias
        self.has_intercept = has_intercept

        # Set the model parameters to None
        self.theta = None
        self.epsilon = 1.0e-12

    def _sigmoid(self, z):
        """
        Calculate element-wise sigmoid of array z
        """

        # For now, using Scipy's numerically stable sigmoid function
        return expit(z)

    def _predict(self, theta, X, offsets, return_logits=False):
        """
        Calculate logistic regression output when an input matrix is pushed through a parameterized LR model
        Output can be logits or sigmoid probabilities
        """
        # Handle dense and sparse theta separately
        if isinstance(theta, np.ndarray):
            z = X.dot(theta) + offsets
        elif scipy.sparse.issparse(theta):
            z = X.dot(theta).toarray().squeeze() + offsets
        else:
            raise Exception(f"Unknown type: {type(theta)!r} for model weights. Accepted types are Numpy ndarray and Scipy sparse matrices")
        return z if return_logits else self._sigmoid(z)

    def _get_number_of_samples(self, X):
        """
        Get number of samples from a data 2d-array
        """
        return X.shape[0]

    def _get_loss_from_regularization(self, theta, intercept_index=0):
        """
        Get loss for regularization term. Exclude intercept if self.regularize_bias is set to false
        """
        # For now, we assume "intercept_index" is always zero
        if self.has_intercept and not self.regularize_bias:
            loss = (self.lambda_l2 / 2.0) * theta[intercept_index + 1:].dot(theta[intercept_index + 1:])
        else:
            loss = (self.lambda_l2 / 2.0) * theta.dot(theta)
        return loss

    def _loss(self, theta, X, y, weights, offsets):
        """
        Calculate loss for weighted binary logistic regression
        """
        n_samples = self._get_number_of_samples(X)

        # For numerical stability, we transform the traditional binary cross entropy loss into a stable, equivalent form
        # (Inspired from - https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits)
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        # = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        # = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        # = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        #                                     = (1 - z) * x + log(1 + exp(-x))
        # = x - x * z + log(1 + exp(-x))
        # = log(exp(x)) - x * z + log(1 + exp(-x))
        # = - x * z + log(1 + exp(x))
        # max(x, 0) - x * z + log(1 + exp(-abs(x)))

        pred = X.dot(theta) + offsets
        cross_entropy_cost = np.maximum(pred, 0) - pred * y + np.log(1 + np.exp(-np.absolute(pred)))

        cost = weights * cross_entropy_cost

        # Compute total cost, including regularization
        total_cost = (1.0 / n_samples) * (cost.sum() + self._get_loss_from_regularization(theta))

        return total_cost

    def _get_gradient_from_regularization(self, theta, intercept_index=0):
        """
        Get gradient for regularization term. Exclude intercept if self.regularize_bias is set to false
        """
        gradient = self.lambda_l2 * theta
        if self.has_intercept and not self.regularize_bias:
            gradient[intercept_index] = 0
        return gradient

    def _gradient(self, theta, X, y, weights, offsets):
        """
        Calculate gradient of loss for weighted binary logistic regression
        """
        n_samples = self._get_number_of_samples(X)
        predictions = self._predict(theta, X, offsets)

        cost_grad = X.T.dot(weights * (predictions - y))

        grad = (1.0 / n_samples) * (cost_grad + self._get_gradient_from_regularization(theta))
        return grad

    @staticmethod
    def _add_column_of_ones(X):
        """
        Add intercept column to a dense/sparse matrix
        """
        if isinstance(X, np.ndarray):
            X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
        else:
            X_with_intercept = scipy.sparse.hstack((np.ones((X.shape[0], 1)), X))
        return X_with_intercept

    def _compute_variance(self, X, y, theta, weights=None, offsets=None, mode=constants.SIMPLE):
        """
        Compute the variance of the coefficients
        :param X:       Feature matrix of dimension n x (d + 1), where n is number of samples
                        and d is the number of features.
        :param y:       Binary label vector of dimension n x 1, entries are either 0 or 1.
        :param weights: Float vector of dimension n x 1, entries are the weight of each sample.
        :param offsets: Float vector of dimension n x 1, entries are the offset for each sample.
                        It gets added to the computed logit.
        :return:        Float vector of length n, entries are the variance of each coefficients.

        For logistic regression, Hessian matrix H = X^t * D * X + \\lambda_2 * I,
        where D = diagonal(\\rho_i (1-\\rho_i)), i = 1, ..., n. \\lambda_2 is the l2 regularizer
        When the mode is SIMPLE:
          variance = inverse(diagonal(H)) = inverse([X_j^t * D * X_j]) = inverse([sum(X_ij^2 * d_i)]).
        When the mode is FULL:
          variance = diagonal(inverse(H))
        """
        # Convert to dense
        if scipy.sparse.issparse(theta):
            theta = theta.toarray()
        if scipy.sparse.issparse(X):
            X = X.toarray()
        # Compute probability
        rho = self._predict(theta, X, offsets, False)
        d = rho*(1-rho)
        n, p = X.shape
        if weights is not None:
            d = d * weights
        # D * X
        dX = np.multiply(X, d[:, np.newaxis])
        if mode == constants.SIMPLE:
            H_diag = np.array([X[:, i].dot(dX[:, i]) for i in range(p)]) + self.lambda_l2
            if self.has_intercept and not self.regularize_bias:
                # First element of the vector corresponds to the intercept
                H_diag[0] -= self.lambda_l2
            return 1.0 / (H_diag + self.epsilon)
        elif mode == constants.FULL:
            H = np.transpose(X).dot(dX) + (self.lambda_l2 + self.epsilon) * np.eye(p)
            if self.has_intercept and not self.regularize_bias:
                # First element corresponds to the intercept
                H[0][0] -= self.lambda_l2
            V = np.linalg.inv(H)
            return np.diagonal(V)
        else:
            raise ValueError(f"Unknown variance computation mode, either SIMPLE or FULL, got {mode}!")

    def fit(self, X, y, weights=None, offsets=None, theta_initial=None,
            variance_mode=None):
        """
        Fit a binary logistic regression model
        :param X:               a dense or sparse matrix of dimensions (n x d), where n is the number of samples,
                                and d is the number of features
        :param y:               vector of binary sample labels; of dimensions (n x 1)  where n is the number of samples
        :param weights:         vector of sample weights; of dimensions (n x 1)  where n is the number of samples
        :param offsets:         vector of sample offsets; of dimensions (n x 1)  where n is the number of samples
        :param theta_initial:   a numpy vector, initial value for the coefficients, useful in warm start.
        :param variance_mode:   String; None, SIMPLE or FULL. None means no variance computation.
                                SIMPLE means variance approximation by taking inverse of Hessian diagonals.
                                FULL means taking diagonal of the inverse of full Hessian matrix.
        :return:    training results dictionary, including learned parameters
        """

        # Assert labels are of binary type only
        assert np.count_nonzero((y != 0) & (y != 1)) == 0

        n_samples = self._get_number_of_samples(X)
        if weights is None:
            weights = np.ones(n_samples)
        if offsets is None:
            offsets = np.zeros(n_samples)

        # Assert all shapes are same
        assert (X.shape[0] == y.shape[0] == weights.shape[0] == offsets.shape[0])

        X_with_intercept = self._add_column_of_ones(X) if self.has_intercept else X
        if theta_initial is None:
            theta_initial = np.zeros(X_with_intercept.shape[1])
        # Run minimization
        result = fmin_l_bfgs_b(func=self._loss,
                               x0=theta_initial,
                               approx_grad=False,
                               fprime=self._gradient,
                               m=self.num_lbfgs_corrections,
                               factr=self.precision,
                               maxiter=self.max_iter,
                               args=(X_with_intercept, y, weights, offsets),
                               disp=0)
        # Extract learned parameters from result
        self.theta = result[0]
        # Compute variance if requested.
        if variance_mode is not None:
            variance = self._compute_variance(X_with_intercept, y, self.theta, weights, offsets, variance_mode)
        else:
            variance = None
        return result, variance

    def predict_proba(self, X, offsets=None, custom_theta=None, return_logits=False):
        """
        Predict binary logistic regression probabilities/logits using a trained model
        :param X:               a dense or sparse matrix of dimensions (n x d), where n is the number of samples,
                                and d is the number of features
        :param y:               vector of binary sample labels; of dimensions (n x 1)  where n is the number of samples
        :param offsets:         vector of sample offsets; of dimensions (n x 1)  where n is the number of samples
        :param custom_theta:    optional weight vector of dimensions (d x 1), overrides learned weights if provided
        :param return_logits    return probabilities if set to True, logits otherwise
        :return:    probabilities/logits
        """
        # Assert X and offsets are compatible dimension-wise
        if offsets is None:
            offsets = np.zeros(self._get_number_of_samples(X))
        assert (X.shape[0] == offsets.shape[0])

        custom_theta = self.theta if custom_theta is None else custom_theta
        if custom_theta is None:
            raise Exception("Custom weights must be provided if attempting inference on untrained model")
        X_with_intercept = self._add_column_of_ones(X) if self.has_intercept else X

        return self._predict(custom_theta, X_with_intercept, offsets, return_logits)

    def compute_metrics(self, X, y, offsets=None, custom_theta=None):
        """
        Compute metrics using a trained binary logistic regression model
        :param X:               a dense or sparse matrix of dimensions (n x d), where n is the number of samples,
                                and d is the number of features
        :param y:               vector of binary sample labels; of dimensions (n x 1)  where n is the number of samples
        :param offsets:         vector of sample offsets; of dimensions (n x 1)  where n is the number of samples
        :param custom_theta:    optional weight vector of dimensions (d x 1), overrides learned weights if provided
        :return:    a dictionary of metrics
        """

        # Assert X , y and offsets are compatible dimension-wise
        if offsets is None:
            offsets = np.zeros(X.shape[0])
        assert (X.shape[0] == y.shape[0] == offsets.shape[0])

        custom_theta = self.theta if custom_theta is None else custom_theta
        if custom_theta is None:
            raise Exception("Custom weights must be provided if attempting metrics computation on untrained model")

        # Run prediction and calculate AUC
        pred = self.predict_proba(X, offsets, custom_theta)
        fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        return {"auc": auc}
