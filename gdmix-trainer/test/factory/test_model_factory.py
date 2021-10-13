import tensorflow as tf

from gdmix.factory.model_factory import ModelFactory
from gdmix.models.custom.fixed_effect_lr_lbfgs_model import FixedEffectLRModelLBFGS
from gdmix.models.custom.random_effect_lr_lbfgs_model import RandomEffectLRLBFGSModel
from gdmix.util import constants
from drivers.test_helper import setup_fake_base_training_params, setup_fake_raw_model_params


class TestModelFactory(tf.test.TestCase):
    """
    Test ModelFactory
    """

    def setUp(self):
        self.model_params = setup_fake_raw_model_params()

    def test_fixed_effect_logistic_regression_lbfgs_model_creation(self):
        fe_model = ModelFactory.get_model(
            base_training_params=setup_fake_base_training_params(training_stage=constants.FIXED_EFFECT,
                                                                 model_type=constants.LOGISTIC_REGRESSION),
            raw_model_params=self.model_params)
        # Assert the type of model
        self.assertIsInstance(fe_model, FixedEffectLRModelLBFGS)

    def test_fixed_effect_linear_regression_lbfgs_model_creation(self):
        fe_model = ModelFactory.get_model(
            base_training_params=setup_fake_base_training_params(training_stage=constants.FIXED_EFFECT,
                                                                 model_type=constants.LINEAR_REGRESSION),
            raw_model_params=self.model_params)
        # Assert the type of model
        self.assertIsInstance(fe_model, FixedEffectLRModelLBFGS)

    def test_random_effect_custom_logistic_regression_model_creation(self):
        re_model = ModelFactory.get_model(
            base_training_params=setup_fake_base_training_params(training_stage=constants.RANDOM_EFFECT,
                                                                 model_type=constants.LOGISTIC_REGRESSION),
            raw_model_params=self.model_params)
        self.assertIsInstance(re_model, RandomEffectLRLBFGSModel)
