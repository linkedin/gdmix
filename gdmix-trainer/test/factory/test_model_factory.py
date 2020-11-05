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
        """
        Set the model.

        Args:
            self: (todo): write your description
        """
        self.params = setup_fake_base_training_params()
        self.model_params = setup_fake_raw_model_params()

    def test_fixed_effect_lr_lbfgs_model_creation(self):
        """
        Test if training learning learning learning rate.

        Args:
            self: (todo): write your description
        """
        fe_model = ModelFactory.get_model(
            base_training_params=setup_fake_base_training_params(training_stage=constants.FIXED_EFFECT,
                                                                 model_type=constants.LOGISTIC_REGRESSION),
            raw_model_params=self.model_params)
        # Assert the type of model
        self.assertIsInstance(fe_model, FixedEffectLRModelLBFGS)

    def test_random_effect_custom_lr_model_creation(self):
        """
        Sets the model learning learning rate.

        Args:
            self: (todo): write your description
        """
        re_model = ModelFactory.get_model(
            base_training_params=setup_fake_base_training_params(training_stage=constants.RANDOM_EFFECT,
                                                                 model_type=constants.LOGISTIC_REGRESSION),
            raw_model_params=self.model_params)
        self.assertIsInstance(re_model, RandomEffectLRLBFGSModel)
