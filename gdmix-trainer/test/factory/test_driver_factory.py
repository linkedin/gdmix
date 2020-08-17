import tensorflow as tf
from gdmix.factory.driver_factory import DriverFactory

from gdmix.drivers.fixed_effect_driver import FixedEffectDriver
from gdmix.drivers.random_effect_driver import RandomEffectDriver
from gdmix.util import constants
from drivers.test_helper import set_fake_tf_config, setup_fake_base_training_params, setup_fake_raw_model_params


class TestDriverFactory(tf.test.TestCase):
    """
    Test DriverFactory
    """

    def setUp(self):
        self.task_type = "worker"
        self.worker_index = 0
        self.num_workers = 5
        set_fake_tf_config(task_type=self.task_type, worker_index=self.worker_index)
        self.params = setup_fake_base_training_params()
        self.model_params = setup_fake_raw_model_params()

    def test_fixed_effect_driver_wiring(self):
        fe_driver = DriverFactory.get_driver(
            base_training_params=setup_fake_base_training_params(constants.FIXED_EFFECT),
            raw_model_params=self.model_params)
        # Assert the type of driver
        self.assertIsInstance(fe_driver, FixedEffectDriver)

    def test_random_effect_driver_wiring(self):
        re_driver = DriverFactory.get_driver(
            base_training_params=setup_fake_base_training_params(constants.RANDOM_EFFECT),
            raw_model_params=self.model_params)
        # Assert the type of driver
        self.assertIsInstance(re_driver, RandomEffectDriver)
