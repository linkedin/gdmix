import logging

from gdmix.drivers.fixed_effect_driver import FixedEffectDriver
from gdmix.drivers.random_effect_driver import RandomEffectDriver
from gdmix.factory.model_factory import ModelFactory
from gdmix.util import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DriverFactory:
    """
    Provider class for creating driver and dependencies

    NOTE - for now, only Estimator-based linear models are supported. In the future, the factory will also
    accept model type as an input parameter
    """

    @staticmethod
    def get_driver(base_training_params, raw_model_params):
        """
        Create driver and associated dependencies, based on type. Only linear, estimator-based models supported
        for now
        :param base_training_params:      Parsed base training parameters common to all models. This could including
        path to training data, validation data, metadata file path, learning rate etc.
        :param raw_model_params:          Raw model parameters, representing model-specific requirements. For example, a
        CNN might expose filter_size as a parameter, a text-based model might expose the size it's word embedding matrix
        as a parameter
        :return:            Fixed or Random effect driver
        """

        driver = DriverFactory.drivers[base_training_params.stage]
        model = ModelFactory.get_model(base_training_params, raw_model_params)
        logger.info(f"Instantiating model {model} and driver {driver}")
        return driver(base_training_params=base_training_params, model=model)

    drivers = {constants.FIXED_EFFECT: FixedEffectDriver, constants.RANDOM_EFFECT: RandomEffectDriver}
