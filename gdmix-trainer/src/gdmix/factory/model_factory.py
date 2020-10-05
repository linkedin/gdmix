import logging

from gdmix.params import Params

from gdmix.models.estimator.fixed_effect_detext_estimator_model import FixedEffectDetextEstimatorModel
from gdmix.models.custom.random_effect_lr_lbfgs_model import RandomEffectLRLBFGSModel
from gdmix.models.custom.fixed_effect_lr_lbfgs_model import FixedEffectLRModelLBFGS
from gdmix.util import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelFactory:
    """
    Provider class for creating model instances and dependencies

    NOTE - for now, only Estimator-based linear models are supported. In the future, the factory will also
    accept model type as an input parameter
    """

    @staticmethod
    def get_model(base_training_params: Params, raw_model_params):
        """
        Create driver and associated dependencies, based on type. Only linear, estimator-based models supported
        for now
        :param base_training_params:      Parsed base training parameters common to all models. This could including
        path to training data, validation data, metadata file path, learning rate etc.
        :param raw_model_params:          Raw model parameters, representing model-specific requirements. For example, a
        CNN might expose filter_size as a parameter, a text-based model might expose the size it's word embedding matrix
        as a parameter
        :return:                Model instances
        """
        model_type = base_training_params.model_type
        driver_type = base_training_params.stage
        logger.info(f"Instantiating {model_type} model and driver")
        if model_type == constants.LOGISTIC_REGRESSION:
            if driver_type == constants.FIXED_EFFECT:
                logger.info("Choosing Scipy-LBFGS FE model")
                model = FixedEffectLRModelLBFGS(
                    raw_model_params=raw_model_params, base_training_params=base_training_params)
            else:
                logger.info("Choosing Scipy RE model")
                model = RandomEffectLRLBFGSModel(raw_model_params=raw_model_params)
        elif model_type == constants.DETEXT:
            model = FixedEffectDetextEstimatorModel(raw_model_params=raw_model_params)
        else:
            raise Exception(f"Unknown training models {model_type}")
        return model
