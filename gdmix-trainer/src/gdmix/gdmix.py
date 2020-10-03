import logging
import sys

from gdmix.factory.driver_factory import DriverFactory
from gdmix.params import Params, SchemaParams
from gdmix.util import constants

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run(args):
    """
    Parse CMD line arguments, instantiate Driver and Model object and handover control to Driver
    :param args: command line arguments
    :return: None
    """
    # Parse base training parameters that are required for all models. For other arguments, the
    # Driver delegates parsing to the specific model it encapsulates
    params = Params.__from_argv__(args, error_on_unknown=False)
    schema_params = SchemaParams.__from_argv__(args, error_on_unknown=False)

    # Log parsed base training parameters
    logger.info(f"Parsed schema params amd gdmix args (params): {params}")

    # Instantiate appropriate driver, encapsulating a specific model
    driver = DriverFactory.get_driver(base_training_params=params, raw_model_params=args)

    # Run driver to either [1] train, [2] run evaluation or [3] export model
    if params.action == constants.ACTION_TRAIN:
        driver.run_training(schema_params=schema_params, export_model=True)
    elif params.action == constants.ACTION_INFERENCE:
        driver.run_inference(schema_params=schema_params)
    else:
        raise Exception(f"Unsupported action {params.action}")


if __name__ == '__main__':
    run(sys.argv)
