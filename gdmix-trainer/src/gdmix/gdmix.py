import argparse
import sys
import logging
from gdmix.util import constants
from gdmix.factory.driver_factory import DriverFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def add_gdmix_params():
    parser = argparse.ArgumentParser()

    parser.add_argument("--" + constants.ACTION, type=str, required=False, default=constants.ACTION_TRAIN,
                        help="Train or inference.")
    parser.add_argument("--" + constants.STAGE, type=str, required=False, default=constants.FIXED_EFFECT,
                        help="Fixed or random effect.")
    parser.add_argument("--" + constants.MODEL_TYPE, type=str, required=True, default=constants.LOGISTIC_REGRESSION,
                        help="The model type to train, e.g, logistic regression, detext, etc.")
    # Input / output files or directories
    parser.add_argument("--" + constants.TRAINING_OUTPUT_DIR, type=str, required=False,
                        help="Training output directory.")
    parser.add_argument("--" + constants.VALIDATION_OUTPUT_DIR, type=str, required=False,
                        help="Validation output directory.")

    # Driver arguments for random effect training
    parser.add_argument("--" + constants.PARTITION_LIST_FILE, type=str, required=False,
                        help="File containing a list of all the partition ids, for random effect only")

    return parser


def validate_base_training_params(params):
    assert params[constants.ACTION] in set([constants.ACTION_INFERENCE, constants.ACTION_TRAIN]), \
        "Action must be either train or inference"
    assert params[constants.STAGE] in set([constants.FIXED_EFFECT, constants.RANDOM_EFFECT]), \
        "Stage must be either fixed_effect or random_effect"
    assert params[constants.MODEL_TYPE] in set([constants.LOGISTIC_REGRESSION, constants.DETEXT]), \
        "Model type must be either logistic_regression or detext"


def add_schema_params():
    parser = argparse.ArgumentParser()
    # Schema names
    parser.add_argument("--" + constants.SAMPLE_ID, type=str, required=False,
                        help="Sample id column name in the input file.")
    parser.add_argument("--" + constants.SAMPLE_WEIGHT, type=str, required=False,
                        help="Sample weight column name in the input file.")
    parser.add_argument("--" + constants.LABEL, type=str, required=False, default="label",
                        help="Label column name in the train/validation file.")
    parser.add_argument("--" + constants.PREDICTION_SCORE, type=str, required=False, default="predictionScore",
                        help="Prediction score column name in the generated result file.")
    parser.add_argument("--" + constants.PREDICTION_SCORE_PER_COORDINATE, type=str, required=False,
                        default="predictionScorePerCoordinate",
                        help="ColumnName of the prediction score without the offset.")
    return parser


def validate_schema_params(training_params, schema_params):
    assert schema_params[constants.SAMPLE_ID] is not None, "Sample_ID is needed"
    if training_params[constants.ACTION] == constants.ACTION_TRAIN:
        assert schema_params[constants.LABEL] is not None
    if training_params[constants.ACTION] == constants.ACTION_INFERENCE:
        assert schema_params[constants.PREDICTION_SCORE] is not None


def run(args):
    """
    Parse CMD line arguments, instantiate Driver and Model object and handover control to Driver
    :param args: command line arguments
    :return: None
    """
    # Parse base training parameters that are required for all models. For other arguments, the
    # Driver delegates parsing to the specific model it encapsulates
    schema_parser = add_schema_params()
    gdmix_parser = add_gdmix_params()
    schema_params, other_args = schema_parser.parse_known_args(args)
    base_training_params, other_args = gdmix_parser.parse_known_args(other_args)
    schema_params = vars(schema_params)
    base_training_params = vars(base_training_params)

    # Log parsed base training parameters
    logger.info("Parsed schema params: {}".format(schema_params))
    logger.info("Parsed gdmix args (params): {}".format(base_training_params))
    if other_args and len(other_args) > 0:
        logger.warning("Other args: {}".format(other_args))

    validate_schema_params(base_training_params, schema_params)
    validate_base_training_params(base_training_params)

    # Instantiate appropriate driver, encapsulating a specific model
    driver = DriverFactory.get_driver(base_training_params=base_training_params,
                                      raw_model_params=other_args)

    # Run driver to either [1] train, [2] run evaluation or [3] export model
    if base_training_params[constants.ACTION] == constants.ACTION_TRAIN:
        driver.run_training(schema_params=schema_params, export_model=True)
    elif base_training_params[constants.ACTION] == constants.ACTION_INFERENCE:
        driver.run_inference(schema_params=schema_params)
    else:
        raise Exception("Unsupported action")


if __name__ == '__main__':
    run(sys.argv)
