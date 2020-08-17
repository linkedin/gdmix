import argparse
from gdmix.util import constants
from gdmix.util.io_utils import str2bool

parser = argparse.ArgumentParser(description="The base logistic regression argparser", add_help=False)

# Input / output files or directories
parser.add_argument("--" + constants.TRAIN_DATA_PATH, type=str, required=False,
                    help="Path of directory holding only training data files.")
parser.add_argument("--" + constants.VALIDATION_DATA_PATH, type=str, required=False,
                    help="Path of directory holding only data files for in-line validation.")
parser.add_argument("--" + constants.METADATA_FILE, type=str, required=False,
                    help="Path to metadata.")
parser.add_argument("--" + constants.MODEL_OUTPUT_DIR, type=str, required=False,
                    help="Model output directory.")

# Column names in the dataset
parser.add_argument("--" + constants.OFFSET, type=str, required=False, default="offset",
                    help="Score from previous model.")
parser.add_argument("--" + constants.FEATURE_BAGS, type=str, required=False,
                    help="Feature bag names that used for training and scoring.")

# Dataset parameters
parser.add_argument("--" + constants.BATCH_SIZE, type=int, required=False, default=16,
                    help="Batch size for training and inline evaluation.")
parser.add_argument("--" + constants.DATA_FORMAT, type=str, required=False, default=constants.TFRECORD,
                    help="avro or tfrecord dataset")

# Arguments for model export
parser.add_argument("--" + constants.FEATURE_FILE, type=str, required=False,
                    help="Feature file")

# Optimizer related parameters
parser.add_argument("--" + constants.REGULARIZE_BIAS, type=str2bool, nargs='?', const=True, required=False,
                    default=False, help="Boolean for L2 regularization of bias term.")
parser.add_argument("--" + constants.L2_REG_WEIGHT, type=float, required=False, default=1.0,
                    help="Weight of L2 regularization for each feature bag.")
parser.add_argument("--" + constants.LBFGS_TOLERANCE, type=float, required=False, default=1e-12,
                    help="LBFGS tolerance.")
parser.add_argument("--" + constants.NUM_OF_LBFGS_CURVATURE_PAIRS, type=int, required=False, default=10,
                    help="Number of curvature pairs for LBFGS training.")
parser.add_argument("--" + constants.NUM_OF_LBFGS_ITERATIONS, type=int, required=False, default=100,
                    help="Number of LBFGS iterations.")
