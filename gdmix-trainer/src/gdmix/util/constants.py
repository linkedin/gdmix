ACTION = "action"
STAGE = "stage"
LABEL_COLUMN_NAME = "label_column_name"
MODEL_TYPE = "model_type"
LBFGS = "lbfgs"
PREDICTION_SCORE_COLUMN_NAME = "prediction_score_column_name"
PREDICTION_SCORE_PER_COORDINATE_COLUMN_NAME = "prediction_score_per_coordinate_column_name"
OFFSET_COLUMN_NAME = "offset_column_name"
UID_COLUMN_NAME = "uid_column_name"
WEIGHT_COLUMN_NAME = "weight_column_name"
FEATURE_BAG = "feature_bag"
PARTITION_ENTITY = "partition_entity"
TRAINING_DATA_DIR = "training_data_dir"
VALIDATION_DATA_DIR = "validation_data_dir"
OUTPUT_MODEL_DIR = "output_model_dir"
DATA_FORMAT = "data_format"
COPY_TO_LOCAL = "copy_to_local"

FEATURE_FILE = "feature_file"

# Training parameters-related constants
BATCH_SIZE = "batch_size"
DELAYED_EXIT_IN_SECONDS = "delayed_exit_in_seconds"
ENABLE_LOCAL_INDEXING = "enable_local_indexing"
L2_REG_WEIGHT = "l2_reg_weight"
LBFGS_TOLERANCE = "lbfgs_tolerance"
MAX_TRAINING_QUEUE_SIZE = "max_training_queue_size"
NUM_OF_CONSUMERS = "num_of_consumers"
NUM_OF_LBFGS_CURVATURE_PAIRS = "num_of_lbfgs_curvature_pairs"
NUM_OF_LBFGS_ITERATIONS = "num_of_lbfgs_iterations"
REGULARIZE_BIAS = "regularize_bias"
TRAINING_QUEUE_TIMEOUT_IN_SECONDS = "training_queue_timeout_in_seconds"

AUC = "auc",
ACCURACY = "accuracy"
ACTIVE = "active"
PASSIVE = "passive"

TRAINING_SCORE_DIR = "training_score_dir"
VALIDATION_SCORE_DIR = "validation_score_dir"
ACTIVE_TRAINING_OUTPUT_FILE = "active_training_output_file"
PASSIVE_TRAINING_OUTPUT_FILE = "passive_training_output_file"
TFRECORD_GLOB_PATTERN = "*.tfrecord"
VALIDATION_OUTPUT_FILE = "validation_output_file"
PASSIVE_TRAINING_DATA_DIR = "passive_training_data_dir"
RANDOM_EFFECT = "random_effect"
FIXED_EFFECT = "fixed_effect"

# Constants for random effect raining
MODEL_IDS_DIR = "model_ids_dir"
PARTITION_INDEX = "partition_index"
PARTITION_LIST_FILE = "partition_list_file"

# String constants related to execution context
IS_CHIEF = "is_chief"
NUM_SHARDS = "num_shards"
SHARD_INDEX = "shard_index"
NUM_EPOCHS = "num_epochs"
NUM_WORKERS = "num_workers"
WORKER = "worker"
CLUSTER_SPEC = "cluster_spec"
TASK_INDEX = "task_index"
TASK_TYPE = "task_type"
TASK_TYPE_CHIEF = "chief"
TASK_TYPE_WORKER = "worker"
TF_CONFIG = "TF_CONFIG"

# Dataset constants
DATASET_MODULE = "dataset_module"
DATASET_CREATOR = "dataset_creator"
TFRECORD = "tfrecord"
INPUT_DIR = "input_dir"
METADATA_FILE = "metadata_file"
ACTION_INFERENCE = "inference"
ACTION_TRAIN = "train"

# Supported models
LINEAR_REGRESSION = "linear_regression"
LOGISTIC_REGRESSION = "logistic_regression"
DETEXT = "detext"

# Variance computation
SIMPLE = "simple"
FULL = "full"
