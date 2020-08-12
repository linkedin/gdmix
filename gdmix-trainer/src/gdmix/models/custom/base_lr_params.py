from dataclasses import dataclass
from typing import List

from gdmix.params import DeprecatedApis


@dataclass
class LRParams(DeprecatedApis):
    """Base logistic regression parameters"""

    # Input / output files or directories
    train_data_path: str  # Path of directory holding only training data files.
    validation_data_path: str  # "Path of directory holding only data files for in-line validation."
    metadata_file: str  # Path to metadata.
    model_output_dir: str  # Model output directory.

    # Column names in the dataset
    feature_bags: List[str]  # Feature bag names that used for training and scoring.

    # Arguments for model export
    feature_file: str  # Feature file for model exporting.

    # Optimizer related parameters
    regularize_bias: bool = True  # Boolean for L2 regularization of bias term.
    l2_reg_weight: float = 1.0  # Weight of L2 regularization for each feature bag.
    lbfgs_tolerance: float = 1e-12  # LBFGS tolerance.
    num_of_lbfgs_curvature_pairs: int = 10  # Number of curvature pairs for LBFGS training.
    num_of_lbfgs_iterations: int = 100  # Number of LBFGS iterations.

    offset: str = "offset"  # Score from previous model.

    # Dataset parameters
    batch_size: int = 16
    data_format: str = "tfrecord"

    def __post_init__(self):
        assert self.batch_size > 0, "Batch size must be positive number"
