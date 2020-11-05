from dataclasses import dataclass
from typing import Optional


@dataclass
class LRParams:
    """Base logistic regression parameters"""

    # Input / output files or directories
    metadata_file: str  # Path to metadata.
    output_model_dir: str  # Model output directory.
    training_data_dir: Optional[str] = None  # Path of directory holding only training data files.
    validation_data_dir: Optional[str] = None  # "Path of directory holding only data files for in-line validation."
    # Column names in the dataset
    feature_bag: Optional[str] = None  # Feature bag name that is used for training and scoring.

    # Arguments for model export
    feature_file: Optional[str] = None  # Feature file for model exporting.

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
