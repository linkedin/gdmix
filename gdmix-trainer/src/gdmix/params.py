from dataclasses import dataclass
from typing import Optional

from gdmix.util import constants
from smart_arg import arg_suite, LateInit

_ACTIONS = (constants.ACTION_INFERENCE, constants.ACTION_TRAIN)
_STAGES = (constants.FIXED_EFFECT, constants.RANDOM_EFFECT)
_MODEL_TYPES = (constants.LOGISTIC_REGRESSION, constants.DETEXT)


@dataclass
class GDMixParams:
    action: str = _ACTIONS[1]  # Train or inference.
    __action = {"choices": _ACTIONS}
    stage: str = _STAGES[0]  # Fixed or random effect.
    __stage = {"choices": _STAGES}
    model_type: str = _MODEL_TYPES[0]  # The model type to train, e.g, logistic regression, detext, etc.
    __model_type = {"choices": _MODEL_TYPES}

    # Input / output directories
    training_score_dir: Optional[str] = None  # Path to the prediction score directory of the training data..
    validation_score_dir: Optional[str] = None  # Path to the prediction score directory of the validation data..

    # Driver arguments for random effect training
    partition_list_file: Optional[str] = None  # File containing a list of all the partition ids, for random effect only

    def __post_init__(self):
        assert self.action in _ACTIONS, f"Action: {self.action} must be in {_ACTIONS}"
        assert self.stage in _STAGES, f"Stage: {self.stage} must be in {_STAGES}"
        assert self.model_type in _MODEL_TYPES, f"Model type: {self.model_type} must be in {_MODEL_TYPES}"


@arg_suite
@dataclass
class SchemaParams:
    # Schema names
    uid_column_name: str = LateInit  # Unique id column name in the train/validation data.
    weight_column_name: Optional[str] = None  # weight column name in the train/validation data.
    label_column_name: Optional[str] = None  # Label column name in the train/validation data.
    prediction_score_column_name: Optional[str] = None  # Prediction score column name in the generated result file.
    prediction_score_per_coordinate_column_name: str = "predictionScorePerCoordinate"  # Column name of the prediction score without the offset.


@arg_suite
@dataclass
class Params(GDMixParams, SchemaParams):
    """GDMix Driver"""

    def __post_init__(self):
        super().__post_init__()
        assert (self.action == constants.ACTION_TRAIN and self.label_column_name) or \
               (self.action == constants.ACTION_INFERENCE and self.prediction_score_column_name)
