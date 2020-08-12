import traceback
import warnings
from dataclasses import dataclass
from typing import Optional

from gdmix.util import constants
from smart_arg import arg_suite


class DeprecatedApis:
    def __getitem__(self, item):
        warnings.warn(f"dict style access (xxx[{item!r}] is deprecated, use xxx.{item} instead at:\n {''.join(traceback.format_stack())}")
        return getattr(self, item)


@dataclass
class GDMixParams(DeprecatedApis):
    ACTIONS = (constants.ACTION_INFERENCE, constants.ACTION_TRAIN)
    STAGES = (constants.FIXED_EFFECT, constants.RANDOM_EFFECT)
    MODEL_TYPES = (constants.LOGISTIC_REGRESSION, constants.DETEXT)

    action: str = ACTIONS[1]  # Train or inference.
    __action = {"choices": ACTIONS}
    stage: str = STAGES[0]  # Fixed or random effect.
    __stage = {"choices": STAGES}
    model_type: str = MODEL_TYPES[0]  # The model type to train, e.g, logistic regression, detext, etc.
    __model_type = {"choices": MODEL_TYPES}

    # Input / output files or directories
    training_output_dir: Optional[str] = None  # Training output directory.
    validation_output_dir: Optional[str] = None  # Validation output directory.

    # Driver arguments for random effect training
    partition_list_file: Optional[str] = None  # File containing a list of all the partition ids, for random effect only

    def __post_init__(self):
        assert self.action in self.ACTIONS, f"Action: {self.action} must be in {self.ACTIONS}"
        assert self.stage in self.STAGES, f"Stage: {self.stage} must be in {self.STAGES}"
        assert self.model_type in self.MODEL_TYPES, f"Model type: {self.model_type} must be in {self.MODEL_TYPES}"


@dataclass
class SchemaParams(DeprecatedApis):
    # Schema names
    sample_id: str  # Sample id column name in the input file.
    sample_weight: Optional[str] = None  # Sample weight column name in the input file.
    label: Optional[str] = None  # Label column name in the train/validation file.
    prediction_score: Optional[str] = None  # Prediction score column name in the generated result file.
    prediction_score_per_coordinate: str = "predictionScorePerCoordinate"  # ColumnName of the prediction score without the offset.


@arg_suite
@dataclass
class Params(GDMixParams, SchemaParams):
    """GDMix Driver"""

    def __post_init__(self):
        super().__post_init__()
        assert (self.action == constants.ACTION_TRAIN and self.label) or (self.action == constants.ACTION_INFERENCE and self.prediction_score)
