import os

from gdmix.params import SchemaParams, Params
from gdmix.util import constants


def set_fake_tf_config(task_type, worker_index):
    """
    Set up fake TF_CONFIG environment variable
    :param task_type:       worker or evaluator
    :param worker_index:    index of node
    :return:    None
    """
    os.environ[constants.TF_CONFIG] = str(
        {"task": {"type": str(task_type), "index": worker_index}, "cluster":
            {"worker": ["node1.example.com:25304",
                        "node2.example.com:32879",
                        "node3.example.com:8068",
                        "node4.example.com:25949",
                        "node5.example.com:28685"],
             "evaluator": ["node6.example.com:21243"]}}).replace("'", '"')


def setup_fake_base_training_params(training_stage=constants.FIXED_EFFECT,
                                    model_type=constants.LOGISTIC_REGRESSION):
    """
    Set up fake parameter dict for testing
    :return: fake parameter dict
    """
    params = {constants.ACTION: "train",
              constants.STAGE: training_stage,
              constants.MODEL_TYPE: model_type,
              constants.TRAINING_OUTPUT_DIR: "dummy_training_output_dir",
              constants.VALIDATION_OUTPUT_DIR: "dummy_validation_output_dir",

              constants.PARTITION_LIST_FILE: os.path.join(os.getcwd(), "test/resources/metadata",
                                                          "partition_list.txt"),

              constants.SAMPLE_ID: "uid",
              constants.SAMPLE_WEIGHT: "weight",
              constants.LABEL: "response",
              constants.PREDICTION_SCORE: "predictionScore",
              constants.PREDICTION_SCORE_PER_COORDINATE: "predictionScorePerCoordinate"
              }
    params = Params(**params)
    object.__delattr__(params, '__frozen__')  # Allow the test code to mutate the params.
    return params


def setup_fake_raw_model_params(training_stage=constants.FIXED_EFFECT):
    """
    Setup the raw stage model params.

    Args:
        training_stage: (todo): write your description
        constants: (todo): write your description
        FIXED_EFFECT: (str): write your description
    """
    raw_model_params = [f"--{constants.SAMPLE_ID}", "uid", f"--{constants.SAMPLE_WEIGHT}", "weight",
                        f"--{constants.FEATURE_BAGS}", "global",
                        f"--{constants.TRAIN_DATA_PATH}", os.path.join(os.getcwd(), "test/resources/train"),
                        f"--{constants.VALIDATION_DATA_PATH}",
                        os.path.join(os.getcwd(), "test/resources/validate"),
                        f"--{constants.MODEL_OUTPUT_DIR}", "dummy_model_output_dir",
                        f"--{constants.METADATA_FILE}",
                        os.path.join(os.getcwd(), "test/resources/fe_lbfgs/metadata/tensor_metadata.json"),
                        f"--{constants.FEATURE_FILE}", "test/resources/fe_lbfgs/featureList/global",
                        ]
    if training_stage == constants.RANDOM_EFFECT:
        raw_model_params.append(f"--{constants.FEATURE_BAGS}")
        raw_model_params.append("per_member")
        raw_model_params.append(f"--{constants.OFFSET}")
        raw_model_params.append("offset")
    return raw_model_params


def setup_fake_schema_params():
    """
    Setup params for the fake fake params.

    Args:
    """
    return SchemaParams(**{constants.SAMPLE_ID: "uid",
                           constants.SAMPLE_WEIGHT: "weight",
                           constants.LABEL: "response",
                           constants.PREDICTION_SCORE: "predictionScore",
                           constants.PREDICTION_SCORE_PER_COORDINATE: "predictionScorePerCoordinate"
                           })
