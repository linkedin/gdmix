import dataclasses
import itertools
import logging
import os
from functools import partial
from multiprocessing import Pool, current_process
from typing import Optional

import fastavro
import numpy as np
import tensorflow as tf
from smart_arg import arg_suite

from gdmix.io.dataset_metadata import DatasetMetadata
from gdmix.io.input_data_pipeline import per_entity_grouped_input_fn
from gdmix.models.api import Model
from gdmix.models.custom.base_lr_params import LRParams
from gdmix.models.custom.binary_logistic_regression import BinaryLogisticRegressionTrainer
from gdmix.models.custom.scipy.job_consumers import InferenceJobConsumer, TrainingJobConsumer, TrainingResult, prepare_jobs
from gdmix.util import constants
from gdmix.util.io_utils import read_json_file, export_linear_model_to_avro, get_feature_map, name_term_to_string, batched_write_avro, \
    get_inference_output_avro_schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tf.compat.v1.disable_eager_execution()


@arg_suite
@dataclasses.dataclass
class REParams(LRParams):
    """Scipy-based custom logistic regression random effect model."""
    # Dataset column names
    partition_entity: Optional[str] = None  # Partition entity name.
    # Training parameters for custom Scipy-based training
    enable_local_indexing: bool = False  # Enable local indexing for model training
    max_training_queue_size: int = 10  # Maximum size of training job queue
    training_queue_timeout_in_seconds: int = 300  # Training queue put timeout in seconds.
    num_of_consumers: int = 2  # Number of consumer processes that will train RE models in parallel.

    def __post_init__(self):
        assert self.max_training_queue_size > self.num_of_consumers, "queue size limit must be larger than the number of consumers"


class RandomEffectLRLBFGSModel(Model):
    """
    Scipy-based custom logistic regression random effect model.

    Supports training entity-based models. Several models can be trained in parallel on multiple processes.
    """
    def __init__(self, raw_model_params):
        super(RandomEffectLRLBFGSModel, self).__init__(raw_model_params)
        self.model_params: REParams = self._parse_parameters(raw_model_params)
        self.checkpoint_path = os.path.join(self.model_params.model_output_dir)
        self.metadata_file = self.model_params.metadata_file
        # If TRAIN_DATA_PATH is set, initialize active/passive training data path, else set to None
        if self.model_params.train_data_path is not None:
            self.training_data_path = os.path.join(self.model_params.train_data_path, constants.ACTIVE)
            self.passive_training_data_path = os.path.join(self.model_params.train_data_path, constants.PASSIVE)
        else:
            self.training_data_path = None
            self.passive_training_data_path = None
        self.validation_data_path = self.model_params.validation_data_path

    def train(self, training_data_path, validation_data_path, metadata_file, checkpoint_path, execution_context, schema_params):
        logger.info("Kicking off random effect custom LR training")
        self._action(constants.ACTION_TRAIN, (training_data_path, validation_data_path), metadata_file, checkpoint_path, execution_context, schema_params)

    def predict(self, output_dir, input_data_path, metadata_file, checkpoint_path, execution_context, schema_params):
        logger.info(f"Running inference on dataset : {input_data_path}, results to be written to path : {output_dir}")
        self._action(constants.ACTION_INFERENCE, (output_dir, input_data_path), metadata_file, checkpoint_path, execution_context, schema_params)

    def _action(self, action, action_context, metadata_file, checkpoint_path, execution_context, schema_params):
        partition_index = execution_context[constants.PARTITION_INDEX]
        # Read tensor metadata
        metadata = read_json_file(metadata_file)
        tensor_metadata = DatasetMetadata(metadata)
        # Extract number of features. NOTE - only one feature bag is supported
        num_features = next(filter(lambda x: x.name == self.model_params.feature_bags[0], tensor_metadata.get_features())).shape[0]
        logger.info(f"Found {num_features} features in feature bag {self.model_params.feature_bags[0]}")
        assert num_features > 0, "number of features must > 0"

        with Pool(self.model_params.num_of_consumers, initializer=lambda: logger.info(f"Process {current_process()} ready to work!")) as pool:
            avro_filename = f"part-{partition_index:05d}.avro"
            if action == constants.ACTION_INFERENCE:
                output_dir, input_data_path = action_context
                model_weights = self._load_weights(os.path.join(checkpoint_path, avro_filename))
                self._predict(pool=pool, input_path=input_data_path, metadata=metadata, tensor_metadata=tensor_metadata, metadata_file=metadata_file,
                              output_file=os.path.join(output_dir, avro_filename), model_weights=model_weights,
                              schema_params=schema_params, use_local_index=True, num_features=num_features)
            elif action == constants.ACTION_TRAIN:
                training_data_path, validation_data_path = action_context
                model_file = os.path.join(self.model_params.model_output_dir, avro_filename)
                # load initial model if available
                model_weights = self._load_weights(model_file, True)
                # Train the model
                model_weights = self._train(pool, training_data_path, metadata_file, model_weights, num_features, schema_params, model_file)

                # shorthand for self._predict
                predict = partial(self._predict, use_local_index=self.model_params.enable_local_indexing, metadata=metadata, tensor_metadata=tensor_metadata,
                                  pool=pool, schema_params=schema_params, num_features=num_features, metadata_file=metadata_file, model_weights=model_weights)
                # Run inference on validation set
                o = execution_context.get(constants.VALIDATION_OUTPUT_FILE, None)
                o and predict(input_path=validation_data_path, output_file=o)

                # Run inference on active training set
                o = execution_context.get(constants.ACTIVE_TRAINING_OUTPUT_FILE, None)
                o and predict(input_path=training_data_path, output_file=o)

                # Run inference on passive training set
                i, o = execution_context.get(constants.PASSIVE_TRAINING_DATA_PATH, None), execution_context.get(constants.PASSIVE_TRAINING_OUTPUT_FILE, None)
                i and o and predict(input_path=i, output_file=o)
            else:
                raise ValueError(f"Invalid action {action!r}.")

    def _train(self, pool, input_path, metadata_file, model_weights: dict, num_features, schema_params, output_model_file):
        logger.info(f"Start training with {f'loaded {len(model_weights)} previous models' if model_weights else 'zeros'} as the model initial value.")
        lr_model = BinaryLogisticRegressionTrainer(regularize_bias=self.model_params.regularize_bias, lambda_l2=self.model_params.l2_reg_weight,
                                                   precision=self.model_params.lbfgs_tolerance / np.finfo(float).eps,
                                                   num_lbfgs_corrections=self.model_params.num_of_lbfgs_curvature_pairs,
                                                   max_iter=self.model_params.num_of_lbfgs_iterations)
        consumer = TrainingJobConsumer(lr_model, name=input_path)
        results = self._pooled_action(pool, consumer, input_path, schema_params, model_weights, num_features, metadata_file,
                                      self.model_params.enable_local_indexing)
        model_weights.update(results)
        logger.info(f"{len(model_weights)} models in total after training/refreshing.")
        # Dump results to model output directory.
        if self.model_params.feature_file:
            self._save_model(output_model_file, model_coefficients=model_weights, num_features=num_features, feature_file=self.model_params.feature_file)
        else:
            logger.info("Both feature file and avro model output directory required to export model. Skipping export")
        return model_weights

    def _predict(self, pool, input_path, metadata, tensor_metadata, output_file, schema_params, num_features, metadata_file, model_weights, use_local_index):
        logger.info(f"Start inference for {input_path}.")
        # Create LR model object for inference
        lr_model = BinaryLogisticRegressionTrainer(regularize_bias=True, lambda_l2=self.model_params.l2_reg_weight)
        consumer = InferenceJobConsumer(lr_model, num_features, schema_params, use_local_index, name=input_path)

        results = self._pooled_action(pool, consumer, input_path, schema_params, model_weights, num_features, metadata_file, gen_index_map=False)

        # Set up output schema
        output_schema = fastavro.parse_schema(get_inference_output_avro_schema(
            metadata,
            has_logits_per_coordinate=True,  # Always true for custom scipy-based LR
            schema_params=schema_params,
            has_weight=any(schema_params.sample_weight == feature.name for feature in tensor_metadata.get_features())))
        batched_write_avro(itertools.chain.from_iterable(results), output_file, output_schema)
        logger.info(f"Inference complete: {input_path}.")

    def _pooled_action(self, pool, consumer, input_path, schema_params, model_weights, num_features, metadata_file, gen_index_map):
        # Create training dataset
        def get_iterator():
            #  iterator and dataset should be created in the same thread to avoid TF failures.
            logger.info(f"creating TF dataset and iterator on {input_path!r}.")
            dataset = per_entity_grouped_input_fn(
                input_path=os.path.join(input_path, constants.TFRECORD_GLOB_PATTERN),
                metadata_file=metadata_file,
                num_shards=1, shard_index=0,
                batch_size=self.model_params.batch_size,
                data_format=self.model_params.data_format,
                entity_name=self.model_params.partition_entity)
            # Create TF iterator
            return tf.compat.v1.data.make_initializable_iterator(dataset)

        # Create the job generator
        jobs = prepare_jobs(get_iterator, self.model_params, schema_params, num_features=num_features, model_weights=model_weights, gen_index_map=gen_index_map)
        # results = map(consumer, jobs)  # Use this line instead of the next for a single process equivalent for easier dubugging
        return pool.imap_unordered(consumer, jobs, self.model_params.max_training_queue_size)

    def _save_model(self, output_file, model_coefficients, num_features, feature_file):
        # Create model IDs, biases, weight indices and weight value arrays. Account for local indexing
        model_ids = list(model_coefficients.keys())
        global_indices = None if self.model_params.enable_local_indexing else np.arange(num_features)
        biases = []
        list_of_weight_indices = []
        list_of_weight_values = []
        for entity_id, (training_result, unique_global_indices) in model_coefficients.items():
            biases.append(training_result[0])
            list_of_weight_values.append(training_result[1:])
            # Indices map to strictly increasing sequence of global indices or range from 0 to d-1, where d is the number of features in the global space
            indices = unique_global_indices if global_indices is None else global_indices
            list_of_weight_indices.append(indices)

        # Create output file
        output_dir = os.path.dirname(output_file)
        tf.io.gfile.exists(output_dir) or tf.io.gfile.makedirs(output_dir)
        # Delegate to export function
        export_linear_model_to_avro(model_ids, list_of_weight_indices, list_of_weight_values, biases, feature_file, output_file)

    def _load_weights(self, model_file, catch_exception=False):
        logger.info(f"Loading model from {model_file}")
        # Handle exception when the model file does not exist
        # two possibilities, either return empty dict, or raise exception.
        if not tf.io.gfile.exists(model_file):
            if catch_exception:
                logger.info(f"No model found at {model_file}.")
                return {}
            else:
                raise FileNotFoundError(f"Model file {model_file} does not exist")

        # Read feature index map
        feature2global_id = get_feature_map(self.model_params.feature_file)

        # Get the model file and read the avro model
        with tf.io.gfile.GFile(model_file, 'rb') as fo:
            return dict(self._convert_avro_model_record_to_sparse_coefficients(record, feature2global_id) for record in fastavro.reader(fo))

    @staticmethod
    def _convert_avro_model_record_to_sparse_coefficients(model_record, feature2global_id):
        # Extract model id
        model_id = np.int64(model_record["modelId"])

        # Extract model coefficients and global indices
        model_coefficients = []
        unique_global_indices = []
        for idx, ntv in enumerate(model_record["means"]):
            model_coefficients.append(np.float64(ntv["value"]))
            # Add global index if non-intercept feature
            if idx:
                name_term_string = name_term_to_string(ntv["name"], ntv["term"])
                unique_global_indices.append(feature2global_id[name_term_string])
        return model_id, TrainingResult(theta=np.array(model_coefficients), unique_global_indices=np.array(unique_global_indices))

    def export(self, output_model_dir):
        logger.info("Model export is done as part of the training() API for random effect LR LBFGS training. Skipping.")

    def _parse_parameters(self, raw_model_parameters) -> REParams:
        return REParams.__from_argv__(raw_model_parameters, error_on_unknown=False)
