import logging
import os
from dataclasses import dataclass, asdict, replace
from multiprocessing import Manager
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
from gdmix.models.custom.scipy.gdmix_process import GDMixProcess
from gdmix.models.custom.scipy.training_job_consumer import TrainingJobConsumer, TrainingResult
from gdmix.models.custom.scipy.utils import convert_to_training_jobs
from gdmix.models.photon_ml_writer import PhotonMLWriter
from gdmix.params import SchemaParams
from gdmix.util import constants
from gdmix.util.io_utils import read_json_file, export_linear_model_to_avro, get_feature_map, name_term_to_string, batched_write_avro, dataset_reader, \
    get_inference_output_avro_schema

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tf.compat.v1.disable_eager_execution()


@arg_suite
@dataclass
class REParams(LRParams):
    """Scipy-based custom logistic regression random effect model."""
    # Dataset column names
    partition_entity: Optional[str] = None  # Partition entity name.
    # Training parameters for custom Scipy-based training
    enable_local_indexing: Optional[bool] = None  # Enable local indexing for model training
    max_training_queue_size: int = 10  # Maximum size of training job queue
    training_queue_timeout_in_seconds: int = 300  # Training queue put timeout in seconds.
    num_of_consumers: int = 2  # Number of consumer processes that will train RE models in parallel.

    def __post_init__(self):
        assert self.max_training_queue_size > self.num_of_consumers, "queue size limit must be larger than the number of consumers"


@dataclass
class PredictionParams(SchemaParams, REParams):
    """For predictions"""


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
        self.partition_index = None

    def train(self, training_data_path, validation_data_path, metadata_file, checkpoint_path, execution_context, schema_params):
        logger.info("Kicking off random effect custom LR training")
        self.partition_index = execution_context[constants.PARTITION_INDEX]

        # Create training and validation datasets
        train_data = per_entity_grouped_input_fn(
            input_path=os.path.join(training_data_path, constants.TFRECORD_GLOB_PATTERN),
            metadata_file=metadata_file,
            num_shards=1, shard_index=0,
            batch_size=self.model_params.batch_size,
            data_format=self.model_params.data_format,
            entity_name=self.model_params.partition_entity)
        validation_data = per_entity_grouped_input_fn(
            input_path=os.path.join(validation_data_path, constants.TFRECORD_GLOB_PATTERN),
            metadata_file=metadata_file,
            num_shards=1, shard_index=0,
            batch_size=self.model_params.batch_size,
            data_format=self.model_params.data_format,
            entity_name=self.model_params.partition_entity)
        logger.info("Training and validation datasets created")

        # Read tensor metadata
        metadata = read_json_file(metadata_file)
        tensor_metadata = DatasetMetadata(metadata)

        # Extract number of features. NOTE - only one feature bag is supported
        num_features = next(filter(lambda x: x.name == self.model_params.feature_bags[0], tensor_metadata.get_features())).shape[0]
        assert num_features > 0, "number of features must > 0"

        # load initial model if available
        initial_model_weights = self._load_weights(self.model_params.model_output_dir, self.partition_index, True)
        if initial_model_weights:
            logger.info("Found a previous model, loaded as an initial point for training.")
        else:
            logger.info("No previous models found, use all zeros as the model initial point")

        # Train using a bounded buffer solution
        with Manager() as manager:
            managed_results_dictionary = manager.dict(initial_model_weights)

            # Training Job Queue
            training_job_queue = manager.Queue(self.model_params.max_training_queue_size)

            # Create and kick-off one or more consumer jobs
            consumer_processes = tuple(
                GDMixProcess(target=TrainingJobConsumer(consumer_id=i,
                                                        training_results=managed_results_dictionary,
                                                        regularize_bias=self.model_params.regularize_bias,
                                                        tolerance=self.model_params.lbfgs_tolerance,
                                                        lambda_l2=self.model_params.l2_reg_weight,
                                                        num_of_curvature_pairs=self.model_params.num_of_lbfgs_curvature_pairs,
                                                        num_iterations=self.model_params.num_of_lbfgs_iterations,
                                                        timeout_in_seconds=self.model_params.training_queue_timeout_in_seconds,
                                                        ),
                             args=(training_job_queue,))
                for i in range(self.model_params.num_of_consumers))
            for consumer_process in consumer_processes:
                consumer_process.start()

            try:
                # Start producing training jobs
                processed_counter = 0
                conversion_params = PredictionParams(**asdict(self.model_params), **asdict(schema_params))
                for training_job in self._produce_training_jobs(train_data, conversion_params, num_features):
                    # Add training jobs to shared queue
                    training_job_queue.put(training_job, True, self.model_params.training_queue_timeout_in_seconds)
                    processed_counter += 1

                    if processed_counter % 1000 == 0:
                        logger.info(f"Submitted {processed_counter} training job(s) so far")

                # Add a dummy payload to allow each consumer to terminate
                for _ in range(self.model_params.num_of_consumers):
                    training_job_queue.put(None)

                # Wait for the consumer(s) to finish
                for consumer_process in consumer_processes:
                    consumer_process.join()

                # Convert managed dictionary to regular dictionary
                results_dictionary = dict(managed_results_dictionary)
                exceptions = tuple((idx, p.exception) for idx, p in enumerate(consumer_processes) if p.exception)
                if exceptions:
                    logger.info(''.join(f"Consumer process with ID: {idx} failed with exception: {exception}\n" for idx, exception in exceptions))
                    # re-raise the first child exception
                    raise RuntimeError from exceptions[0][1][0]
            except Exception as e:
                raise Exception("Random effect custom LR training failed.") from e

        # Dump results to model output directory.
        if self.model_params.feature_file and self.model_params.model_output_dir:
            self._save_model(model_index=self.partition_index,
                             model_coefficients=results_dictionary,
                             feature_file=self.model_params.feature_file,
                             output_dir=self.model_params.model_output_dir)
        else:
            logger.info("Both feature file and avro model output directory required to export model. Skipping export")

        # Run inference on active training set
        if constants.ACTIVE_TRAINING_OUTPUT_FILE in execution_context:
            logger.info("Running inference on the active training dataset")
            self._predict(inference_dataset=train_data, model_coefficients=results_dictionary, metadata=metadata,
                          tensor_metadata=tensor_metadata,
                          output_file=execution_context[constants.ACTIVE_TRAINING_OUTPUT_FILE],
                          prediction_params=conversion_params)
            logger.info("Inference on active training dataset complete")

        # Run inference on passive training set
        if constants.PASSIVE_TRAINING_DATA_PATH in execution_context and constants.PASSIVE_TRAINING_OUTPUT_FILE in execution_context:
            passive_train_data = per_entity_grouped_input_fn(
                input_path=os.path.join(execution_context[constants.PASSIVE_TRAINING_DATA_PATH], constants.TFRECORD_GLOB_PATTERN),
                metadata_file=metadata_file,
                num_shards=1, shard_index=0,
                batch_size=self.model_params.batch_size,
                data_format=self.model_params.data_format,
                entity_name=self.model_params.partition_entity)
            logger.info("Running inference on the passive training dataset")
            self._predict(inference_dataset=passive_train_data, model_coefficients=results_dictionary,
                          metadata=metadata,
                          tensor_metadata=tensor_metadata,
                          output_file=execution_context[constants.PASSIVE_TRAINING_OUTPUT_FILE],
                          prediction_params=conversion_params)
            logger.info("Inference on passive training dataset complete")

        # Run inference on validation set
        if constants.VALIDATION_OUTPUT_FILE in execution_context:
            logger.info("Running inference on the validation dataset")
            self._predict(inference_dataset=validation_data, model_coefficients=results_dictionary, metadata=metadata,
                          tensor_metadata=tensor_metadata,
                          output_file=execution_context[constants.VALIDATION_OUTPUT_FILE],
                          prediction_params=conversion_params)
            logger.info("Inference on validation dataset complete")

    def _produce_training_jobs(self, train_data, conversion_params, num_features):
        logger.info("Kicking off training job producer")
        for features_val, labels_val in dataset_reader(train_data):
            yield from convert_to_training_jobs(features_val, labels_val,
                                                conversion_params,
                                                num_features=num_features,
                                                enable_local_indexing=self.model_params.enable_local_indexing)

    def _save_model(self, model_index, model_coefficients, feature_file, output_dir):

        # Create model IDs, biases, weight indices and weight value arrays. Account for local indexing
        model_ids = list(model_coefficients.keys())
        biases = []
        list_of_weight_indices = []
        list_of_weight_values = []
        for entity_id in model_coefficients:
            biases.append(model_coefficients[entity_id].training_result[0])
            list_of_weight_values.append(model_coefficients[entity_id].training_result[1:])
            if self.model_params.enable_local_indexing:
                # Indices map to strictly increasing sequence of global indices
                list_of_weight_indices.append(model_coefficients[entity_id].unique_global_indices)
            else:
                # Indices range from 0 to d-1, where d is the number of features in the global space
                list_of_weight_indices.append(np.arange(model_coefficients[entity_id].training_result[1:].shape[0]))

        # Create output file
        output_file = os.path.join(output_dir, f"part-{model_index:05d}.avro")
        if not tf.io.gfile.exists(output_dir):
            tf.io.gfile.makedirs(output_dir)
        # Delegate to export function
        export_linear_model_to_avro(model_ids, list_of_weight_indices, list_of_weight_values, biases, feature_file, output_file)

    def _load_weights(self, model_dir, model_index, catch_exception=False):
        model_file = os.path.join(model_dir, "part-{0:05d}.avro".format(model_index))
        logger.info("Loading model from {}".format(model_file))
        model_exist = tf.io.gfile.exists(model_file)

        # Handle exception when the model file does not exist
        # two possibilities, either return empty dict, or raise exception.
        if not model_exist:
            if catch_exception:
                return dict()
            else:
                raise FileNotFoundError(f"Model file {model_file} does not exist")

        # Read feature index map
        feature2global_id = get_feature_map(self.model_params.feature_file)

        # Get the model file and read the avro model
        model_coefficients = {}
        with tf.io.gfile.GFile(model_file, 'rb') as fo:
            avro_reader = fastavro.reader(fo)
            for record in avro_reader:
                model_id, training_result = self._convert_avro_model_record_to_sparse_coefficients(record, feature2global_id)
                model_coefficients[model_id] = training_result
        return model_coefficients

    def _convert_avro_model_record_to_sparse_coefficients(self, model_record, feature2global_id):
        # Extract model id
        model_id = np.int64(model_record["modelId"])

        # Extract model coefficients and global indices
        model_coefficients = []
        unique_global_indices = []
        for idx, ntv in enumerate(model_record["means"]):
            model_coefficients.append(np.float64(ntv["value"]))
            # Add global index if non-intercept feature
            if idx != 0:
                name_term_string = name_term_to_string(ntv["name"], ntv["term"])
                unique_global_indices.append(feature2global_id[name_term_string])
        return model_id, TrainingResult(training_result=np.array(model_coefficients), unique_global_indices=np.array(unique_global_indices))

    def predict(self, output_dir, input_data_path, metadata_file, checkpoint_path, execution_context, schema_params):
        logger.info(f"Running inference on dataset : {input_data_path}, results to be written to path : {output_dir}")

        # Create output file path
        self.partition_index = execution_context[constants.PARTITION_INDEX]
        output_file = os.path.join(output_dir, f"part-{self.partition_index:05d}.avro")

        # Create training and validation datasets
        inference_dataset = per_entity_grouped_input_fn(
            input_path=os.path.join(input_data_path, constants.TFRECORD_GLOB_PATTERN),
            metadata_file=metadata_file,
            num_shards=1, shard_index=0,
            batch_size=self.model_params.batch_size,
            data_format=self.model_params.data_format,
            entity_name=self.model_params.partition_entity)

        # Read model from secondary storage
        model_weights = self._load_weights(model_dir=checkpoint_path, model_index=self.partition_index)

        # Create tensor metadata
        metadata = read_json_file(metadata_file)
        tensor_metadata = DatasetMetadata(metadata)

        # Force local indexing while running prediction
        self.model_params = replace(self.model_params, enable_local_indexing=True)

        # Delegate to in-memory scoring function
        self._predict(inference_dataset=inference_dataset, model_coefficients=model_weights, metadata=metadata,
                      tensor_metadata=tensor_metadata, output_file=output_file,
                      prediction_params=PredictionParams(**asdict(self.model_params), **asdict(schema_params)))

    def _predict(self, inference_dataset, model_coefficients, metadata, tensor_metadata, output_file, prediction_params):
        # Delegate inference to PhotonMLWriter object
        # Create dataset iterator
        iterator = tf.compat.v1.data.make_initializable_iterator(inference_dataset)
        validation_schema = self.__infer_schema(iterator, metadata, prediction_params, tensor_metadata)
        num_features = next(filter(lambda x: x.name == prediction_params.feature_bags[0], tensor_metadata.get_features())).shape[0]

        # Create LR trainer object for inference
        lr_trainer = BinaryLogisticRegressionTrainer(regularize_bias=True, lambda_l2=self.model_params.l2_reg_weight)

        # Create PhotonMLWriter object
        inference_runner = PhotonMLWriter(lr_trainer, model_coefficients, num_features, prediction_params)
        batched_write_avro(inference_runner.get_batch_iterator(iterator), output_file, validation_schema, inference_runner.INFERENCE_LOG_FREQUENCY)

    def __infer_schema(self, iterator, metadata, prediction_params, tensor_metadata):
        features, labels = iterator.get_next()
        # Set up output schema
        has_label = prediction_params.label in labels
        has_logits_per_coordinate = True  # Always true for custom scipy-based LR
        has_weight = prediction_params.sample_weight in (feature.name for feature in tensor_metadata.get_features())
        validation_schema = fastavro.parse_schema(get_inference_output_avro_schema(metadata,
                                                                                   has_label,
                                                                                   has_logits_per_coordinate,
                                                                                   prediction_params,
                                                                                   has_weight=has_weight))
        return validation_schema

    def export(self, output_model_dir):
        logger.info("Model export is done as part of the training() API for random effect LR LBFGS training. Skipping.")

    def _parse_parameters(self, raw_model_parameters):
        return REParams.__from_argv__(raw_model_parameters, error_on_unknown=False)
