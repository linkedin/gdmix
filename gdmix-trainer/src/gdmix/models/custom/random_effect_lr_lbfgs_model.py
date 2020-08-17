import argparse
import os
import logging
import numpy as np
import tensorflow as tf
import fastavro

from gdmix.io.input_data_pipeline import per_entity_grouped_input_fn
from gdmix.io.dataset_metadata import DatasetMetadata
from gdmix.models.api import Model
from gdmix.models.custom.binary_logistic_regression import BinaryLogisticRegressionTrainer
from gdmix.models.custom.base_lr_argparser import parser as lr_parser
from gdmix.models.custom.scipy.utils import convert_to_training_jobs
from gdmix.models.photon_ml_writer import PhotonMLWriter
from gdmix.util import constants
from gdmix.util.io_utils import read_json_file, str2bool, export_scipy_lr_model_to_avro, read_feature_list
from multiprocessing import Queue, Manager
from gdmix.models.custom.scipy.training_job_consumer import TrainingJobConsumer, TrainingResult
from gdmix.models.custom.scipy.gdmix_process import GDMixProcess

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

tf.compat.v1.disable_eager_execution()


class RandomEffectLRLBFGSModel(Model):
    """
    Scipy-based custom logistic regression random effect model.

    Supports training entity-based models. Several models can be trained in parallel on multiple processes.
    """

    COMMA_SEPARATOR = ","

    def __init__(self, raw_model_params):
        self.model_params = self._parse_parameters(raw_model_params)
        if constants.FEATURE_BAGS in self.model_params:
            self.model_params[constants.FEATURE_BAGS] = list(self.model_params[constants.FEATURE_BAGS].split(','))
        self.checkpoint_path = os.path.join(self.model_params[constants.MODEL_OUTPUT_DIR])
        self.metadata_file = self.model_params[constants.METADATA_FILE]
        if self.model_params[constants.TRAIN_DATA_PATH] is not None:
            self.training_data_path = os.path.join(self.model_params[constants.TRAIN_DATA_PATH], constants.ACTIVE)
            self.passive_training_data_path = os.path.join(self.model_params[constants.TRAIN_DATA_PATH],
                                                           constants.PASSIVE)
        self.validation_data_path = self.model_params[constants.VALIDATION_DATA_PATH]
        self.partition_index = None

    def train(self, training_data_path, validation_data_path, metadata_file, checkpoint_path, execution_context,
              schema_params):
        logger.info("Kicking off random effect custom LR training")
        self.partition_index = execution_context[constants.PARTITION_INDEX]

        # Create training and validation datasets
        train_data = per_entity_grouped_input_fn(
            input_path=os.path.join(training_data_path, constants.TFRECORD_REGEX_PATTERN),
            metadata_file=metadata_file,
            num_shards=1, shard_index=0,
            batch_size=self.model_params[constants.BATCH_SIZE],
            data_format=self.model_params[constants.DATA_FORMAT],
            entity_name=self.model_params[constants.PARTITION_ENTITY])
        validation_data = per_entity_grouped_input_fn(
            input_path=os.path.join(validation_data_path, constants.TFRECORD_REGEX_PATTERN),
            metadata_file=metadata_file,
            num_shards=1, shard_index=0,
            batch_size=self.model_params[constants.BATCH_SIZE],
            data_format=self.model_params[constants.DATA_FORMAT],
            entity_name=self.model_params[constants.PARTITION_ENTITY])
        logger.info("Training and validation datasets created")

        # Assert that the queue size limit is larger than the number of consumers
        assert (self.model_params[constants.MAX_TRAINING_QUEUE_SIZE] > self.model_params[constants.NUM_OF_CONSUMERS])

        # Queue 1 - Training Job Queue
        training_job_queue = Queue(self.model_params[constants.MAX_TRAINING_QUEUE_SIZE])

        # Create a bunch of consumers
        training_job_consumers = [
            TrainingJobConsumer(consumer_id=i,
                                regularize_bias=self.model_params[constants.REGULARIZE_BIAS],
                                tolerance=self.model_params[constants.LBFGS_TOLERANCE],
                                lambda_l2=self.model_params[constants.L2_REG_WEIGHT],
                                num_of_curvature_pairs=self.model_params[constants.NUM_OF_LBFGS_CURVATURE_PAIRS],
                                num_iterations=self.model_params[constants.NUM_OF_LBFGS_ITERATIONS]) for i in
            range(self.model_params[constants.NUM_OF_CONSUMERS])]

        # Read tensor metadata
        metadata = read_json_file(metadata_file)
        tensor_metadata = DatasetMetadata(metadata)

        # Extract number of features. NOTE - only one feature bag is supported
        num_features = next(filter(lambda x: x.name == self.model_params[constants.FEATURE_BAGS][0],
                                   tensor_metadata.get_features())).shape[0]
        assert num_features > 0, "number of features must > 0"

        # Train using a bounded buffer solution
        with Manager() as manager:
            managed_results_dictionary = manager.dict()

            # Create and kick-off one or more consumer jobs
            consumer_processes = [
                GDMixProcess(target=training_job_consumer, args=(training_job_queue, managed_results_dictionary,
                                                                 self.model_params[
                                                                     constants.TRAINING_QUEUE_TIMEOUT_IN_SECONDS],))
                for training_job_consumer in training_job_consumers]
            for consumer_process in consumer_processes:
                consumer_process.start()

            try:
                # Start producing training jobs
                self._produce_training_jobs(train_data, training_job_queue, schema_params, num_features)

                # Wait for the consumer(s) to finish
                for consumer_process in consumer_processes:
                    consumer_process.join()

                # Convert managed dictionary to regular dictionary
                results_dictionary = dict(managed_results_dictionary)
            except Exception as e:
                for idx, consumer_process in enumerate(consumer_processes):
                    if consumer_process.exception:
                        logger.info("Consumer process with ID: {} failed with exception: {}".format(idx, consumer_process.exception))
                raise Exception("Random effect custom LR training failed. Exception: {}".format(e))

        # Dump results to model output directory.
        if self._model_params_dict_contains_valid_value_for_key(constants.FEATURE_FILE) and \
                self._model_params_dict_contains_valid_value_for_key(constants.MODEL_OUTPUT_DIR):
            self._save_model(model_index=self.partition_index,
                             model_coefficients=results_dictionary,
                             feature_file=self.model_params[constants.FEATURE_FILE],
                             output_dir=self.model_params[constants.MODEL_OUTPUT_DIR])
        else:
            logger.info(
                "Both feature file and avro model output directory required to export model. Skipping export")

        # Run inference on active training set
        if constants.ACTIVE_TRAINING_OUTPUT_FILE in execution_context:
            logger.info("Running inference on the active training dataset")
            self._predict(inference_dataset=train_data, model_coefficients=results_dictionary, metadata=metadata,
                          tensor_metadata=tensor_metadata,
                          output_file=execution_context[constants.ACTIVE_TRAINING_OUTPUT_FILE],
                          prediction_params={**self.model_params, **schema_params})
            logger.info("Inference on active training dataset complete")

        # Run inference on passive training set
        if all(key in execution_context for key in
               (constants.PASSIVE_TRAINING_DATA_PATH, constants.PASSIVE_TRAINING_OUTPUT_FILE)):
            passive_train_data = per_entity_grouped_input_fn(
                input_path=os.path.join(execution_context[constants.PASSIVE_TRAINING_DATA_PATH],
                                        constants.TFRECORD_REGEX_PATTERN),
                metadata_file=metadata_file,
                num_shards=1, shard_index=0,
                batch_size=self.model_params[constants.BATCH_SIZE],
                data_format=self.model_params[constants.DATA_FORMAT],
                entity_name=self.model_params[constants.PARTITION_ENTITY])
            logger.info("Running inference on the passive training dataset")
            self._predict(inference_dataset=passive_train_data, model_coefficients=results_dictionary,
                          metadata=metadata,
                          tensor_metadata=tensor_metadata,
                          output_file=execution_context[constants.PASSIVE_TRAINING_OUTPUT_FILE],
                          prediction_params={**self.model_params, **schema_params})
            logger.info("Inference on passive training dataset complete")

        # Run inference on validation set
        if constants.VALIDATION_OUTPUT_FILE in execution_context:
            logger.info("Running inference on the validation dataset")
            self._predict(inference_dataset=validation_data, model_coefficients=results_dictionary, metadata=metadata,
                          tensor_metadata=tensor_metadata,
                          output_file=execution_context[constants.VALIDATION_OUTPUT_FILE],
                          prediction_params={**self.model_params, **schema_params})
            logger.info("Inference on validation dataset complete")

    def _produce_training_jobs(self, train_data, training_job_queue, schema_params, num_features):
        logger.info("Kicking off training job producer")
        # Create TF iterator
        iterator = tf.compat.v1.data.make_initializable_iterator(train_data)
        features, labels = iterator.get_next()
        logger.info("Dataset initialized")
        # Iterate through TF dataset in a throttled manner
        # (Forking after the TensorFlow runtime creates internal threads is unsafe, use config provided in this
        # link -
        # https://github.com/tensorflow/tensorflow/issues/14442)
        processed_counter = 0
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(use_per_session_threads=True)) as sess:
            sess.run(iterator.initializer)
            while True:
                try:
                    # Extract and process raw entity data
                    features_val, labels_val = sess.run([features, labels])
                    training_jobs = convert_to_training_jobs(features_val, labels_val,
                                                             {**self.model_params, **schema_params},
                                                             num_features=num_features,
                                                             enable_local_indexing=self.model_params[
                                                                 constants.ENABLE_LOCAL_INDEXING])

                    # Add training jobs to shared queue
                    [training_job_queue.put(training_job, True,
                                            self.model_params[constants.TRAINING_QUEUE_TIMEOUT_IN_SECONDS]) for
                     training_job in training_jobs]
                    processed_counter += len(training_jobs)

                    if processed_counter % 1000 == 0:
                        logger.info("Submitted {} training job(s) so far".format(processed_counter))

                except tf.errors.OutOfRangeError:
                    break
        # Add a dummy payload to allow each consumer to terminate
        for i in range(self.model_params[constants.NUM_OF_CONSUMERS]):
            training_job_queue.put(None)

    def _save_model(self, model_index, model_coefficients, feature_file, output_dir):

        # Create model IDs, biases, weight indices and weight value arrays. Account for local indexing
        model_ids = list(model_coefficients.keys())
        biases = []
        list_of_weight_indices = []
        list_of_weight_values = []
        for entity_id in model_coefficients:
            biases.append(model_coefficients[entity_id].training_result[0])
            list_of_weight_values.append(model_coefficients[entity_id].training_result[1:])
            if self.model_params[constants.ENABLE_LOCAL_INDEXING]:
                # Indices map to strictly increasing sequence of global indices
                list_of_weight_indices.append(model_coefficients[entity_id].unique_global_indices)
            else:
                # Indices range from 0 to d-1, where d is the number of features in the global space
                list_of_weight_indices.append(np.arange(model_coefficients[entity_id].training_result[1:].shape[0]))

        # Create output file
        output_file = os.path.join(output_dir, "part-{0:05d}.avro".format(model_index))
        if not tf.io.gfile.exists(output_dir):
            tf.io.gfile.makedirs(output_dir)
        # Delegate to export function
        export_scipy_lr_model_to_avro(model_ids, list_of_weight_indices, list_of_weight_values, biases, feature_file,
                                      output_file)

    def _load_weights(self, model_dir, model_index):
        assert tf.io.gfile.exists(model_dir), "Model path {} doesn't exist".format(model_dir)

        # Read feature file and map features to global index
        feature_list = read_feature_list(self.model_params[constants.FEATURE_FILE])
        feature2global_id = {feat[0] + RandomEffectLRLBFGSModel.COMMA_SEPARATOR + feat[1]: global_id for global_id, feat
                             in enumerate(feature_list)}

        # Get the model file and read the avro model
        model_file = os.path.join(model_dir, "part-{0:05d}.avro".format(model_index))
        model_coefficients = {}
        with tf.io.gfile.GFile(model_file, 'rb') as fo:
            avro_reader = fastavro.reader(fo)
            for record in avro_reader:
                model_id, training_result = self._convert_avro_model_record_to_sparse_coefficients(record,
                                                                                                   feature2global_id)
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
                unique_global_indices.append(
                    feature2global_id[ntv["name"] + RandomEffectLRLBFGSModel.COMMA_SEPARATOR + ntv["term"]])

        return model_id, TrainingResult(training_result=np.array(model_coefficients),
                                        unique_global_indices=np.array(unique_global_indices))

    def _model_params_dict_contains_valid_value_for_key(self, key):
        return key in self.model_params and self.model_params[key] is not None

    def predict(self, output_dir, input_data_path, metadata_file, checkpoint_path, execution_context, schema_params):
        logger.info("Running inference on dataset : {}, results to be written to path : {}".format(
            input_data_path, output_dir))

        # Create output file path
        self.partition_index = execution_context[constants.PARTITION_INDEX]
        output_file = os.path.join(output_dir, "part-{0:05d}.avro".format(self.partition_index))

        # Create training and validation datasets
        inference_dataset = per_entity_grouped_input_fn(
            input_path=os.path.join(input_data_path, constants.TFRECORD_REGEX_PATTERN),
            metadata_file=metadata_file,
            num_shards=1, shard_index=0,
            batch_size=self.model_params[constants.BATCH_SIZE],
            data_format=self.model_params[constants.DATA_FORMAT],
            entity_name=self.model_params[constants.PARTITION_ENTITY])

        # Read model from secondary storage
        model_weights = self._load_weights(model_dir=checkpoint_path, model_index=self.partition_index)

        # Create tensor metadata
        metadata = read_json_file(metadata_file)
        tensor_metadata = DatasetMetadata(metadata)

        # Force local indexing while running prediction
        self.model_params[constants.ENABLE_LOCAL_INDEXING] = True

        # Delegate to in-memory scoring function
        self._predict(inference_dataset=inference_dataset, model_coefficients=model_weights, metadata=metadata,
                      tensor_metadata=tensor_metadata, output_file=output_file,
                      prediction_params={**self.model_params, **schema_params})

    def _predict(self, inference_dataset, model_coefficients, metadata, tensor_metadata, output_file,
                 prediction_params):

        # Create LR trainer object for inference
        lr_trainer = BinaryLogisticRegressionTrainer(regularize_bias=True,
                                                     lambda_l2=self.model_params[constants.L2_REG_WEIGHT])

        # Create PhotonMLWriter object
        prediction_params.update(self.model_params)
        inference_runner = PhotonMLWriter(schema_params=prediction_params)

        # Delegate inference to PhotonMLWriter object
        inference_runner.run_custom_scipy_re_inference(inference_dataset=inference_dataset,
                                                       model_coefficients=model_coefficients,
                                                       lr_model=lr_trainer,
                                                       metadata=metadata,
                                                       tensor_metadata=tensor_metadata,
                                                       output_file=output_file)

    def export(self, output_model_dir):
        logger.info(
            "Model export is done as part of the training() API for random effect LR LBFGS training. Skipping.")

    def _parse_parameters(self, raw_model_parameters):
        parser = argparse.ArgumentParser(parents=[lr_parser])

        # Dataset column names
        parser.add_argument("--" + constants.PARTITION_ENTITY, type=str, required=False,
                            help="Partition entity name.")

        # Training parameters for custom Scipy-based training
        parser.add_argument("--" + constants.MAX_TRAINING_QUEUE_SIZE, type=int, required=False, default=5000,
                            help="Maximum size of training job queue")
        parser.add_argument("--" + constants.TRAINING_QUEUE_TIMEOUT_IN_SECONDS, type=int, required=False, default=300,
                            help="Training queue put timeout in seconds.")
        parser.add_argument("--" + constants.NUM_OF_CONSUMERS, type=int, required=False, default=8,
                            help="Number of consumer processes that will train RE models in parallel.")
        parser.add_argument("--" + constants.ENABLE_LOCAL_INDEXING, type=str2bool, nargs='?', const=True,
                            required=False, default=True, help="Enable local indexing for model training")
        model_params, other_args = parser.parse_known_args(raw_model_parameters)
        return vars(model_params)
