import abc
import os
import logging
import tensorflow as tf
from gdmix.util import constants

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Driver(abc.ABC):
    """
    Abstract class. Must be subclassed to support fixed or random effect training.

    Provides business logic for the following use cases:

    1) Training a Tensorflow (TF) model
    2) Running inference on a TF model
    3) Exporting a TF model to a specific format

    Apart from containing business logic for orchestrating training, inference or model export, the Driver class also
    supports the following:

    1) Cluster configuration - Sets up cluster configuration for local (eg. for random effect training), asynchronous
     distributed or synchronous distributed (eg. for fixed effect training).
    2) Parameter validation - Verifies user-provided parameters for fixed and random effect training.
    The verification is specific for the use cases of training, inference and model export
    3) Data management - Sets up datasets for training or inference
    4)Use-case specific business logic - Encapsulates training/prediction business logic by interacting with
    multiple model APIs for fixed or random effect training

    """

    def __init__(self, base_training_params, model, effect_name):
        self.base_training_params = base_training_params
        self.model = model
        # Verify parameters and setup cluster
        self._validate_params()
        self.execution_context = self._setup_cluster()
        self.effect_name = effect_name

    @abc.abstractmethod
    def _validate_params(self):
        """
        Validate parameters for training, inference and model export. Behavior specific to fixed or random effect
        :return: None
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def _setup_cluster(self):
        """
        Set up execution context for cluster configuration for fixed or random effect. Fixed effect models are expected
        to be trained either synchronously or asynchronously. Random effect training happens on the local machine,
        in isolation from the rest of the cluster.
        :return: None
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def _get_partition_list(self):
        """
        Get list of partitions for different kinds of effects. While fixed effect training usually runs on the entire
        dataset, random effect training on a node can run on one or more data partitions
        :return: list of partitions the current worker must train on
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def _anchor_directory(self, directory_path, partition_index):
        """
        Anchor a directory by attaching partition index specific subdirectory for random effect training.
        For fixed effect training, anchoring is not requiring
        :param directory_path:  Path to directory that needs to be anchored
        :param partition_index: Current partition index for which training/inference/exporting is bring run
        :return: anchored directory
        """
        """

        :return:
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    def run_training(self, schema_params, export_model=False, output_model_dir=None):
        """
        Run Tensorflow training for fixed or random effect training. Training is performed on the entire dataset
        for fixed effect training, or on a list of partitions for fixed effect training.

        :param schema_params        parameters for schema field keyword definition
        :param export_model:        boolean indicating whether model should be exported or not
        :param output_model_dir:    model directory where model should be exported, if export_model is set to True
        :return:    None
        """
        # Log distributed execution context, which includes cluster configuration
        logger.info(f"Commencing {self.effect_name} training")
        logger.info(f"Execution context : {self.execution_context}")

        # Create partition_index_list
        partition_index_list = self._get_partition_list()
        logger.info(f"This worker on work on the following list of partitions : {partition_index_list}")

        # Sequentially train model on partitions
        for partition_index in partition_index_list:
            logger.info(f"Commencing {self.effect_name} training for partition index : {partition_index}")

            # Resolve partitioned data directory from raw path params from user
            checkpoint_path = self._anchor_directory(
                self.model.checkpoint_path,
                partition_index)
            training_data_dir = self._anchor_directory(self.model.training_data_dir,
                                                       partition_index)
            validation_data_dir = self._anchor_directory(self.model.validation_data_dir,
                                                         partition_index)

            # Train model
            self.execution_context[constants.PARTITION_INDEX] = partition_index
            self.model.train(training_data_dir=training_data_dir,
                             validation_data_dir=validation_data_dir,
                             metadata_file=self.model.metadata_file,
                             checkpoint_path=checkpoint_path,
                             execution_context=self._prepare_training_context(partition_index),
                             schema_params=schema_params)

            # Chief should export model
            is_chief = self.execution_context[constants.IS_CHIEF]
            if export_model and is_chief:
                logger.info(f"Exporting model to directory : {output_model_dir}")
                self.model.export(output_model_dir=output_model_dir)

    def run_inference(self, schema_params):
        """
        Run Tensorflow inference using a trained model on a dataset
        Inference output is stored in the AVRO format
        :param schema_params        parameters for schema field keyword definition
        :return: None
        """
        # Log distributed execution context, which includes cluster configuration
        logger.info("Commencing {} inference".format(self.effect_name))
        logger.info("Execution context : {}".format(self.execution_context))

        logger.info('Creating dirs recursively at: {0}'.format(
            self.base_training_params.validation_score_dir))

        if self.execution_context[constants.TASK_TYPE] != constants.TASK_TYPE_WORKER:
            logger.info("Only workers should run inference. Exiting")
            return

        # Create partition_index_list
        partition_index_list = self._get_partition_list()

        for partition_index in partition_index_list:
            logger.info(
                "Commencing {} inference for partition index : {}".format(self.effect_name, partition_index))

            self.execution_context[constants.PARTITION_INDEX] = partition_index
            for input_path, output_path in ((self.model.training_data_dir, self.base_training_params.training_score_dir),
                                            (self.model.validation_data_dir, self.base_training_params.validation_score_dir)):
                if input_path:
                    # Resolve partitioned data path from raw path params from user
                    data_path = self._anchor_directory(input_path, partition_index)
                    output_dir = os.path.join(self._anchor_directory(output_path, partition_index))

                    # Run inference
                    self.model.predict(output_dir=output_dir,
                                       input_data_path=data_path,
                                       metadata_file=self.model.metadata_file,
                                       checkpoint_path=self.model.checkpoint_path,
                                       execution_context=self.execution_context,
                                       schema_params=schema_params)
            logger.info(
                "Inference for partition index : {} complete".format(partition_index))

        logger.info("Inference complete")

    def export_model(self, output_model_dir):
        """
        Export TF model into the SavedModel format
        :param output_model_dir:    model directory where model should be exported
        :return: None
        """
        logger.info("Exporting model to directory : {}".format(output_model_dir))
        self.model.export(output_model_dir=output_model_dir)

    def _prepare_training_context(self, partition_index):
        # If training Scipy-based model for Random effect, add output files to training context
        if self.base_training_params.stage == constants.RANDOM_EFFECT:
            active_training_inference_output_file = os.path.join(
                self._anchor_directory(self.base_training_params.training_score_dir, partition_index),
                "part-{0:05d}-active.avro".format(self.execution_context[constants.TASK_INDEX]))
            passive_training_inference_output_file = os.path.join(
                self._anchor_directory(self.base_training_params.training_score_dir, partition_index),
                "part-{0:05d}-passive.avro".format(self.execution_context[constants.TASK_INDEX]))
            validation_inference_output_file = os.path.join(
                self._anchor_directory(self.base_training_params.validation_score_dir, partition_index),
                "part-{0:05d}.avro".format(self.execution_context[constants.TASK_INDEX]))
            training_context = dict(self.execution_context)

            # If passive dataset exists, add it to training context
            passive_dataset_path = self._anchor_directory(self.model.passive_training_data_dir, partition_index)
            if tf.io.gfile.exists(passive_dataset_path) and len(tf.io.gfile.glob(
                    os.path.join(passive_dataset_path, constants.TFRECORD_GLOB_PATTERN))) != 0:
                training_context[constants.PASSIVE_TRAINING_DATA_DIR] = passive_dataset_path
            # Add paths for inference output
            training_context[constants.ACTIVE_TRAINING_OUTPUT_FILE] = active_training_inference_output_file
            training_context[constants.PASSIVE_TRAINING_OUTPUT_FILE] = passive_training_inference_output_file
            training_context[constants.VALIDATION_OUTPUT_FILE] = validation_inference_output_file
            return training_context
        else:
            return self.execution_context
