import logging
import numpy as np
import os
import psutil
import shutil
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import time

from dataclasses import dataclass
from fastavro import parse_schema
from smart_arg import arg_suite
from scipy.optimize import fmin_l_bfgs_b
from tensorflow.python.ops import collective_ops
from typing import Optional

from gdmix.io.dataset_metadata import DatasetMetadata
from gdmix.io.input_data_pipeline import per_record_input_fn
from gdmix.models.api import Model
from gdmix.models.custom.base_lr_params import LRParams
from gdmix.params import SchemaParams, Params
from gdmix.util import constants
from gdmix.util.distribution_utils import shard_input_files
from gdmix.util.io_utils import add_dummy_weight, read_json_file, try_write_avro_blocks, \
    export_linear_model_to_avro, load_linear_models_from_avro, copy_files, \
    get_inference_output_avro_schema, low_rpc_call_glob
from gdmix.util.model_utils import threshold_coefficients

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(module)s:%(message)s', datefmt='%Y/%m/%d %I:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_VARIANCE_MODE = (constants.FULL, constants.SIMPLE)


def logging(msg):
    """ logging util. """
    logger.info("[FELR] {}".format(msg))


def snooze_after_tf_session_closure(tf_session, duration_in_seconds):
    """
    Snooze after the tf session is closed. This is to avoid the worker exits too quickly,
    resulting in hanging of remote workers who may contact this worker during their session closure.
    This is a workaround to this issue:
    https://github.com/tensorflow/tensorflow/issues/21745
    :param tf_session: a Tensorflow session or MoninitoredTrainingSession.
    :param duration_in_seconds: snooze duration in seconds
    :return: None
    """
    tf_session.close()
    time.sleep(duration_in_seconds)


@arg_suite
@dataclass
class FixedLRParams(LRParams):
    """
    Hyperparameter data class for linear model with scipy LBFGS + TF.
    """
    copy_to_local: bool = True  # Copying data to local or not.
    num_server_creation_retries: int = 50  # Number of retries to establish tf server.
    retry_interval: int = 2  # Number of seconds between retries.
    delayed_exit_in_seconds: int = 60  # Number of seconds before exiting
    disable_fixed_effect_scoring_after_training: bool = False  # Boolean for disabling scoring using the trained model at training phase
    fixed_effect_variance_mode: Optional[str] = None  # How to compute fixed effect training variance. None, FULL or SIMPLE

    def __post_init__(self):
        assert self.fixed_effect_variance_mode is None \
            or self.fixed_effect_variance_mode in _VARIANCE_MODE, \
            f"Action: {self.fixed_effect_variance_mode} must be in {_VARIANCE_MODE}"


class FixedEffectLRModelLBFGS(Model):
    """
    Linear model with scipy LBFGS + TF.
    Support logistic regression by default and plain linear regression if parameter "self.model_type='linear_regression'".
    """
    # TF all reduce op group identifier
    TF_ALL_REDUCE_GROUP_KEY = 0

    def __init__(self, raw_model_params, base_training_params: Params):
        self.model_params: FixedLRParams = self._parse_parameters(raw_model_params)
        self.training_output_dir = base_training_params.training_score_dir
        self.validation_output_dir = base_training_params.validation_score_dir
        self.model_type = base_training_params.model_type
        self.local_training_input_dir = "local_training_input_dir"
        self.lbfgs_iteration = 0
        self.training_data_dir = self.model_params.training_data_dir
        self.validation_data_dir = self.model_params.validation_data_dir
        self.metadata_file = self.model_params.metadata_file
        self.checkpoint_path = self.model_params.output_model_dir
        self.data_format = self.model_params.data_format
        self.offset_column_name = self.model_params.offset_column_name
        self.feature_bag_name = self.model_params.feature_bag
        self.feature_file = self.model_params.feature_file if self.feature_bag_name else None
        self.batch_size = int(self.model_params.batch_size)
        self.copy_to_local = self.model_params.copy_to_local
        self.num_correction_pairs = self.model_params.num_of_lbfgs_curvature_pairs
        self.factor = self.model_params.lbfgs_tolerance / np.finfo(float).eps
        self.has_intercept = self.model_params.has_intercept
        self.is_regularize_bias = self.model_params.regularize_bias
        self.max_iteration = self.model_params.num_of_lbfgs_iterations
        self.l2_reg_weight = self.model_params.l2_reg_weight
        self.sparsity_threshold = self.model_params.sparsity_threshold
        if self.model_type == constants.LOGISTIC_REGRESSION:
            self.disable_fixed_effect_scoring_after_training = self.model_params.disable_fixed_effect_scoring_after_training
        else:
            # disable inference after training for plain linear regression
            self.disable_fixed_effect_scoring_after_training = True

        self.metadata = self._load_metadata()
        self.tensor_metadata = DatasetMetadata(self.metadata_file)
        self.num_features = self._get_num_features()
        self.model_coefficients = None

        self.num_server_creation_retries = self.model_params.num_server_creation_retries
        self.retry_interval = self.model_params.retry_interval
        self.delayed_exit_in_seconds = self.model_params.delayed_exit_in_seconds
        self.server = None
        self.fixed_effect_variance_mode = self.model_params.fixed_effect_variance_mode
        self.epsilon = 1.0e-12

        # validate parameters:
        assert self.feature_file is None or (self.feature_file and tf.io.gfile.exists(
            self.feature_file)), f"feature file {self.feature_file} doesn't exist."

        # validate only support compute variance for logistic regression model
        if self.fixed_effect_variance_mode is not None:
            assert self.model_type == constants.LOGISTIC_REGRESSION, f"doesn't support variance computation for model type {self.mdoel_type}."

    def _create_local_cache(self):
        """ Create a local cache directory to store temporary files. """
        os.makedirs(self.local_training_input_dir, exist_ok=True)

    def _remove_local_cache(self):
        """ Clean up the local cache. """
        shutil.rmtree(self.local_training_input_dir)

    def _load_metadata(self):
        """ Read metadata file from json format. """
        assert tf.io.gfile.exists(self.metadata_file), "metadata file %s does not exist" % self.metadata_file
        return read_json_file(self.metadata_file)

    @staticmethod
    def _get_assigned_files(input_data_path, num_shards, shard_index):
        """
        Get the assigned files from the shard
        :param input_data_path:
        :return: a list of assigned file names.
        """
        assigned_files, sample_level_shard = shard_input_files(input_data_path, num_shards, shard_index)
        assert not sample_level_shard, "Doesn't support sample level sharding," \
                                       "number of files must >= number of workers"
        return assigned_files

    def _get_num_features(self):
        """ Get number of features from metadata. """
        if self.feature_bag_name is None:
            # intercept only model, we pad one dummy feature of zero value.
            num_features = 1
        else:
            num_features = self.tensor_metadata.get_feature_shape(self.feature_bag_name)[0]
        assert num_features > 0, "number of features must > 0"
        return num_features

    def _has_feature(self, feature_column_name):
        """ Check if tensor schema has the provided feature field. """
        return feature_column_name in self.tensor_metadata.get_feature_names()

    @staticmethod
    def _get_feature_bag_tensor(all_features, feature_bag, batch_size):
        """
        Method to get feature tensor. If feature exists, it will return the feature tensor.
        If this is an intercept only model, e.g. no feature exists, it will return a all zero tensor.
        :param all_features: a dict with all features.
        :param feature_bag: feature bag name
        :param batch_size: batch size
        :return: feature tensor
        """
        if feature_bag:
            feature_tensor = all_features[feature_bag]
        else:
            feature_tensor = tf.sparse.SparseTensor(indices=[[0, 0]], values=[0.0], dense_shape=[batch_size, 1])
        return feature_tensor

    def _has_label(self, label_column_name):
        """ Check if tensor schema has the provided label field. """
        return label_column_name in self.tensor_metadata.get_label_names()

    def _create_server(self, execution_context):
        if self.server:
            return
        cluster_spec = execution_context[constants.CLUSTER_SPEC]
        task_index = execution_context[constants.TASK_INDEX]
        config = tf1.ConfigProto()
        config.experimental.collective_group_leader = '/job:worker/replica:0/task:0'
        exception = None
        for i in range(self.num_server_creation_retries):
            try:
                logging(f"No. {i + 1} attempt to create a TF Server, "
                        f"max {self.num_server_creation_retries} attempts")
                self.server = tf1.distribute.Server(cluster_spec,
                                                    config=config,
                                                    job_name='worker',
                                                    task_index=task_index)
                return
            except Exception as e:
                exception = e
                # sleep for retry_interval seconds before next retry
                time.sleep(self.retry_interval)
        raise exception

    def _scoring_fn(self, diter, x_placeholder, num_workers, num_iterations, schema_params: SchemaParams):
        """ Implement the forward pass to get logit. """
        sample_id_list = tf.constant([], tf.int64)
        label_list = tf.constant([], tf.float32)
        weight_list = tf.constant([], tf.float32)
        prediction_score_list = tf.constant([], tf.float64)
        prediction_score_per_coordinate_list = tf.constant([], tf.float64)
        # for variance computation
        variances_dimension = self.num_features + 1 if self.has_intercept else self.num_features
        H = tf.zeros([variances_dimension, variances_dimension])
        if self.fixed_effect_variance_mode == constants.SIMPLE:
            H = tf.zeros(variances_dimension)

        feature_bag_name = self.feature_bag_name
        sample_id_column_name = schema_params.uid_column_name
        label_column_name = schema_params.label_column_name
        sample_weight_column_name = schema_params.weight_column_name
        offset_column_name = self.offset_column_name
        has_offset = self._has_feature(offset_column_name)
        has_label = self._has_label(label_column_name)
        has_weight = self._has_feature(sample_weight_column_name)
        i = tf.constant(0, tf.int64)

        def cond(i, sample_id_list, label_list, weight_list, prediction_score_list, prediction_score_per_coordinate_list, H):
            return tf.less(i, num_iterations)

        def body(i, sample_id_list, label_list, weight_list, prediction_score_list, prediction_score_per_coordinate_list, H):
            i += 1
            all_features, all_labels = diter.get_next()
            sample_ids = all_features[sample_id_column_name]
            current_batch_size = tf.shape(sample_ids)[0]
            features = self._get_feature_bag_tensor(all_features, feature_bag_name, current_batch_size)
            offsets = all_features[offset_column_name] if has_offset else tf.zeros(current_batch_size, tf.float64)
            weights = all_features[sample_weight_column_name] if has_weight else tf.ones(current_batch_size)
            labels = tf.cast(all_labels[label_column_name], tf.float32) if has_label else tf.zeros(current_batch_size)

            sample_id_list = tf.concat([sample_id_list, sample_ids], axis=0)
            weight_list = tf.concat([weight_list, weights], axis=0)
            label_list = tf.concat([label_list, labels], axis=0)

            if self.has_intercept:
                w = x_placeholder[:-1]
                b = x_placeholder[-1]
            else:
                w = x_placeholder
            logits_no_bias = tf.sparse.sparse_dense_matmul(tf.cast(features, tf.float64),
                                                           tf.cast(tf.expand_dims(w, 1), tf.float64))
            if self.has_intercept:
                logits = logits_no_bias + tf.expand_dims(tf.ones(current_batch_size, tf.float64) * tf.cast(b, tf.float64), 1)
            else:
                logits = logits_no_bias
            prediction_score_per_coordinate_list = tf.concat([prediction_score_per_coordinate_list,
                                                              tf.reshape(logits, [-1])], axis=0)

            logits_with_offsets = logits + tf.expand_dims(tf.cast(offsets, tf.float64), 1)
            prediction_score_list = tf.concat([prediction_score_list, tf.reshape(logits_with_offsets, [-1])], axis=0)

            # Compute variance for training data
            if self.fixed_effect_variance_mode is not None:
                rho = tf.cast(tf.math.sigmoid(tf.reshape(logits_with_offsets, [-1])), tf.float32)
                d = rho * (tf.ones(tf.shape(rho)) - rho)
                if has_weight:
                    d = d * tf.cast(weights, tf.float32)

                features_to_dense = tf.sparse.to_dense(features)
                if self.has_intercept:
                    # add intercept column
                    intercept_column = tf.expand_dims(tf.ones(current_batch_size), 1)
                    features_for_variance_compute = tf.concat([features_to_dense, intercept_column], axis=1)
                else:
                    features_for_variance_compute = features_to_dense
                # # compute X^t * D * X
                dx = features_for_variance_compute * tf.expand_dims(d, axis=1)
                batched_H = tf.matmul(features_for_variance_compute, dx, transpose_a=True, a_is_sparse=True, b_is_sparse=True)

                if self.fixed_effect_variance_mode == constants.SIMPLE:
                    H += tf.linalg.diag_part(batched_H)
                elif self.fixed_effect_variance_mode == constants.FULL:
                    H += batched_H

            return i, sample_id_list, label_list, weight_list, prediction_score_list, prediction_score_per_coordinate_list, H

        _, sample_id_list, label_list, weight_list, prediction_score_list, prediction_score_per_coordinate_list, H\
            = tf.while_loop(cond, body,
                            loop_vars=[i, sample_id_list, label_list, weight_list,
                                       prediction_score_list, prediction_score_per_coordinate_list, H],
                            shape_invariants=[i.get_shape()] + [tf.TensorShape([None])] * 5 + [H.get_shape()])

        if self.fixed_effect_variance_mode is not None and num_workers > 1:
            H = collective_ops.all_reduce(
                H, num_workers, FixedEffectLRModelLBFGS.TF_ALL_REDUCE_GROUP_KEY, 2,
                merge_op='Add', final_op='Id')

        return sample_id_list, label_list, weight_list, prediction_score_list, prediction_score_per_coordinate_list, H

    def _train_model_fn(self, diter, x_placeholder, num_workers, num_features, num_iterations,
                        schema_params: SchemaParams):
        """ The training objective function and the gradients. """
        value = tf.constant(0.0, tf.float64)
        if self.has_intercept:
            # Add intercept
            gradients = tf.constant(np.zeros(num_features + 1))
        else:
            gradients = tf.constant(np.zeros(num_features))
        feature_bag_name = self.feature_bag_name
        label_column_name = schema_params.label_column_name
        sample_weight_column_name = schema_params.weight_column_name
        offset_column_name = self.offset_column_name
        is_regularize_bias = self.is_regularize_bias
        has_weight = self._has_feature(sample_weight_column_name)
        has_offset = self._has_feature(offset_column_name)
        has_intercept = self.has_intercept
        i = 0

        def cond(i, value, gradients):
            return i < num_iterations

        def body(i, value, gradients):
            i += 1
            all_features, all_labels = diter.get_next()
            labels = all_labels[label_column_name]
            current_batch_size = tf.shape(labels)[0]
            features = self._get_feature_bag_tensor(all_features, feature_bag_name, current_batch_size)
            weights = all_features[sample_weight_column_name] if has_weight else tf.ones(current_batch_size, tf.float64)
            offsets = all_features[offset_column_name] if has_offset else tf.zeros(current_batch_size, tf.float64)

            if self.has_intercept:
                w = x_placeholder[:-1]
                b = x_placeholder[-1]
            else:
                w = x_placeholder
            logits_no_bias = tf.sparse.sparse_dense_matmul(tf.cast(features, tf.float64),
                                                           tf.cast(tf.expand_dims(w, 1), tf.float64)
                                                           ) + tf.expand_dims(tf.cast(offsets, tf.float64), 1)
            if self.has_intercept:
                logits = logits_no_bias + tf.expand_dims(tf.ones(current_batch_size, tf.float64) * tf.cast(b, tf.float64), 1)
            else:
                logits = logits_no_bias

            if self.model_type == constants.LOGISTIC_REGRESSION:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float64),
                                                               logits=tf.reshape(tf.cast(logits, tf.float64), [-1]))
            else:
                loss = tf.math.squared_difference(tf.cast(labels, tf.float64),
                                                  tf.reshape(tf.cast(logits, tf.float64), [-1]))

            weighted_loss = tf.cast(weights, tf.float64) * loss
            # regularzer has the option to include or exclude bias
            # Note: The L2 is computed on the entire weight vector, this is fine if the dataset has
            # all the features. In some cases, e.g incremental learning, the incremental dataset
            # may only have a subset of the entire features, so the L2 should not be applied to those
            # weights that are not in the dataset. Revisit it when we implement incremental learning.
            # Alternatively, the features that are in the prior models but not the current dataset
            # should not be copied to initial coefficients for warm-start, but needed for inference.
            batch_value = tf.reduce_sum(weighted_loss)
            batch_gradients = tf.gradients(batch_value, x_placeholder)[0]
            value += batch_value
            gradients += batch_gradients
            return i, value, gradients

        _, value, gradients = tf.while_loop(cond, body, [i, value, gradients])
        regularizer = tf.nn.l2_loss(x_placeholder) if (is_regularize_bias or not has_intercept)\
            else tf.nn.l2_loss(x_placeholder[:-1])
        # Divide the regularizer by number of workers because we will sum the contribution of each worker
        # in the all reduce step.
        loss_reg = regularizer * self.l2_reg_weight / float(num_workers)
        value += loss_reg
        gradients += tf.gradients(loss_reg, x_placeholder)[0]
        if num_workers > 1:
            # sum all reduce
            reduced_value = collective_ops.all_reduce(
                value, num_workers, FixedEffectLRModelLBFGS.TF_ALL_REDUCE_GROUP_KEY, 0,
                merge_op='Add', final_op='Id')
            reduced_gradients = collective_ops.all_reduce(
                gradients, num_workers, FixedEffectLRModelLBFGS.TF_ALL_REDUCE_GROUP_KEY, 1,
                merge_op='Add', final_op='Id')
            return reduced_value, reduced_gradients
        else:
            return value, gradients

    def _compute_loss_and_gradients(self, x, tf_session, x_placeholder, ops, task_index):
        """ Compute loss and gradients, invoked by Scipy LBFGS solver. """
        self.lbfgs_iteration += 1

        start_time = time.time()
        init_dataset_op, value_op, gradients_op = ops
        tf_session.run(init_dataset_op)
        value, gradients = tf_session.run([value_op, gradients_op], feed_dict={x_placeholder: x})
        logging(f"Funcall #{self.lbfgs_iteration:4}, total loss = {value}, "
                f"memory used: {self._check_memory()} GB, took {time.time() - start_time} seconds")
        return value, gradients

    def _write_inference_result(self, sample_ids, labels, weights, prediction_score,
                                prediction_score_per_coordinate, task_index, schema_params: SchemaParams, output_dir):
        """ Write inference results. """
        output_avro_schema = get_inference_output_avro_schema(
            self.metadata,
            True,
            schema_params,
            has_weight=self._has_feature(schema_params.weight_column_name))
        parsed_schema = parse_schema(output_avro_schema)

        records = []
        for rec_id, rec_label, rec_weight, rec_prediction_score, rec_prediction_score_per_coordinate in \
                zip(sample_ids, labels, weights, prediction_score, prediction_score_per_coordinate):
            rec = {schema_params.uid_column_name: int(rec_id),
                   schema_params.prediction_score_column_name: float(rec_prediction_score),
                   schema_params.prediction_score_per_coordinate_column_name: float(
                       rec_prediction_score_per_coordinate)}
            if self._has_label(schema_params.label_column_name):
                rec[schema_params.label_column_name] = float(rec_label)
            if self._has_feature(schema_params.weight_column_name):
                rec[schema_params.weight_column_name] = int(rec_weight)
            records.append(rec)

        # Write to a local file then copy to the destination directory
        remote_is_hdfs = output_dir.startswith("hdfs://")
        local_file_name = f"part-{task_index:05d}.avro"
        output_file = local_file_name if remote_is_hdfs else os.path.join(output_dir, local_file_name)
        error_msg = f"worker {task_index} encountered error in writing inference results"
        with open(output_file, 'wb') as f:
            try_write_avro_blocks(f, parsed_schema, records, None, error_msg)
        logging(f"Worker {task_index} has written inference result to local file {output_file}")
        if remote_is_hdfs:
            copy_files([output_file], output_dir)
            os.remove(output_file)
            logging(f"Worker {task_index} has copied inference result to directory {output_dir}")

    def _scoring(self, x, tf_session, x_placeholder, ops, task_index, schema_params, output_dir, compute_training_variance=False):
        """ Run scoring on training or validation dataset. """
        start_time = time.time()
        if compute_training_variance:
            sample_ids_op, labels_op, weights_op, prediction_score_op, prediction_score_per_coordinate_op, variances_op = ops
            sample_ids, labels, weights, prediction_score, prediction_score_per_coordinate, H = tf_session.run(
                [sample_ids_op, labels_op, weights_op, prediction_score_op, prediction_score_per_coordinate_op, variances_op],
                feed_dict={x_placeholder: x})

            if self.fixed_effect_variance_mode == constants.SIMPLE:
                H += self.l2_reg_weight
                if self.has_intercept and not self.is_regularize_bias:
                    # The last element corresponds to the intercept, subtract the l2_reg_weight for the intercept
                    H[-1] -= self.l2_reg_weight
                self.variances = 1.0 / (H + self.epsilon)
            elif self.fixed_effect_variance_mode == constants.FULL:
                H += np.diag([self.l2_reg_weight + self.epsilon] * H.shape[0])
                if self.has_intercept and not self.is_regularize_bias:
                    # The last element corresponds to the intercept, subtract the l2_reg_weight for the intercept
                    H[-1][-1] -= self.l2_reg_weight
                V = np.linalg.inv(H)
                self.variances = np.diagonal(V)
        else:
            sample_ids_op, labels_op, weights_op, prediction_score_op, prediction_score_per_coordinate_op = ops
            sample_ids, labels, weights, prediction_score, prediction_score_per_coordinate = tf_session.run(
                [sample_ids_op, labels_op, weights_op, prediction_score_op, prediction_score_per_coordinate_op],
                feed_dict={x_placeholder: x})

        self._write_inference_result(sample_ids, labels, weights, prediction_score,
                                     prediction_score_per_coordinate, task_index,
                                     schema_params, output_dir)
        logging(f"Inference --- {time.time() - start_time} seconds ---")

    def _check_memory(self):
        """ Check memory usage. """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e9

    def _get_num_iterations(self, input_files, metadata_file):
        """ Get the number of samples each worker assigned.
            This works for tfrecord only.
        :param input_files: a list of TFRecord files.
        :param metadata_file: the metadata associated with the TFRecord files.
        :return: number of iterations
        """
        start_time = time.time()
        assert (self.data_format == constants.TFRECORD)
        # reset the default graph, so it has been called before the main graph is built.
        tf1.reset_default_graph()
        num_iterations = 0
        dataset = per_record_input_fn(input_files, metadata_file, 1, 0, self.batch_size, self.data_format,
                                      build_features=False)
        data_iterator = tf1.data.make_initializable_iterator(dataset)
        next_item = data_iterator.get_next()
        with tf1.device('device:CPU:0'), tf1.Session() as sess:
            sess.run(data_iterator.initializer)
            while True:
                try:
                    sess.run(next_item)
                    num_iterations += 1
                except tf.errors.OutOfRangeError:
                    break
        end_time = time.time()
        logging(f'It took {end_time - start_time} seconds to count {num_iterations} batches '
                f'with batch size {self.batch_size}.')
        return num_iterations

    def train(self, training_data_dir, validation_data_dir, metadata_file, checkpoint_path,
              execution_context, schema_params):
        """ Overwrite train method from parent class. """
        logging("Kicking off fixed effect LR LBFGS training")

        task_index = execution_context[constants.TASK_INDEX]
        num_workers = execution_context[constants.NUM_WORKERS]
        is_chief = execution_context[constants.IS_CHIEF]
        self._create_server(execution_context)

        assigned_training_files = self._get_assigned_files(training_data_dir, num_workers, task_index)
        if self.copy_to_local:
            training_input_dir = self.local_training_input_dir
            self._create_local_cache()
            actual_training_files = copy_files(assigned_training_files, training_input_dir)
            # After copy the worker's shard to local, we don't shard the local files any more.
            training_data_num_shards = 1
            training_data_shard_index = 0
        else:
            training_input_dir = training_data_dir
            actual_training_files = assigned_training_files
            training_data_num_shards = num_workers
            training_data_shard_index = task_index

        # Compute the number of iterations before the main graph is built.
        training_data_num_iterations = self._get_num_iterations(actual_training_files, metadata_file)
        if validation_data_dir:
            assigned_validation_files = self._get_assigned_files(validation_data_dir, num_workers, task_index)
            validation_data_num_iterations = self._get_num_iterations(assigned_validation_files, metadata_file)
        # Define the graph here, keep session open to let scipy L-BFGS solver repeatedly call
        # _compute_loss_and_gradients
        # Reset the graph.
        tf1.reset_default_graph()
        with tf1.variable_scope('worker{}'.format(task_index)), \
                tf1.device('job:worker/task:{}/device:CPU:0'.format(task_index)):

            # Define ops for training
            training_dataset = per_record_input_fn(training_input_dir,
                                                   metadata_file,
                                                   training_data_num_shards,
                                                   training_data_shard_index,
                                                   self.batch_size,
                                                   self.data_format)
            training_data_iterator = tf1.data.make_initializable_iterator(training_dataset)
            init_training_dataset_op = training_data_iterator.initializer
            training_x_placeholder = tf1.placeholder(tf.float64, shape=[None])
            value_op, gradients_op = self._train_model_fn(training_data_iterator,
                                                          training_x_placeholder,
                                                          num_workers,
                                                          self.num_features,
                                                          training_data_num_iterations,
                                                          schema_params)
            training_ops = (init_training_dataset_op, value_op, gradients_op)

            # Define ops for inference
            inference_x_placeholder = tf1.placeholder(tf.float64, shape=[None])
            if not self.disable_fixed_effect_scoring_after_training or self.fixed_effect_variance_mode is not None:
                inference_training_data_iterator = tf1.data.make_one_shot_iterator(training_dataset)
                training_sample_ids_op, training_labels_op, training_weights_op, training_prediction_score_op, \
                    training_prediction_score_per_coordinate_op, H_op = self._scoring_fn(
                        inference_training_data_iterator,
                        inference_x_placeholder,
                        num_workers,
                        training_data_num_iterations,
                        schema_params)

            if validation_data_dir:
                valid_dataset = per_record_input_fn(validation_data_dir,
                                                    metadata_file,
                                                    num_workers,
                                                    task_index,
                                                    self.batch_size,
                                                    self.data_format)
                inference_validation_data_iterator = tf1.data.make_one_shot_iterator(valid_dataset)
                valid_sample_ids_op, valid_labels_op, valid_weights_op, valid_prediction_score_op, valid_prediction_score_per_coordinate_op, _ = \
                    self._scoring_fn(
                        inference_validation_data_iterator,
                        inference_x_placeholder,
                        num_workers,
                        validation_data_num_iterations,
                        schema_params)

            if num_workers > 1:
                all_reduce_sync_op = collective_ops.all_reduce(
                    tf.constant(0.0, tf.float64),
                    num_workers,
                    FixedEffectLRModelLBFGS.TF_ALL_REDUCE_GROUP_KEY,
                    0,
                    merge_op='Add',
                    final_op='Id')

            init_variables_op = tf1.global_variables_initializer()

        session_creator = tf1.train.ChiefSessionCreator(master=self.server.target)
        tf_session = tf1.train.MonitoredSession(session_creator=session_creator)
        tf_session.run(init_variables_op)

        # load existing model if available
        logging("Try to load initial model coefficients...")
        prev_model = self._load_model(catch_exception=True)
        expected_model_size = self.num_features + 1 if self.has_intercept else self.num_features
        if prev_model is None:
            logging("No initial model found, use all zeros instead.")
            use_zero = True
        elif len(prev_model) != expected_model_size:
            logging(f"Initial model size is {len(prev_model)},"
                    f"expected {expected_model_size}, use all zeros instead.")
            use_zero = True
        else:
            use_zero = False
        if use_zero:
            x0 = np.zeros(expected_model_size)
        else:
            logging("Found a previous model,  loaded as the initial point for training")
            x0 = prev_model

        # Run all reduce warm up
        logging("All-reduce-warmup starts...")
        if num_workers > 1:
            start_time = time.time()
            tf_session.run([all_reduce_sync_op])
            logging("All-reduce-warmup --- {} seconds ---".format(time.time() - start_time))

        # Start training
        logging("Training starts...")
        start_time = time.time()
        self.model_coefficients, f_min, info = fmin_l_bfgs_b(
            func=self._compute_loss_and_gradients,
            x0=x0,
            approx_grad=False,
            m=self.num_correction_pairs,  # number of variable metrics corrections. default is 10.
            factr=self.factor,  # control precision, smaller the better.
            maxiter=self.max_iteration,
            args=(tf_session, training_x_placeholder, training_ops, task_index),
            disp=0)
        logging("Training --- {} seconds ---".format(time.time() - start_time))
        logging("\n------------------------------\nf_min: {}\nnum of funcalls: {}\ntask msg:"
                "{}\n------------------------------".format(f_min, info['funcalls'], info['task']))

        logging(f"Zeroing coefficients equal to or below {self.sparsity_threshold}")
        self.model_coefficients = threshold_coefficients(self.model_coefficients, self.sparsity_threshold)

        if not self.disable_fixed_effect_scoring_after_training or self.fixed_effect_variance_mode is not None:
            logging("Inference training data starts...")
            inference_training_data_ops = (training_sample_ids_op, training_labels_op, training_weights_op,
                                           training_prediction_score_op, training_prediction_score_per_coordinate_op)
            if self.fixed_effect_variance_mode is not None:
                inference_training_data_ops = inference_training_data_ops + (H_op,)
            self._scoring(self.model_coefficients,
                          tf_session,
                          inference_x_placeholder,
                          inference_training_data_ops,
                          task_index,
                          schema_params,
                          self.training_output_dir,
                          self.fixed_effect_variance_mode is not None)
        if validation_data_dir:
            logging("Inference validation data starts...")
            inference_validation_data_ops = (valid_sample_ids_op, valid_labels_op, valid_weights_op,
                                             valid_prediction_score_op, valid_prediction_score_per_coordinate_op)
            self._scoring(self.model_coefficients,
                          tf_session,
                          inference_x_placeholder,
                          inference_validation_data_ops,
                          task_index,
                          schema_params,
                          self.validation_output_dir)

        # Final sync up and then reliably terminate all workers
        if (num_workers > 1):
            tf_session.run([all_reduce_sync_op])

        snooze_after_tf_session_closure(tf_session, self.delayed_exit_in_seconds)

        if is_chief:
            self._save_model()

        # remove the cached training input files
        if self.copy_to_local:
            self._remove_local_cache()

    def _save_model(self):
        """ Save the trained linear model in avro format. """
        compute_training_variance = self.fixed_effect_variance_mode is not None
        if self.has_intercept:
            if compute_training_variance:
                bias = (self.model_coefficients[-1], self.variances[-1])
            else:
                bias = self.model_coefficients[-1]
        else:
            bias = None
        expanded_bias = None if bias is None else [bias]

        if self.feature_bag_name is None:
            # intercept only model
            list_of_weight_indices = None
            list_of_weight_values = None
        else:
            if self.has_intercept:
                weights = self.model_coefficients[:-1]
                variances = self.variances[:-1] if compute_training_variance else None
            else:
                weights = self.model_coefficients
                variances = self.variances if compute_training_variance else None
            indices = np.arange(weights.shape[0])
            list_of_weight_values = [weights] if variances is None else [(weights, variances)]
            list_of_weight_indices = [indices]
        output_file = os.path.join(self.checkpoint_path, "part-00000.avro")
        if self.model_type == constants.LOGISTIC_REGRESSION:
            model_class = "com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel"
        else:
            model_class = "com.linkedin.photon.ml.supervised.regression.LinearRegressionModel"
        export_linear_model_to_avro(model_ids=["global model"],
                                    list_of_weight_indices=list_of_weight_indices,
                                    list_of_weight_values=list_of_weight_values,
                                    biases=expanded_bias,
                                    feature_file=self.feature_file,
                                    output_file=output_file,
                                    model_class=model_class,
                                    sparsity_threshold=self.sparsity_threshold)

    def _load_model(self, catch_exception=False):
        """ Load model from avro file. """
        model = None
        logging("Loading model from {}".format(self.checkpoint_path))
        model_exist = self.checkpoint_path and tf.io.gfile.exists(self.checkpoint_path)
        if model_exist:
            model_file = low_rpc_call_glob("{}/*.avro".format(self.checkpoint_path))
            if len(model_file) == 1:
                model = load_linear_models_from_avro(model_file[0], self.feature_file)[0]
            elif not catch_exception:
                raise ValueError("Load model failed, no model file or multiple model"
                                 " files found in the model diretory {}".format(self.checkpoint))
        elif not catch_exception:
            raise FileNotFoundError("checkpoint path {} doesn't exist".format(self.checkpoint_path))
        if self.feature_bag_name is None and model is not None:
            # intercept only model, add a dummy weight.
            model = add_dummy_weight(model)
        return model

    def export(self, output_model_dir):
        logging("No need model export for LR model. ")

    def predict(self,
                output_dir,
                input_data_path,
                metadata_file,
                checkpoint_path,
                execution_context,
                schema_params):
        # Overwrite predict method from parent class.
        logging("Kicking off fixed effect LR predict")

        task_index = execution_context[constants.TASK_INDEX]
        num_workers = execution_context[constants.NUM_WORKERS]
        # Prediction uses local server
        self.server = tf1.train.Server.create_local_server()
        # Compute the number of iterations before the main graph is built.
        assigned_files = self._get_assigned_files(input_data_path, num_workers, task_index)
        data_num_iterations = self._get_num_iterations(assigned_files, metadata_file)
        # Define the graph here, keep session open to let scipy L-BFGS solver repeatedly call
        # _compute_loss_and_gradients
        # Inference is conducted in local mode.
        # Reset the default graph.
        tf1.reset_default_graph()
        with tf1.variable_scope('worker{}'.format(task_index)), tf1.device('device:CPU:0'):
            dataset = per_record_input_fn(input_data_path,
                                          metadata_file,
                                          num_workers,
                                          task_index,
                                          self.batch_size,
                                          self.data_format)
            x_placeholder = tf1.placeholder(tf.float64, shape=[None])

            data_iterator = tf1.data.make_one_shot_iterator(dataset)
            sample_ids_op, labels_op, weights_op, scores_op, scores_and_offsets_op, _ = self._scoring_fn(
                data_iterator,
                x_placeholder,
                num_workers,
                data_num_iterations,
                schema_params)
            init_variables_op = tf1.global_variables_initializer()

        session_creator = tf1.train.ChiefSessionCreator(master=self.server.target)
        tf_session = tf1.train.MonitoredSession(session_creator=session_creator)
        tf_session.run(init_variables_op)

        predict_ops = (sample_ids_op, labels_op, weights_op, scores_op, scores_and_offsets_op)
        model_coefficients = self._load_model()
        self._scoring(model_coefficients,
                      tf_session,
                      x_placeholder,
                      predict_ops,
                      task_index,
                      schema_params,
                      output_dir)
        logging("Snooze before closing the session")
        snooze_after_tf_session_closure(tf_session, self.delayed_exit_in_seconds)
        logging("Closed the session")

    def _parse_parameters(self, raw_model_parameters):
        params = FixedLRParams.__from_argv__(raw_model_parameters, error_on_unknown=False)
        logging(params)
        return params
