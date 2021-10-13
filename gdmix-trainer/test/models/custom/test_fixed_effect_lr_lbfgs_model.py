import fastavro
import json
import numpy as np
import os
import tempfile
import tensorflow as tf
from collections import namedtuple
from drivers.test_helper import setup_fake_base_training_params, setup_fake_schema_params
from gdmix.io.input_data_pipeline import GZIP, GZIP_SUFFIX, ZLIB, ZLIB_SUFFIX
from gdmix.models.custom.fixed_effect_lr_lbfgs_model import FixedEffectLRModelLBFGS
from gdmix.util import constants
from gdmix.util.io_utils import load_linear_models_from_avro, export_linear_model_to_avro, low_rpc_call_glob
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import expit

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

AllPaths = namedtuple("AllPaths", "training_data_dir "
                                  "validation_data_dir "
                                  "metadata_file "
                                  "feature_file "
                                  "training_score_dir "
                                  "validation_score_dir "
                                  "output_model_dir")

ExpectedData = namedtuple("ExpectedData", "features "
                                          "labels "
                                          "offsets "
                                          "previous_model "
                                          "coefficients "
                                          "per_coordinate_scores "
                                          "total_scores")

_NUM_FEATURES = 10
_NUM_SAMPLES = 100
_NUM_WORKERS = 1  # Multi-worker tests do not work, they hang!
_L2_REG_WEIGHT = 1.0

_NUM_LBFGS_CORRECTIONS = 10
_PRECISION = 1.0e-12
_LARGE_MAX_ITERS = 100
_SMALL_MAX_ITERS = 1

# Ports are hard-coded for now. We should verify them to be unused.
# Packages like portpicker come in handy. (https://github.com/google/python_portpicker)
_PORTS = [13456, 13548, 14356, 14553, 15234, 15379, 15412, 15645, 15654, 16345, 16354, 16435, 16453, 16534, 16543]


class TestFixedEffectLRModelLBFGS(tf.test.TestCase):
    """
    Test linear model with lbfgs solver
    """

    def setUp(self):
        self.datasets_without_offset = {}
        self.datasets_with_offset = {}
        self.datasets_with_offset_and_previous_model = {}
        self.datasets_with_offset_intercept_only = {}
        self.datasets_with_offset_without_intercept = {}
        for model_type in [constants.LOGISTIC_REGRESSION, constants.LINEAR_REGRESSION]:
            self.datasets_without_offset[model_type] = _create_expected_data(False, 0, False, False, True, model_type)
            self.datasets_with_offset[model_type] = _create_expected_data(True, 0, False, False, True, model_type)
            self.datasets_with_offset_and_previous_model[model_type] = _create_expected_data(True, 0, True, False,
                                                                                             True, model_type)
            self.datasets_with_offset_intercept_only[model_type] = _create_expected_data(True, 0, False, True,
                                                                                         True, model_type)
            self.datasets_with_offset_without_intercept[model_type] = _create_expected_data(True, 0, False, False,
                                                                                            False, model_type)

    def testSingleWorkerTraining(self):
        """
        Single worker, test training.
        :return: None
        """
        for idx, model_type in enumerate([constants.LINEAR_REGRESSION, constants.LOGISTIC_REGRESSION]):
            # test single worker training without offset
            self._run_single_worker(False, _PORTS[7 * idx + 0], False, False, model_type=model_type)
            # test single worker training with offset
            self._run_single_worker(True, _PORTS[7 * idx + 1], False, False, model_type=model_type)
            # test single worker training with offset and previous model
            self._run_single_worker(True, _PORTS[7 * idx + 2], True, False, model_type=model_type)
            # test single worker training intercept only model with offset
            self._run_single_worker(True, _PORTS[7 * idx + 3], False, True, model_type=model_type)
            # test single worker training without offset and without validation dataset
            self._run_single_worker(True, _PORTS[7 * idx + 4], False, True, False, model_type=model_type)
            # test single worker training without offset and disable scoring right after training
            self._run_single_worker(True, _PORTS[7 * idx + 5], False, True, True, True, model_type=model_type)
            # test single worker training where model does not have bias
            self._run_single_worker(has_offset=True, port=_PORTS[7 * idx + 6], use_previous_model=False,
                                    intercept_only=False, has_intercept=False, model_type=model_type)

    def testSingleWorkerPrediction(self):
        """
        Single worker, test prediction
        :return: None
        """
        base_dir = tempfile.mkdtemp()
        model_type = constants.LOGISTIC_REGRESSION
        paths = _prepare_paths(base_dir, True, model_type=model_type)
        training_params = _get_params(paths, _LARGE_MAX_ITERS, False, model_type=model_type)
        datasets = self.datasets_with_offset[model_type]
        _write_model(datasets['training'].coefficients, paths.feature_file, paths.output_model_dir, False)
        _write_tfrecord_datasets(datasets, paths, _NUM_WORKERS, True, ZLIB, model_type=model_type)
        proc_func = _ProcFunc(0, [_PORTS[6]], training_params)
        proc_func.__call__(paths, False)
        self._check_scores(datasets['validation'], paths.validation_score_dir)
        tf.io.gfile.rmtree(base_dir)

    def _check_model(self, coefficients, model_dir, feature_file):
        """
        Check if model coefficients are as expected
        :param coefficients: Expected coefficients
        :param model_dir: Model directory
        :param feature_file: path to the feature file
        :return: None
        """
        model_file = os.path.join(model_dir, "part-00000.avro")
        model = load_linear_models_from_avro(model_file, feature_file)[0]
        self.assertAllClose(coefficients, model, msg='models mismatch')

    def _check_scores(self, expected_scores, score_path):
        """
        Check if the scores are as expected.
        :param expected_scores: A namedtuple ExpectedData containing per_coordinate_scores and total_scores
        :param score_path: The directory where the actual scores are stored.
        :return: None
        """
        expected_per_coordinate_scores = expected_scores.per_coordinate_scores
        expected_total_scores = expected_scores.total_scores
        actual_per_coordinate_scores, actual_total_scores = _read_scores(score_path)
        self.assertAllClose(expected_per_coordinate_scores, actual_per_coordinate_scores,
                            msg='per_coordinate scores mismatch')
        self.assertAllClose(expected_total_scores, actual_total_scores,
                            msg='total score mismatch')

    def _run_single_worker(self,
                           has_offset,
                           port,
                           use_previous_model,
                           intercept_only,
                           has_validation_data_dir=True,
                           disable_fixed_effect_scoring_after_training=False,
                           has_intercept=True,
                           model_type=constants.LOGISTIC_REGRESSION):
        """
        A test for single worker training. Dataset were pre-generated.
        The model, training scores and validation scores are checked against expected values.
        :param has_offset: Whether to include offset in the training and validation dataset.
        :param port: Port number used for gRPC communication
        :param use_previous_model: whether to use the previous model
        :param intercept_only: whether this is an intercept-only model (no features)
        :param has_validation_data_dir: whether has validation data dir
        :param disable_fixed_effect_scoring_after_training: whether to disable scoring right after training
        :param has_intercept: whether to include intercept in the model
        :param model_type: the type of linear model to use (e.g, "linear_regression", "logistic_regression", etc.)
        :return: None
        """
        assert has_intercept or not intercept_only
        base_dir = tempfile.mkdtemp()
        if has_offset:
            if use_previous_model:
                datasets = self.datasets_with_offset_and_previous_model[model_type]
            else:
                if intercept_only:
                    datasets = self.datasets_with_offset_intercept_only[model_type]
                elif has_intercept:
                    datasets = self.datasets_with_offset[model_type]
                else:
                    datasets = self.datasets_with_offset_without_intercept[model_type]
        else:
            datasets = self.datasets_without_offset[model_type]
        paths = _prepare_paths(base_dir, has_offset, datasets["training"].previous_model,
                               intercept_only, has_intercept, model_type=model_type)
        if use_previous_model:
            training_params = _get_params(paths, _SMALL_MAX_ITERS, False, model_type=model_type)
        else:
            training_params = _get_params(
                paths, _LARGE_MAX_ITERS, intercept_only, has_validation_data_dir,
                disable_fixed_effect_scoring_after_training, has_intercept, model_type=model_type)
        _write_tfrecord_datasets(datasets, paths, 2, has_offset, model_type=model_type)
        proc_func = _ProcFunc(0, [port], training_params)
        proc_func.__call__(paths, True)
        self._check_model(datasets['training'].coefficients, paths.output_model_dir, paths.feature_file)
        if has_validation_data_dir:
            self._check_scores(datasets['validation'], paths.validation_score_dir)
        if not disable_fixed_effect_scoring_after_training and model_type != constants.LINEAR_REGRESSION:
            self._check_scores(datasets['training'], paths.training_score_dir)
        tf.io.gfile.rmtree(base_dir)


def _prepare_paths(base_dir, has_offset, previous_model=None, intercept_only=False, has_intercept=True,
                   model_type=constants.LOGISTIC_REGRESSION):
    """
    Get an AllPaths namedtuple containing all needed paths.
    Create the directories needed for testing.
    Create feature and metadata files.
    Create previous model if needed.
    :param base_dir: The base directory where all the subfolers will be created.
    :param has_offset: Whether to include offset in the training dataset.
    :param previous_model: Previous model coefficents for warm start training.
    :param intercept_only: Whether the model is intercept only.
    :param has_intercept: Whether the model uses intercept.
    :param model_type: Type of the linear model.
    :return: AllPaths namedtuple
    """
    assert has_intercept or not intercept_only
    if intercept_only:
        feature_dir = None
        feature_file = None
    else:
        feature_dir = os.path.join(base_dir, "featureList")
        feature_file = os.path.join(feature_dir, "global")
    metadata_dir = os.path.join(base_dir, "metadata")
    all_paths = AllPaths(
        training_data_dir=os.path.join(base_dir, "trainingData"),
        validation_data_dir=os.path.join(base_dir, "validationData"),
        metadata_file=os.path.join(metadata_dir, "tensor_metadata.json"),
        feature_file=feature_file,
        training_score_dir=os.path.join(base_dir, "trainingScore"),
        validation_score_dir=os.path.join(base_dir, "validationScore"),
        output_model_dir=os.path.join(base_dir, "modelOutput"))
    if feature_dir:
        tf.io.gfile.mkdir(feature_dir)
    tf.io.gfile.mkdir(metadata_dir)
    tf.io.gfile.mkdir(all_paths.training_data_dir)
    tf.io.gfile.mkdir(all_paths.validation_data_dir)
    tf.io.gfile.mkdir(all_paths.output_model_dir)
    tf.io.gfile.mkdir(all_paths.training_score_dir)
    tf.io.gfile.mkdir(all_paths.validation_score_dir)
    if feature_file:
        _create_feature_file(all_paths.feature_file)
    _create_metadata_file(all_paths.metadata_file, has_offset, model_type=model_type)
    if previous_model is not None:
        _write_model(previous_model, all_paths.feature_file, all_paths.output_model_dir,
                     intercept_only, has_intercept)
    return all_paths


def _get_params(paths, max_iters, intercept_only, has_validation_data_dir=True,
                disable_fixed_effect_scoring_after_training=False, has_intercept=True,
                model_type=constants.LOGISTIC_REGRESSION):
    """
    Get the various parameter for model initialization.
    :param paths: An AllPaths namedtuple.
    :param max_iters: maximum l-BFGS iterations.
    :param intercept_only: whether the model has intercept only, no other features.
    :param has_validation_data_dir: whether to use validation data
    :param disable_fixed_effect_scoring_after_training: whether to disable scoring
    :param has_intercept: whether to include intercept in the model
    :param model_type: the type of linear model to use (e.g, "linear_regression", "logistic_regression", etc.)
    :return: Three different parameter sets.
    """
    base_training_params = setup_fake_base_training_params(training_stage=constants.FIXED_EFFECT,
                                                           model_type=model_type)
    base_training_params.training_score_dir = paths.training_score_dir
    base_training_params.validation_score_dir = paths.validation_score_dir

    schema_params = setup_fake_schema_params()

    raw_model_params = ['--' + constants.TRAINING_DATA_DIR, paths.training_data_dir,
                        '--' + constants.METADATA_FILE, paths.metadata_file,
                        '--' + constants.NUM_OF_LBFGS_ITERATIONS, f"{max_iters}",
                        '--' + constants.OUTPUT_MODEL_DIR, paths.output_model_dir,
                        '--' + constants.COPY_TO_LOCAL, 'False',
                        '--' + constants.BATCH_SIZE, '16',
                        '--' + constants.L2_REG_WEIGHT, f"{_L2_REG_WEIGHT}",
                        "--" + constants.REGULARIZE_BIAS, 'True',
                        "--" + constants.DELAYED_EXIT_IN_SECONDS, '1']

    if has_validation_data_dir:
        raw_model_params.extend(['--' + constants.VALIDATION_DATA_DIR, paths.validation_data_dir])

    if disable_fixed_effect_scoring_after_training:
        raw_model_params.extend(['--disable_fixed_effect_scoring_after_training', 'True'])

    if not intercept_only:
        raw_model_params.extend(['--' + constants.FEATURE_BAG, 'global',
                                 '--' + constants.FEATURE_FILE, paths.feature_file])
    if has_intercept:
        raw_model_params.extend(['--has_intercept', 'True'])
    else:
        raw_model_params.extend(['--has_intercept', 'False', '--regularize_bias', 'False'])
    return base_training_params, schema_params, raw_model_params


def _build_execution_context(worker_index, ports):
    """
    Create execution context for the model.
    :param worker_index: The index of this worker in a distributed setting.
    :param ports: A list of port numbers that will be used in setting up the servers.
    :return: The generated execution context.
    """
    TF_CONFIG = {'cluster': {'worker': [f'localhost:{ports[i]}' for i in range(_NUM_WORKERS)]},
                 'task': {'index': worker_index, 'type': 'worker'}}
    os.environ["TF_CONFIG"] = json.dumps(TF_CONFIG)
    tf_config_json = json.loads(os.environ["TF_CONFIG"])
    cluster = tf_config_json.get('cluster')
    cluster_spec = tf.train.ClusterSpec(cluster)
    execution_context = {
        constants.TASK_TYPE: tf_config_json.get('task', {}).get('type'),
        constants.TASK_INDEX: tf_config_json.get('task', {}).get('index'),
        constants.CLUSTER_SPEC: cluster_spec,
        constants.NUM_WORKERS: tf.train.ClusterSpec(cluster).num_tasks(constants.WORKER),
        constants.NUM_SHARDS: tf.train.ClusterSpec(cluster).num_tasks(constants.WORKER),
        constants.SHARD_INDEX: tf_config_json.get('task', {}).get('index'),
        constants.IS_CHIEF: tf_config_json.get('task', {}).get('index') == 0
    }
    return execution_context


def _create_expected_data(has_offset, seed, use_previous_model, intercept_only, has_intercept=True,
                          model_type=constants.LOGISTIC_REGRESSION):
    """
    Generated expected data for comparison.
    :param has_offset: Whether to use offset.
    :param seed: Random seed
    :param use_previous_model: Whether to generate/use a previous model.
    :param intercept_only: whether the model has intercept only
    :param has_intercept: whether the model uses intercept
    :return: Training and validation datasets.
    """
    np.random.seed(seed)
    if intercept_only:
        training_features = None
        validation_features = None
    else:
        training_features = np.random.rand(_NUM_SAMPLES, _NUM_FEATURES)
        validation_features = np.random.rand(_NUM_SAMPLES, _NUM_FEATURES)
    training_labels = np.random.randint(2, size=_NUM_SAMPLES)
    validation_labels = np.random.randint(2, size=_NUM_SAMPLES)
    if model_type == constants.LINEAR_REGRESSION:
        training_labels = training_labels.astype(np.float64)
        validation_labels = validation_labels.astype(np.float64)
    if has_offset:
        training_offsets = np.random.rand(_NUM_SAMPLES)
        validation_offsets = np.random.rand(_NUM_SAMPLES)
    else:
        training_offsets = np.zeros(_NUM_SAMPLES)
        validation_offsets = np.zeros(_NUM_SAMPLES)
    if intercept_only:
        train_features_plus_one = np.ones((_NUM_SAMPLES, 1))
        validation_features_plus_one = np.ones((_NUM_SAMPLES, 1))
    elif has_intercept:
        train_features_plus_one = np.hstack((training_features, np.ones((_NUM_SAMPLES, 1))))
        validation_features_plus_one = np.hstack((validation_features, np.ones((_NUM_SAMPLES, 1))))
    else:
        train_features_plus_one = training_features
        validation_features_plus_one = validation_features

    previous_model = _solve_for_coefficients(train_features_plus_one, training_labels,
                                             training_offsets, _LARGE_MAX_ITERS, model_type=model_type)
    if use_previous_model:
        coefficients = _solve_for_coefficients(train_features_plus_one, training_labels,
                                               training_offsets, _SMALL_MAX_ITERS, previous_model,
                                               model_type=model_type)
    else:
        coefficients = previous_model
    training_per_coordinate_scores, training_total_scores = _predict(coefficients, train_features_plus_one,
                                                                     training_offsets)
    validation_per_coordinate_scores, validation_total_scores = _predict(coefficients, validation_features_plus_one,
                                                                         validation_offsets)

    return {'training': ExpectedData(training_features if intercept_only else training_features.astype(np.float32),
                                     training_labels,
                                     training_offsets.astype(np.float32),
                                     previous_model.astype(np.float32) if use_previous_model else None,
                                     coefficients.astype(np.float32),
                                     training_per_coordinate_scores.astype(np.float32),
                                     training_total_scores.astype(np.float32)),
            'validation': ExpectedData(
                validation_features if intercept_only else validation_features.astype(np.float32),
                validation_labels,
                validation_offsets.astype(np.float32),
                previous_model.astype(np.float32) if use_previous_model else None,
                coefficients.astype(np.float32),
                validation_per_coordinate_scores.astype(np.float32),
                validation_total_scores.astype(np.float32))}


def _predict(theta, features, offsets):
    """
    Function to get the prediction scores
    :param theta: Model coefficients.
    :param features: Input feature matrix.
    :param offsets: Input offsets.
    :return: Per_coordinate_scores and total_scores.
    """
    per_coordinate_scores = features.dot(theta)
    total_scores = per_coordinate_scores + offsets
    return per_coordinate_scores, total_scores


def _solve_for_coefficients(features, labels, offsets, max_iter, theta_initial=None,
                            model_type=constants.LOGISTIC_REGRESSION):
    """
    Solve LR mdoels with LBFGS solver.
    :param features: Feature matrix
    :param labels: Label vector
    :param offsets: Input offsets.
    :param max_iter: maximum number of LBFGS steps.
    :param theta_initial: Initial value for theta.
    :param model_type: Linear model type.
    :return: Estimated coefficients.
    """

    def _loss(theta, features, offsets, labels, model_type):
        _, pred = _predict(theta, features, offsets)
        if model_type == constants.LOGISTIC_REGRESSION:
            loss = np.maximum(pred, 0) - pred * labels + np.log(1 + np.exp(-np.absolute(pred)))
        else:
            loss = np.square(labels.astype(np.float64) - pred.astype(np.float64))
        loss = loss.sum() + _L2_REG_WEIGHT / 2.0 * theta.dot(theta)
        return loss

    def _gradient(theta, features, offsets, labels, model_type):
        _, logit = _predict(theta, features, offsets)
        if model_type == constants.LOGISTIC_REGRESSION:
            pred = expit(logit)
            cost_grad = features.T.dot(pred - labels)
        else:
            cost_grad = 2.0 * features.T.dot(logit - labels)
        reg_grad = _L2_REG_WEIGHT * theta
        grad = cost_grad + reg_grad
        return grad

    if theta_initial is None:
        theta_initial = np.zeros(features.shape[1])  # including the intercept
    # Run minimization
    result = fmin_l_bfgs_b(func=_loss,
                           x0=theta_initial,
                           approx_grad=False,
                           fprime=_gradient,
                           m=_NUM_LBFGS_CORRECTIONS,
                           factr=_PRECISION,
                           maxiter=max_iter,
                           args=(features, offsets, labels, model_type),
                           disp=0)

    # Extract learned parameters from result
    return result[0]


def _write_tfrecord_datasets(data, paths, num_files, has_offset, compression_type=None,
                             model_type=constants.LOGISTIC_REGRESSION):
    """
    Write the generated data to tfrecord files.
    :param data: A dict of input data, including training and validtion datasets.
    :param paths: An AllPaths namedtuple including all needed paths.
    :param num_files: The number of files to be generated.
    :param has_offset: Whether to use offset.
    :param compression_type: None (uncompressed), ZLIB and GZIP.
    :param model_type: Type of the linear model.
    :return: None
    """
    training, validation = data['training'], data['validation']
    _write_single_dataset(training, paths.training_data_dir, num_files, has_offset, compression_type,
                          model_type=model_type)
    _write_single_dataset(validation, paths.validation_data_dir, num_files, has_offset, compression_type,
                          model_type=model_type)


def _write_single_dataset(data, output_path, num_files, has_offset, compression_type=None,
                          model_type=constants.LOGISTIC_REGRESSION):
    """
    Write a single dataset to tfrecord files.
    :param data: A dict of input data, including training and validation datasets.
    :param output_path: Output path for the generated files.
    :param num_files: The number of files to be generated.
    :param has_offset: Whether to use offset.
    :param compression_type: None (uncompressed), ZLIB and GZIP.
    :param model_type: Type of the linear model.
    :return:
    """
    if compression_type == GZIP:
        suffix = GZIP_SUFFIX
    elif compression_type == ZLIB:
        suffix = ZLIB_SUFFIX
    else:
        suffix = None

    for i in range(num_files):
        indices = np.arange(i, _NUM_SAMPLES, num_files)
        filename = os.path.join(output_path, f"part-{i:05}.tfrecord")
        if suffix:
            filename = os.path.join(output_path, f"part-{i:05}.tfrecord{suffix}")
        with tf.io.TFRecordWriter(filename, options=compression_type) as writer:
            for index in indices:
                if data.features is None:
                    feature_indices, feature_values = None, None
                else:
                    feature_indices, feature_values = _get_sparse_representation(data.features[index])
                if has_offset:
                    offsets = data.offsets[index]
                else:
                    offsets = None
                example = _get_example(feature_indices, feature_values, data.labels[index], 1.0,
                                       index, offsets, model_type=model_type)
                writer.write(example.SerializeToString())


def _write_model(coefficients, feature_file, output_model_dir, intercept_only, has_intercept=True):
    """
    Write model to an avro file in Photon-ML format.
    :param coefficients: Model coefficients, the last element is the bias/intercept.
    :param feature_file: A file with all the features.
    :param output_model_dir: Output directory for the model file.
    :param intercept_only: Whether this is an intercept only model.
    :param has_intercept: Whether to use intercept.
    :return: None
    """
    model_file = os.path.join(output_model_dir, "part-00000.avro")
    bias = coefficients[-1]
    if intercept_only:
        list_of_weight_indices = None
        list_of_weight_values = None
    else:
        weights = coefficients[:-1]
        list_of_weight_indices = np.expand_dims(np.arange(weights.shape[0]), axis=0)
        list_of_weight_values = np.expand_dims(weights, axis=0),
    biases = np.expand_dims(bias, axis=0) if has_intercept else None
    export_linear_model_to_avro(model_ids=["global model"],
                                list_of_weight_indices=list_of_weight_indices,
                                list_of_weight_values=list_of_weight_values,
                                biases=biases,
                                feature_file=feature_file,
                                output_file=model_file)


def _get_sparse_representation(feature):
    """
    Create sparse representation of the features, i.e. indices and values format.
    :param feature: A list of floats. Assume the input is dense.
    :return: indices and values.
    """
    indices = np.arange(len(feature))
    values = feature
    return indices, values


def _get_example(index, value, label, weight, uid, offset, model_type=constants.LOGISTIC_REGRESSION):
    """
    Create a single tfrecord example from various inputs.
    :param index: Index of the feature, part of the sparse representation.
    :param value: Value of the feature, part of the sparse representation.
    :param label: Label of the example.
    :param weight: Weight of the example.
    :param uid: UID of the example.
    :param offset: Offset of the example, or None.
    :param model_type: Type of the linear model.
    :return: A TF example.
    """
    tf_feature = {
        'response': tf.train.Feature(float_list=tf.train.FloatList(
            value=[label])) if model_type == constants.LINEAR_REGRESSION else tf.train.Feature(
            int64_list=tf.train.Int64List(value=[label])),
        'weight': tf.train.Feature(float_list=tf.train.FloatList(
            value=[weight])),
        'uid': tf.train.Feature(int64_list=tf.train.Int64List(
            value=[uid])),
    }
    if index is not None and value is not None:
        tf_feature.update({
            'global_indices': tf.train.Feature(int64_list=tf.train.Int64List(
                value=index)),
            'global_values': tf.train.Feature(float_list=tf.train.FloatList(
                value=value))})
    if offset:
        tf_feature['offset'] = tf.train.Feature(float_list=tf.train.FloatList(
            value=[offset]))
    features = tf.train.Features(feature=tf_feature)
    return tf.train.Example(features=features)


def _create_metadata_file(filename, has_offset, model_type=constants.LOGISTIC_REGRESSION):
    """
    Create the metadata for the dataset.
    :param filename: Output file name.
    :param has_offset: Whether to use offset.
    :param model_type: Type of the linear model.
    :return:
    """
    response_type = "int" if model_type == constants.LOGISTIC_REGRESSION else "float"
    metadata = {
        "features": [{
            "name": "weight",
            "dtype": "float",
            "shape": [],
            "isSparse": False
        }, {
            "name": "global",
            "dtype": "float",
            "shape": [_NUM_FEATURES],
            "isSparse": True
        }, {
            "name": "uid",
            "dtype": "long",
            "shape": [],
            "isSparse": False
        }],
        "labels": [{
            "name": "response",
            "dtype": response_type,
            "shape": [],
            "isSparse": False
        }]
    }
    if has_offset:
        metadata["features"].append({
            "name": "offset",
            "dtype": "float",
            "shape": [],
            "isSparse": False
        })
    with open(filename, 'w') as f:
        json.dump(metadata, f)


def _get_feature_list():
    """
    get the list of features
    :return: A list of the features.
    """
    return [f"feature,{i:05}" for i in range(_NUM_FEATURES)]


def _create_feature_file(output):
    """
    Create the featureList file.
    :param output: output file path.
    :return: None
    """
    with open(output, 'w') as f:
        for n in _get_feature_list():
            f.write(n + "\n")


def _read_scores(score_path):
    """
    Read the generated scores for comparison with expected values.
    :param score_path: Path for the scores.
    :return: Per_coordinate_scores and total_scores.
    """
    records = []
    for avro_file in low_rpc_call_glob("{}/*.avro".format(score_path)):
        with open(avro_file, 'rb') as fo:
            avro_reader = fastavro.reader(fo)
            for rec in avro_reader:
                records.append((rec['uid'], rec['predictionScorePerCoordinate'], rec['predictionScore']))
    records.sort()
    _, per_coordinate_scores, total_scores = list(zip(*records))
    return per_coordinate_scores, total_scores


class _ProcFunc:
    """
    This is a class used by multiprocessing.Process to launch the training.
    The multiprocessing training however does not work so far. It causes hang.
    So it is limited to single-worker single-process usage.
    """

    def __init__(self, worker_index, ports, training_params):
        self.execution_context = _build_execution_context(worker_index, ports)
        self.base_training_params = training_params[0]
        self.schema_params = training_params[1]
        self.raw_model_params = training_params[2]
        self.model = FixedEffectLRModelLBFGS(self.raw_model_params, self.base_training_params)

    def __call__(self, paths, isTrain):
        """
         Call method to train or predict.
        :param paths: AllPaths namedtuple.
        :param isTrain: Whether this is train.
        :return: None
        """
        if isTrain:
            self.model.train(training_data_dir=paths.training_data_dir,
                             validation_data_dir=paths.validation_data_dir,
                             metadata_file=paths.metadata_file,
                             checkpoint_path=paths.output_model_dir,
                             execution_context=self.execution_context,
                             schema_params=self.schema_params)
        else:
            self.model.predict(output_dir=paths.validation_score_dir,
                               input_data_path=paths.validation_data_dir,
                               metadata_file=paths.metadata_file,
                               checkpoint_path=paths.output_model_dir,
                               execution_context=self.execution_context,
                               schema_params=self.schema_params)
