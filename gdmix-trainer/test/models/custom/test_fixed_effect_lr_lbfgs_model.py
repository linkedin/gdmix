import os
import tempfile
import logging
import tensorflow as tf
import json
import fastavro
from drivers.test_helper import setup_fake_base_training_params, setup_fake_schema_params
from gdmix.util import constants
from gdmix.models.custom.fixed_effect_lr_lbfgs_model import FixedEffectLRModelLBFGS
from os.path import join as pathJoin

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
epsilon = 1e-10
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TestFixedEffectLRModelLBFGS(tf.test.TestCase):
    """
    Test logistic regression model with lbfgs solver
    """
    def setUp(self):
        self.input_dir = pathJoin(os.getcwd(), "test/resources/fe_lbfgs")
        self.train_data_path = pathJoin(self.input_dir, 'training_data')
        self.validation_data_path = pathJoin(self.input_dir, 'training_data')
        self.metadata_file = pathJoin(self.input_dir, 'metadata/tensor_metadata.json')
        self.feature_file = pathJoin(self.input_dir, "featureList/global")
        self.output_dir = tempfile.mkdtemp()
        self.training_score_dir = pathJoin(self.output_dir, 'train_score')
        self.validation_score_dir = pathJoin(self.output_dir, 'validation_score')
        self.predict_dir = pathJoin(self.output_dir, 'predict')
        self.model_output_dir = pathJoin(self.output_dir, 'model_output_dir')
        tf.io.gfile.mkdir(self.training_score_dir)
        tf.io.gfile.mkdir(self.validation_score_dir)
        tf.io.gfile.mkdir(self.model_output_dir)
        tf.io.gfile.mkdir(self.predict_dir)

        self.base_training_params, self.schema_params, self.raw_model_params = self._get_params()
        self.model = FixedEffectLRModelLBFGS(self.raw_model_params, self.base_training_params)
        self.execution_context = self._build_execution_context()

    def doCleanups(self):
        tf.io.gfile.rmtree(self.output_dir)

    def _get_params(self):
        base_training_params = setup_fake_base_training_params(training_stage=constants.FIXED_EFFECT)
        base_training_params[constants.TRAINING_OUTPUT_DIR] = self.training_score_dir
        base_training_params[constants.VALIDATION_OUTPUT_DIR] = self.validation_score_dir

        schema_params = setup_fake_schema_params()

        raw_model_params = ['--' + constants.FEATURE_BAGS, 'global',
                            '--' + constants.TRAIN_DATA_PATH, self.train_data_path,
                            '--' + constants.VALIDATION_DATA_PATH, self.validation_data_path,
                            '--' + constants.METADATA_FILE, self.metadata_file,
                            '--' + constants.FEATURE_FILE, self.feature_file,
                            '--' + constants.NUM_OF_LBFGS_ITERATIONS, '1',
                            '--' + constants.MODEL_OUTPUT_DIR, self.model_output_dir,
                            '--' + constants.COPY_TO_LOCAL, 'False',
                            # Batch size > number samples to make sure
                            # there is no shuffling of data among batches
                            '--' + constants.BATCH_SIZE, '64',
                            '--' + constants.L2_REG_WEIGHT, '0.01',
                            "--" + constants.REGULARIZE_BIAS, 'True']
        return base_training_params, schema_params, raw_model_params

    def _build_execution_context(self):
        TF_CONFIG = {'cluster': {'worker': ['localhost:17189']},
                     'task': {'index': 0, 'type': 'worker'},
                     'environment': 'cloud'}
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

    def test_train_and_predict(self):
        self.model.train(training_data_path=self.train_data_path,
                         validation_data_path=self.validation_data_path,
                         metadata_file=self.metadata_file,
                         checkpoint_path=self.model_output_dir,
                         execution_context=self.execution_context,
                         schema_params=self.schema_params)
        assert(os.listdir(self.model_output_dir))
        assert(os.listdir(self.training_score_dir))
        assert(os.listdir(self.validation_score_dir))

        # prediction is done in local model, reset the graph built during training.
        tf.compat.v1.reset_default_graph()
        self.model.predict(output_dir=self.predict_dir,
                           input_data_path=self.validation_data_path,
                           metadata_file=self.metadata_file,
                           checkpoint_path=self.model_output_dir,
                           execution_context=self.execution_context,
                           schema_params=self.schema_params)
        logger.info("[LBFGS] predict directory: {}".format(self.predict_dir))
        assert(os.listdir(self.predict_dir))

        # Check scoring result
        def read_score_result(result_dir):
            records = []
            for avro_file in tf.io.gfile.glob("{}/*.avro".format(result_dir)):
                with open(avro_file, 'rb') as fo:
                    avro_reader = fastavro.reader(fo)
                    for rec in avro_reader:
                        records.append(rec)
            return records

        def check_result(actual, expected):
            for key in ['uid', 'weight', 'response']:
                assert(expected[key] == actual[key])
            for key in ['predictionScorePerCoordinate', 'predictionScore']:
                assert(abs(expected[key] - actual[key]) < epsilon)

        validation_score_result = read_score_result(self.validation_score_dir)
        uid_0_expected_value = {
            'uid': 0,
            'response': 0,
            'weight': 1.0,
            'predictionScore': 1.1810954809188843,
            'predictionScorePerCoordinate': 1.1810954809188843,
        }
        uid_1_expected_value = {
            'uid': 1,
            'response': 0,
            'weight': 1.0,
            'predictionScore': -0.9699453115463257,
            'predictionScorePerCoordinate': -0.9699453115463257,
        }

        check_result(validation_score_result[0], uid_0_expected_value)
        check_result(validation_score_result[1], uid_1_expected_value)

        predict_result = read_score_result(self.predict_dir)
        for actual, expected in zip(predict_result, validation_score_result):
            # This assumes the ordering of the dataset is preserved.
            # This is only true when the batch size > dataset size.
            check_result(actual, expected)
