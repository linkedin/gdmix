import os
import tempfile
import tensorflow as tf
from unittest import mock
from unittest.mock import Mock
from gdmix.util import constants
from gdmix.drivers.fixed_effect_driver import FixedEffectDriver
from gdmix.drivers.random_effect_driver import RandomEffectDriver
from .test_helper import set_fake_tf_config, setup_fake_base_training_params, \
    setup_fake_schema_params


class TestDriver(tf.test.TestCase):
    """
    Test Fixed and Random effect drivers
    """

    def setUp(self):
        self.task_type = "worker"
        self.worker_index = 0
        self.num_workers = 5
        set_fake_tf_config(task_type=self.task_type, worker_index=self.worker_index)
        self.base_training_params = setup_fake_base_training_params()
        self.schema_params = setup_fake_schema_params()
        self.output_dir = tempfile.mkdtemp()

        self.mock_model = Mock()
        self.mock_model.metadata_file = os.path.join(os.getcwd(), "test/resources/metadata/tensor_metadata.json")
        self.mock_model.validation_data_dir = os.path.join(os.getcwd(), "test/resources/validate")
        self.mock_model.checkpoint_path = self.output_dir
        self.mock_model.training_data_dir = os.path.join(os.getcwd(), "test/resources/train")

        self.fixed_effect_driver = FixedEffectDriver(base_training_params=self.base_training_params,
                                                     model=self.mock_model)
        self.random_effect_driver = RandomEffectDriver(base_training_params=self.base_training_params,
                                                       model=self.mock_model)

    def tearDown(self):
        # Clean up the checkpoint dir created by the driver
        tf.io.gfile.rmtree(self.output_dir)

    def test_drivers_without_tfconfig(self):
        """
        Test fixed and random effect driver constructors without TF_CONFIG, expect to set local mode
        :return: None
        """
        if constants.TF_CONFIG in os.environ:
            del os.environ[constants.TF_CONFIG]
        expected_fe_execution_context = {constants.TASK_TYPE: 'worker',
                                         constants.TASK_INDEX: 0,
                                         constants.CLUSTER_SPEC: {"worker": ["localhost:2222"]},
                                         constants.NUM_WORKERS: 1,
                                         constants.NUM_SHARDS: 1,
                                         constants.SHARD_INDEX: 0,
                                         constants.IS_CHIEF: True}

        fixed_effect_driver = FixedEffectDriver(base_training_params=self.base_training_params,
                                                model=self.mock_model)
        self.assertEqual(fixed_effect_driver.execution_context, expected_fe_execution_context)

        random_effect_driver = RandomEffectDriver(base_training_params=self.base_training_params,
                                                  model=self.mock_model)
        expected_re_execution_context = expected_fe_execution_context
        expected_re_execution_context[constants.CLUSTER_SPEC] = None
        self.assertEqual(random_effect_driver.execution_context, expected_re_execution_context)

    def test_fixed_effect_cluster_spec(self):
        """
        Test the cluster specification for fixed effect training
        :return: None
        """
        fe_execution_context = self.fixed_effect_driver.execution_context

        # Assert cluster specification
        self.assertEqual(fe_execution_context[constants.TASK_INDEX], self.worker_index)
        self.assertEqual(fe_execution_context[constants.TASK_TYPE], self.task_type)
        self.assertEqual(fe_execution_context[constants.NUM_WORKERS], self.num_workers)
        self.assertEqual(fe_execution_context[constants.NUM_SHARDS], self.num_workers)
        self.assertEqual(fe_execution_context[constants.SHARD_INDEX], self.worker_index)

    def test_random_effect_cluster_spec(self):
        """
        Test the cluster specification for random effect training
        :return: None
        """
        re_execution_context = self.random_effect_driver.execution_context

        # Assert cluster specification
        self.assertEqual(re_execution_context[constants.TASK_INDEX], self.worker_index)
        self.assertEqual(re_execution_context[constants.TASK_TYPE], self.task_type)
        self.assertEqual(re_execution_context[constants.NUM_WORKERS], self.num_workers)
        self.assertEqual(re_execution_context[constants.NUM_SHARDS], 1)
        self.assertEqual(re_execution_context[constants.SHARD_INDEX], 0)

    def test_fixed_effect_training(self):
        """
        Test the fixed effect driver during training
        :return: None
        """
        # Run training
        self.fixed_effect_driver.run_training(schema_params=self.schema_params, export_model=False, output_model_dir=None)

        # Assert model is trained only once with the right parameters
        self.mock_model.train.assert_called_once_with(training_data_dir=self.mock_model.training_data_dir,
                                                      validation_data_dir=self.mock_model.validation_data_dir,
                                                      metadata_file=self.mock_model.metadata_file,
                                                      checkpoint_path=self.mock_model.checkpoint_path,
                                                      execution_context=self.fixed_effect_driver.execution_context,
                                                      schema_params=self.schema_params)

    def test_random_effect_training(self):
        """
        Test the random effect driver during training
        :return: None
        """
        # Run training
        self.random_effect_driver.run_training(schema_params=self.schema_params, export_model=False, output_model_dir=None)

        # Read dummy partition index list. Parse the partitions random effect worker should work on
        with tf.io.gfile.GFile(self.base_training_params.partition_list_file) as f:
            line = f.readline()
        all_partitions = [int(l) for l in line.split(',')]
        partition_index_list = [all_partitions[i] for i in
                                (list(range(self.random_effect_driver.execution_context[constants.TASK_INDEX],
                                            len(all_partitions),
                                            self.random_effect_driver.execution_context[constants.NUM_WORKERS])))]

        # Gather all the calls to compile() and train() method of the mock model
        train_calls = []

        for partition_index in partition_index_list:
            checkpoint_path = self.random_effect_driver._anchor_directory(self.mock_model.checkpoint_path, partition_index)
            training_data_dir = self.random_effect_driver._anchor_directory(self.mock_model.training_data_dir, partition_index)
            validation_data_dir = self.random_effect_driver._anchor_directory(self.mock_model.validation_data_dir, partition_index)
            train_calls.append(mock.call(training_data_dir=training_data_dir,
                                         validation_data_dir=validation_data_dir,
                                         metadata_file=self.mock_model.metadata_file,
                                         checkpoint_path=checkpoint_path,
                                         execution_context=self.random_effect_driver.execution_context,
                                         schema_params=self.schema_params))

        # Assert model was called with the right calls
        self.mock_model.train.assert_has_calls(train_calls)

    def test_fixed_effect_inference(self):
        """
        Test the fixed effect driver during inference
        :param test_create_dataset: mock create_dataset function
        :return: None
        """
        self.base_training_params.action = constants.ACTION_INFERENCE
        # Run inference
        self.fixed_effect_driver.run_inference(schema_params=self.schema_params)
        inference_calls = []
        inference_calls.append(mock.call(output_dir=os.path.join(
                                             self.base_training_params.training_score_dir),
                                         input_data_path=self.mock_model.training_data_dir,
                                         metadata_file=self.mock_model.metadata_file,
                                         checkpoint_path=self.mock_model.checkpoint_path,
                                         execution_context=self.fixed_effect_driver.execution_context,
                                         schema_params=self.schema_params))
        inference_calls.append(mock.call(output_dir=os.path.join(
                                             self.base_training_params.validation_score_dir),
                                         input_data_path=self.mock_model.validation_data_dir,
                                         metadata_file=self.mock_model.metadata_file,
                                         checkpoint_path=self.mock_model.checkpoint_path,
                                         execution_context=self.fixed_effect_driver.execution_context,
                                         schema_params=self.schema_params))
        # Assert model was called with the right calls
        self.mock_model.predict.assert_has_calls(inference_calls)

    def test_random_effect_inference(self):
        """
        Test the random effect driver during inference
        :param test_create_dataset: mock create_dataset function
        :return: None
        """
        self.base_training_params.action = constants.ACTION_INFERENCE
        # Run inference
        self.random_effect_driver.run_inference(schema_params=self.schema_params)

        # Read dummy partition index list. Parse the partitions random effect worker should work on
        with tf.io.gfile.GFile(self.base_training_params.partition_list_file) as f:
            line = f.readline()
        all_partitions = [int(l) for l in line.split(',')]
        partition_index_list = [all_partitions[i] for i in
                                (list(range(self.random_effect_driver.execution_context[constants.TASK_INDEX],
                                            len(all_partitions),
                                            self.random_effect_driver.execution_context[constants.NUM_WORKERS])))]

        # Gather all the calls to create_dataset(), compile() and train() method of the mock dataset_loader and model
        infer_calls = []
        for partition_index in partition_index_list:
            checkpoint_path = os.path.join(self.mock_model.checkpoint_path)
            training_data_dir = self.random_effect_driver._anchor_directory(
                self.mock_model.training_data_dir, partition_index)
            validation_data_dir = self.random_effect_driver._anchor_directory(
                self.mock_model.validation_data_dir, partition_index)
            infer_calls.append(mock.call(output_dir=os.path.join(
                                             self.base_training_params.training_score_dir,
                                             RandomEffectDriver._RANDOM_EFFECT_PARTITION_DIR_PREFIX + str(partition_index)),
                                         input_data_path=training_data_dir,
                                         metadata_file=self.mock_model.metadata_file,
                                         checkpoint_path=checkpoint_path,
                                         execution_context=self.random_effect_driver.execution_context,
                                         schema_params=self.schema_params))
            infer_calls.append(mock.call(output_dir=os.path.join(
                                             self.base_training_params.validation_score_dir,
                                             RandomEffectDriver._RANDOM_EFFECT_PARTITION_DIR_PREFIX + str(partition_index)),
                                         input_data_path=validation_data_dir,
                                         metadata_file=self.mock_model.metadata_file,
                                         checkpoint_path=checkpoint_path,
                                         execution_context=self.random_effect_driver.execution_context,
                                         schema_params=self.schema_params))

        # Assert create_dataset() and model were called with the right calls
        # test_create_dataset.assert_has_calls(create_dataset_calls)
        self.mock_model.predict.assert_has_calls(infer_calls)
