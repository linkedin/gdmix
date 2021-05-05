import os
from scipy.sparse import coo_matrix
import tempfile
import tensorflow as tf

from drivers.test_helper import setup_fake_raw_model_params, \
    setup_fake_base_training_params, setup_fake_schema_params
from fastavro import reader
from gdmix.util import constants
from gdmix.util.io_utils import low_rpc_call_glob
from gdmix.models.custom.random_effect_lr_lbfgs_model import RandomEffectLRLBFGSModel
from models.custom.test_optimizer_helper import compute_coefficients_and_variance

test_dataset_path = os.path.join(os.getcwd(), "test/resources/grouped_per_member_train")
fake_feature_file = "fake_feature_file.csv"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestRandomEffectCustomLRModel(tf.test.TestCase):
    """
    Test for random effect custom LR model
    """
    def setUp(self):
        tf.compat.v1.reset_default_graph()
        self.base_dir = tempfile.mkdtemp()

    def tearDown(self):
        tf.io.gfile.rmtree(self.base_dir)

    def get_raw_params(self, partition_entity='memberId', num_of_lbfgs_iterations=None, intercept_only=False):
        base_training_params = setup_fake_base_training_params(training_stage=constants.RANDOM_EFFECT)
        base_training_params.batch_size = 2
        # flatten the params
        raw_params = list(base_training_params.__to_argv__())
        model_params = setup_fake_raw_model_params(training_stage=constants.RANDOM_EFFECT)
        raw_params.extend(model_params)
        raw_params.extend(['--' + constants.MODEL_IDS_DIR, test_dataset_path])
        raw_params.extend(['--' + constants.FEATURE_FILE, os.path.join(test_dataset_path, fake_feature_file)])
        raw_params.extend(['--' + constants.PARTITION_ENTITY, partition_entity])
        raw_params.extend(['--' + constants.LABEL_COLUMN_NAME, 'response'])
        raw_params.extend(['--' + constants.L2_REG_WEIGHT, '0.1'])
        if num_of_lbfgs_iterations:
            raw_params.extend(['--' + constants.NUM_OF_LBFGS_ITERATIONS, f'{num_of_lbfgs_iterations}'])
        if intercept_only:
            feature_bag_index = raw_params.index(f'--{constants.FEATURE_BAG}')
            raw_params.pop(feature_bag_index)
            raw_params.pop(feature_bag_index)
            assert(f'--{constants.FEATURE_BAG}' not in raw_params)
            assert('per_member' not in raw_params)
        return raw_params

    def test_train_should_fail_if_producer_or_consumer_fails(self):

        # Create raw params with fake partition entity
        raw_params = self.get_raw_params(partition_entity='fake_partition_entity')
        avro_model_output_dir = tempfile.mkdtemp(dir=self.base_dir)
        raw_params.extend(['--' + constants.OUTPUT_MODEL_DIR, avro_model_output_dir])

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)
        assert re_lr_model

        checkpoint_dir = tempfile.mkdtemp(dir=self.base_dir)
        training_context = {constants.PARTITION_INDEX: 0,
                            constants.PASSIVE_TRAINING_DATA_DIR: test_dataset_path}
        schema_params = setup_fake_schema_params()

        # Training should fail as partition entity doesnt exist in dataset
        with self.assertRaises(Exception):
            re_lr_model.train(training_data_dir=test_dataset_path, validation_data_dir=test_dataset_path,
                              metadata_file=os.path.join(test_dataset_path, "data.json"),
                              checkpoint_path=checkpoint_dir, execution_context=training_context,
                              schema_params=schema_params)

    def test_train_and_predict(self):

        # Create and add AVRO model output directory to raw parameters
        raw_params = self.get_raw_params()
        avro_model_output_dir = tempfile.mkdtemp(dir=self.base_dir)
        raw_params.extend(['--' + constants.OUTPUT_MODEL_DIR, avro_model_output_dir])
        raw_params.extend(['--' + constants.ENABLE_LOCAL_INDEXING, 'True'])

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)
        assert re_lr_model

        # TEST 1 - Training (with scoring)
        checkpoint_dir = tempfile.mkdtemp(dir=self.base_dir)
        active_train_fd, active_train_output_file = tempfile.mkstemp(dir=self.base_dir)
        passive_train_fd, passive_train_output_file = tempfile.mkstemp(dir=self.base_dir)
        training_context = {constants.ACTIVE_TRAINING_OUTPUT_FILE: active_train_output_file,
                            constants.PASSIVE_TRAINING_OUTPUT_FILE: passive_train_output_file,
                            constants.PARTITION_INDEX: 0,
                            constants.PASSIVE_TRAINING_DATA_DIR: test_dataset_path}
        schema_params = setup_fake_schema_params()
        re_lr_model.train(training_data_dir=test_dataset_path, validation_data_dir=test_dataset_path,
                          metadata_file=os.path.join(test_dataset_path, "data.json"), checkpoint_path=checkpoint_dir,
                          execution_context=training_context, schema_params=schema_params)

        # Cycle through model AVRO output and assert each record is a dictionary
        with open(os.path.join(avro_model_output_dir, f"part-{0:05d}.avro"), 'rb') as fo:
            for record in reader(fo):
                self.assertTrue(isinstance(record, dict))

        # Cycle through output file and assert each record is a dictionary
        with open(active_train_output_file, 'rb') as fo:
            for record in reader(fo):
                self.assertTrue(isinstance(record, dict))
        with open(passive_train_output_file, 'rb') as fo:
            for record in reader(fo):
                self.assertTrue(isinstance(record, dict))

        # TEST 2 - Cold prediction
        predict_output_dir = tempfile.mkdtemp(dir=self.base_dir)
        re_lr_model.predict(output_dir=predict_output_dir, input_data_path=test_dataset_path,
                            metadata_file=os.path.join(test_dataset_path, "data.json"),
                            checkpoint_path=avro_model_output_dir,
                            execution_context=training_context, schema_params=schema_params)
        with open(os.path.join(predict_output_dir, "part-{0:05d}.avro".format(0)), 'rb') as fo:
            for record in reader(fo):
                self.assertTrue(isinstance(record, dict))

        # TEST 3 - Assert scoring-while-training and cold prediction produce same output
        with open(active_train_output_file, 'rb') as fo:
            active_training_records = [record for record in reader(fo)]
        with open(os.path.join(predict_output_dir, "part-{0:05d}.avro".format(0)), 'rb') as fo:
            prediction_records = [record for record in reader(fo)]
        for active_training_record, prediction_record in zip(active_training_records, prediction_records):
            self.assertEqual(active_training_record, prediction_record)

    def _check_intercept_only_model(self, models):
        """
        Check the intercept only model.
        The coefficients should be an exact length-2 array, the second element is 0.0.
        The unique_global_indices is [0].
        :param models: a dictionary of {entity: TrainingResult}.
        :return: None
        """
        for model_id in models:
            theta = models[model_id].theta
            indices = models[model_id].unique_global_indices
            self.assertEqual(len(theta), 2)
            self.assertEqual(theta[1], 0.0)
            self.assertEqual(indices, [0])

    def _create_dataset(self, dataset_idx):
        if dataset_idx == 1:
            bytes_member_ids = [b'xyz']
            member_ids = ['xyz']
            offsets = [[1.0, 2.0, 3.0, -1.0, 0.3, -0.7]]
            responses = [[1, 0, 1, 0, 0, 1]]
            uids = [[1, 2, 3, 4, 5, 6]]
            weights = [[1.0, 0.8, 2.0, 3.0, 2.1, 1.7]]
            per_member_indices = [[[0, 2], [0, 1], [1, 2], [0, 2], [1, 2], [0, 1]]]
            per_member_values = [[[0.55, -0.95], [0.22, -1.05], [0.90, 0.50],
                                  [1.99, 0.48], [0.37, -1.64], [0.33, 0.17]]]
        elif dataset_idx == 2:
            bytes_member_ids = [b'abc102', b'zyz234']
            member_ids = ['abc102', 'zyz234']
            offsets = [[1.0, 2.0], [-1.0, -2.0]]
            responses = [[1, 0], [0, 0]]
            uids = [[1234, 5678], [1345, 3214]]
            weights = [[1.0, 0.8], [0.5, 0.74]]
            per_member_indices = [[[1, 5, 10], [1, 50, 99]], [[1, 3], [2, 20]]]
            per_member_values = [[[0.3, -2.3, 0.9], [1.4, 99.8, -1.2]], [[1.23, 4.5], [-1.0, 3.0]]]
        else:
            raise ValueError(f'Unknown dataset index {dataset_idx}, it has to be 1 or 2')
        dataset = {'bytes_member_ids': bytes_member_ids, 'member_ids': member_ids,
                   'offsets': offsets, 'responses': responses, 'uids': uids, 'weights': weights,
                   'per_member_indices': per_member_indices, 'per_member_values': per_member_values}
        return dataset

    def _create_dataset_with_string_entity_id(self, dataset_idx, filename):
        dataset = self._create_dataset(dataset_idx)
        bytes_member_ids = dataset['bytes_member_ids']
        member_ids = dataset['member_ids']
        offsets = dataset['offsets']
        responses = dataset['responses']
        uids = dataset['uids']
        weights = dataset['weights']
        per_member_indices = dataset['per_member_indices']
        per_member_values = dataset['per_member_values']

        with tf.io.TFRecordWriter(filename) as file_writer:
            for i in range(len(bytes_member_ids)):
                context = tf.train.Features(feature={
                    'memberId': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_member_ids[i]])),
                    'offset': tf.train.Feature(float_list=tf.train.FloatList(value=offsets[i])),
                    'response': tf.train.Feature(int64_list=tf.train.Int64List(value=responses[i])),
                    'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=uids[i])),
                    'weight': tf.train.Feature(float_list=tf.train.FloatList(value=weights[i]))
                })
                index_list, value_list = [], []
                for j in range(len(uids[i])):
                    index_list.append(
                        tf.train.Feature(int64_list=tf.train.Int64List(value=per_member_indices[i][j])))
                    value_list.append(
                        tf.train.Feature(float_list=tf.train.FloatList(value=per_member_values[i][j])))
                feature_lists = tf.train.FeatureLists(feature_list={
                    'per_member_indices': tf.train.FeatureList(feature=index_list),
                    'per_member_values': tf.train.FeatureList(feature=value_list)
                })

                sequence_example = tf.train.SequenceExample(context=context, feature_lists=feature_lists)
                file_writer.write(sequence_example.SerializeToString())
        return member_ids

    def _run_warm_start(self, string_entity_id, intercept_only, enable_local_index):

        # Step 1: train an initial model
        # Create and add AVRO model output directory to raw parameters
        raw_params = self.get_raw_params(intercept_only=intercept_only)
        avro_model_output_dir = tempfile.mkdtemp(dir=self.base_dir)
        raw_params.extend(['--' + constants.OUTPUT_MODEL_DIR, avro_model_output_dir])
        if enable_local_index:
            raw_params.extend(['--' + constants.ENABLE_LOCAL_INDEXING, 'True'])

        train_data_dir = test_dataset_path
        if string_entity_id:
            train_tfrecord_dir = tempfile.mkdtemp(dir=self.base_dir)
            train_tfrecord_file = os.path.join(train_tfrecord_dir, 'train.tfrecord')
            # create dataset with string entity id
            model_ids = self._create_dataset_with_string_entity_id(2, train_tfrecord_file)
            train_data_dir = train_tfrecord_dir
            # set up metadata file
            metadata_file = os.path.join(test_dataset_path, "data_with_string_entity_id.json")
        elif intercept_only:
            metadata_file = os.path.join(train_data_dir, "data_intercept_only.json")
        else:
            metadata_file = os.path.join(train_data_dir, "data.json")

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)

        # Initial training to get the warm start model
        checkpoint_dir = tempfile.mkdtemp(dir=self.base_dir)
        active_train_fd, active_train_output_file = tempfile.mkstemp(dir=self.base_dir)
        passive_train_fd, passive_train_output_file = tempfile.mkstemp(dir=self.base_dir)
        training_context = {constants.ACTIVE_TRAINING_OUTPUT_FILE: active_train_output_file,
                            constants.PASSIVE_TRAINING_OUTPUT_FILE: passive_train_output_file,
                            constants.PARTITION_INDEX: 0,
                            constants.PASSIVE_TRAINING_DATA_DIR: train_data_dir}
        schema_params = setup_fake_schema_params()
        re_lr_model.train(training_data_dir=train_data_dir, validation_data_dir=train_data_dir,
                          metadata_file=metadata_file, checkpoint_path=checkpoint_dir,
                          execution_context=training_context, schema_params=schema_params)

        avro_model_output_file = os.path.join(avro_model_output_dir, f"part-{0:05d}.avro")
        # Read back the model as the warm start initial point.
        initial_model = re_lr_model._load_weights(avro_model_output_file, False)
        if intercept_only:
            self._check_intercept_only_model(initial_model)

        # Step 2: Train for 1 l-bfgs step with warm start
        raw_params = self.get_raw_params('memberId', 1, intercept_only)
        raw_params.extend(['--' + constants.OUTPUT_MODEL_DIR, avro_model_output_dir])
        if enable_local_index:
            raw_params.extend(['--' + constants.ENABLE_LOCAL_INDEXING, 'True'])

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)

        schema_params = setup_fake_schema_params()
        re_lr_model.train(training_data_dir=train_data_dir, validation_data_dir=train_data_dir,
                          metadata_file=metadata_file, checkpoint_path=checkpoint_dir,
                          execution_context=training_context, schema_params=schema_params)
        final_model = re_lr_model._load_weights(avro_model_output_file, False)

        if intercept_only:
            self._check_intercept_only_model(final_model)
        # Check the model has already converged.
        self.assertEqual(len(initial_model), len(final_model))
        for model_id in initial_model:
            if string_entity_id:
                # make sure the model is is string not bytes.
                self.assertTrue(model_id in model_ids)
            self.assertAllClose(initial_model[model_id].theta,
                                final_model[model_id].theta,
                                msg='models mismatch')

        # Step 3: Train for 1 l-bfgs step with cold start
        # Remove the model file to stop the warm start
        model_files = low_rpc_call_glob(os.path.join(avro_model_output_dir, '*.avro'))
        for f in model_files:
            tf.io.gfile.remove(f)
        # Train for 1 l-bfgs step.
        re_lr_model.train(training_data_dir=train_data_dir, validation_data_dir=train_data_dir,
                          metadata_file=metadata_file, checkpoint_path=checkpoint_dir,
                          execution_context=training_context, schema_params=schema_params)
        cold_model = re_lr_model._load_weights(avro_model_output_file, False)

        if intercept_only:
            self._check_intercept_only_model(cold_model)

        # Check the models are different.
        self.assertEqual(len(cold_model), len(final_model))
        for model_id in cold_model:
            if string_entity_id:
                # make sure the model is is string not bytes.
                self.assertTrue(model_id in model_ids)
            self.assertNotAllClose(cold_model[model_id].theta,
                                   final_model[model_id].theta,
                                   msg='models should not be close')

    def test_warm_start_global_indexing(self):
        self._run_warm_start(string_entity_id=False, intercept_only=False, enable_local_index=False)

    def test_warm_start_local_indexing(self):
        self._run_warm_start(string_entity_id=False, intercept_only=False, enable_local_index=True)

    def test_warm_start_intercept_only_global_indexing(self):
        self._run_warm_start(string_entity_id=False, intercept_only=True, enable_local_index=False)

    def test_warm_start_intercept_only_local_indexing(self):
        self._run_warm_start(string_entity_id=False, intercept_only=True, enable_local_index=True)

    def test_warm_start_string_entity_id_global_indexing(self):
        self._run_warm_start(string_entity_id=True, intercept_only=False, enable_local_index=False)

    def test_warm_start_string_entity_id_local_indexing(self):
        self._run_warm_start(string_entity_id=True, intercept_only=False, enable_local_index=True)

    def test_model_with_variance(self):
        dataset_idx = 1
        variance_mode = constants.FULL
        # Create training data
        raw_params = self.get_raw_params()
        avro_model_output_dir = tempfile.mkdtemp(dir=self.base_dir)
        raw_params.extend(['--' + constants.OUTPUT_MODEL_DIR, avro_model_output_dir])
        raw_params.extend(['--' + constants.ENABLE_LOCAL_INDEXING, 'True'])
        raw_params.extend(['--variance_mode', variance_mode])

        # Replace the feature file
        feature_idx = raw_params.index('--' + constants.FEATURE_FILE)
        del raw_params[feature_idx:feature_idx+2]
        raw_params.extend(['--' + constants.FEATURE_FILE,
                           os.path.join(test_dataset_path, 'dataset_1_feature_file.csv')])

        # For this test, we need set l2 to 0.0. See the comments in test_optimizer_helper
        l2_idx = raw_params.index('--' + constants.L2_REG_WEIGHT)
        del raw_params[l2_idx:l2_idx+2]
        raw_params.extend(['--' + constants.L2_REG_WEIGHT, '0.0'])

        train_tfrecord_dir = tempfile.mkdtemp(dir=self.base_dir)
        train_tfrecord_file = os.path.join(train_tfrecord_dir, 'train.tfrecord')
        # Create dataset with string entity id
        model_ids = self._create_dataset_with_string_entity_id(dataset_idx, train_tfrecord_file)
        train_data_dir = train_tfrecord_dir
        # set up metadata file
        metadata_file = os.path.join(test_dataset_path, "dataset_1.json")

        # Create random effect LR LBFGS Model
        trainer = RandomEffectLRLBFGSModel(raw_model_params=raw_params)

        # train the model
        checkpoint_dir = tempfile.mkdtemp(dir=self.base_dir)
        active_train_fd, active_train_output_file = tempfile.mkstemp(dir=self.base_dir)
        passive_train_fd, passive_train_output_file = tempfile.mkstemp(dir=self.base_dir)
        training_context = {constants.ACTIVE_TRAINING_OUTPUT_FILE: active_train_output_file,
                            constants.PASSIVE_TRAINING_OUTPUT_FILE: passive_train_output_file,
                            constants.PARTITION_INDEX: 0,
                            constants.PASSIVE_TRAINING_DATA_DIR: train_data_dir}
        schema_params = setup_fake_schema_params()
        trainer.train(training_data_dir=train_data_dir, validation_data_dir=train_data_dir,
                      metadata_file=metadata_file, checkpoint_path=checkpoint_dir,
                      execution_context=training_context, schema_params=schema_params)
        avro_model_output_file = os.path.join(avro_model_output_dir, f"part-{0:05d}.avro")

        # Read back the model as the warm start initial point.
        model = trainer._load_weights(avro_model_output_file, False)
        actual_mean = model[model_ids[0]].theta
        actual_variance = model[model_ids[0]].variance

        # Get expected model coefficients and variance
        dataset = self._create_dataset(dataset_idx)
        offsets = dataset['offsets'][0]
        y = dataset['responses'][0]
        weights = dataset['weights'][0]
        per_member_indices = dataset['per_member_indices'][0]
        per_member_values = dataset['per_member_values'][0]
        # Convert per-member features to COO matrix
        rows = []
        cols = []
        vals = []
        nrows = len(per_member_indices)
        for ridx in range(len(per_member_indices)):
            for cidx in range(len(per_member_indices[ridx])):
                rows.append(ridx)
                cols.append(per_member_indices[ridx][cidx])
                vals.append(per_member_values[ridx][cidx])
        X = coo_matrix((vals, (rows, cols)), shape=(nrows, 3))
        expected = compute_coefficients_and_variance(X=X, y=y,
                                                     weights=weights, offsets=offsets,
                                                     variance_mode=variance_mode)
        # Compare
        self.assertAllClose(expected[0], actual_mean, rtol=1e-04, atol=1e-04,
                            msg='Mean mismatch')
        self.assertAllClose(expected[1], actual_variance, rtol=1e-04, atol=1e-04,
                            msg='Variance mismatch')
