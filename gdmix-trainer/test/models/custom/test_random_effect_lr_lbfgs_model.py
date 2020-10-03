import os
import tempfile
import tensorflow as tf

from drivers.test_helper import setup_fake_raw_model_params, \
    setup_fake_base_training_params, setup_fake_schema_params
from fastavro import reader
from gdmix.util import constants
from gdmix.models.custom.random_effect_lr_lbfgs_model import RandomEffectLRLBFGSModel

test_dataset_path = os.path.join(os.getcwd(), "test/resources/grouped_per_member_train")
fake_feature_file = "fake_feature_file.csv"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestRandomEffectCustomLRModel(tf.test.TestCase):
    """
    Test for random effect custom LR model
    """

    def get_raw_params(self, partition_entity='memberId', num_of_lbfgs_iterations=None):
        base_training_params = setup_fake_base_training_params(training_stage=constants.RANDOM_EFFECT)
        base_training_params.batch_size = 2
        # flatten the params
        raw_params = list(base_training_params.__to_argv__())
        model_params = setup_fake_raw_model_params(training_stage=constants.RANDOM_EFFECT)
        raw_params.extend(model_params)
        raw_params.extend(['--' + constants.MODEL_IDS_DIR, test_dataset_path])
        raw_params.extend(['--' + constants.FEATURE_FILE, os.path.join(test_dataset_path, fake_feature_file)])
        raw_params.extend(['--' + constants.PARTITION_ENTITY, partition_entity])
        raw_params.extend(['--' + constants.LABEL, 'response'])
        raw_params.extend(['--' + constants.L2_REG_WEIGHT, '0.1'])
        if num_of_lbfgs_iterations:
            raw_params.extend(['--' + constants.NUM_OF_LBFGS_ITERATIONS, f'{num_of_lbfgs_iterations}'])
        return base_training_params, raw_params

    def test_train_should_fail_if_producer_or_consumer_fails(self):

        # Create raw params with fake partition entity
        base_training_params, raw_params = self.get_raw_params(partition_entity='fake_partition_entity')
        avro_model_output_dir = tempfile.mkdtemp()
        raw_params.extend(['--' + constants.MODEL_OUTPUT_DIR, avro_model_output_dir])

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)
        assert re_lr_model

        checkpoint_dir = tempfile.mkdtemp()
        training_context = {constants.PARTITION_INDEX: 0,
                            constants.PASSIVE_TRAINING_DATA_PATH: test_dataset_path}
        schema_params = setup_fake_schema_params()

        # Training should fail as partition entity doesnt exist in dataset
        with self.assertRaises(Exception):
            re_lr_model.train(training_data_path=test_dataset_path, validation_data_path=test_dataset_path,
                              metadata_file=os.path.join(test_dataset_path, "data.json"),
                              checkpoint_path=checkpoint_dir, execution_context=training_context,
                              schema_params=schema_params)
        tf.io.gfile.rmtree(checkpoint_dir)
        tf.io.gfile.rmtree(avro_model_output_dir)

    def test_train_and_predict(self):

        # Create and add AVRO model output directory to raw parameters
        base_training_params, raw_params = self.get_raw_params()
        avro_model_output_dir = tempfile.mkdtemp()
        raw_params.extend(['--' + constants.MODEL_OUTPUT_DIR, avro_model_output_dir])
        raw_params.extend(['--' + constants.ENABLE_LOCAL_INDEXING, 'True'])

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)
        assert re_lr_model

        # TEST 1 - Training (with scoring)
        checkpoint_dir = tempfile.mkdtemp()
        active_train_fd, active_train_output_file = tempfile.mkstemp()
        passive_train_fd, passive_train_output_file = tempfile.mkstemp()
        training_context = {constants.ACTIVE_TRAINING_OUTPUT_FILE: active_train_output_file,
                            constants.PASSIVE_TRAINING_OUTPUT_FILE: passive_train_output_file,
                            constants.PARTITION_INDEX: 0,
                            constants.PASSIVE_TRAINING_DATA_PATH: test_dataset_path}
        schema_params = setup_fake_schema_params()
        re_lr_model.train(training_data_path=test_dataset_path, validation_data_path=test_dataset_path,
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
        predict_output_dir = tempfile.mkdtemp()
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

        # remove the temp dir(s) and file(s).
        os.close(active_train_fd)
        tf.io.gfile.remove(active_train_output_file)
        os.close(passive_train_fd)
        tf.io.gfile.remove(passive_train_output_file)
        tf.io.gfile.rmtree(avro_model_output_dir)
        tf.io.gfile.rmtree(checkpoint_dir)
        tf.io.gfile.rmtree(predict_output_dir)

    def test_warm_start(self):

        # Step 1: train an initial model
        # Create and add AVRO model output directory to raw parameters
        base_training_params, raw_params = self.get_raw_params()
        avro_model_output_dir = tempfile.mkdtemp()
        raw_params.extend(['--' + constants.MODEL_OUTPUT_DIR, avro_model_output_dir])

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)

        # Initial training to get the warm start model
        checkpoint_dir = tempfile.mkdtemp()
        active_train_fd, active_train_output_file = tempfile.mkstemp()
        passive_train_fd, passive_train_output_file = tempfile.mkstemp()
        training_context = {constants.ACTIVE_TRAINING_OUTPUT_FILE: active_train_output_file,
                            constants.PASSIVE_TRAINING_OUTPUT_FILE: passive_train_output_file,
                            constants.PARTITION_INDEX: 0,
                            constants.PASSIVE_TRAINING_DATA_PATH: test_dataset_path}
        schema_params = setup_fake_schema_params()
        re_lr_model.train(training_data_path=test_dataset_path, validation_data_path=test_dataset_path,
                          metadata_file=os.path.join(test_dataset_path, "data.json"), checkpoint_path=checkpoint_dir,
                          execution_context=training_context, schema_params=schema_params)

        avro_model_output_file = os.path.join(avro_model_output_dir, f"part-{0:05d}.avro")
        # Read back the model as the warm start initial point.
        initial_model = re_lr_model._load_weights(avro_model_output_file, 0)

        # Step 2: Train for 1 l-bfgs step with warm start
        base_training_params, raw_params = self.get_raw_params('memberId', 1)
        raw_params.extend(['--' + constants.MODEL_OUTPUT_DIR, avro_model_output_dir])

        # Create random effect LR LBFGS Model
        re_lr_model = RandomEffectLRLBFGSModel(raw_model_params=raw_params)

        schema_params = setup_fake_schema_params()
        re_lr_model.train(training_data_path=test_dataset_path, validation_data_path=test_dataset_path,
                          metadata_file=os.path.join(test_dataset_path, "data.json"), checkpoint_path=checkpoint_dir,
                          execution_context=training_context, schema_params=schema_params)
        final_model = re_lr_model._load_weights(avro_model_output_file, 0)

        # Check the model has already converged.
        self.assertEqual(len(initial_model), len(final_model))
        for model_id in initial_model:
            self.assertAllClose(initial_model[model_id].theta,
                                final_model[model_id].theta,
                                msg='models mismatch')

        # Step 3: Train for 1 l-bfgs step with cold start
        # Remove the model file to stop the warm start
        model_files = tf.io.gfile.glob(os.path.join(avro_model_output_dir, '*.avro'))
        for f in model_files:
            tf.io.gfile.remove(f)
        # Train for 1 l-bfgs step.
        re_lr_model.train(training_data_path=test_dataset_path, validation_data_path=test_dataset_path,
                          metadata_file=os.path.join(test_dataset_path, "data.json"), checkpoint_path=checkpoint_dir,
                          execution_context=training_context, schema_params=schema_params)
        cold_model = re_lr_model._load_weights(avro_model_output_file, 0)

        # Check the model has already converged.
        self.assertEqual(len(cold_model), len(final_model))
        for model_id in cold_model:
            self.assertNotAllClose(cold_model[model_id].theta,
                                   final_model[model_id].theta,
                                   msg='models should not be close')

        # remove the temp dir(s) and file(s).
        os.close(active_train_fd)
        tf.io.gfile.remove(active_train_output_file)
        os.close(passive_train_fd)
        tf.io.gfile.remove(passive_train_output_file)
        tf.io.gfile.rmtree(avro_model_output_dir)
        tf.io.gfile.rmtree(checkpoint_dir)
