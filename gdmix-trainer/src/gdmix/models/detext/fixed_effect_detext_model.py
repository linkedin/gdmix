import logging
import os

import detext.run_detext as detext_driver
import tensorflow as tf
from detext.run_detext import DetextArg
from detext.train.data_fn import input_fn_tfrecord
from detext.train import train_flow_helper, train_model_helper
from detext.utils import parsing_utils
from gdmix.models.api import Model
from gdmix.models.detext_writer import DetextWriter
from gdmix.util import constants
from gdmix.util.distribution_utils import shard_input_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FixedEffectDetextModel(Model):
    """
    DeText model class to support fixed-effect training.
    """

    def __init__(self, raw_model_params):
        super().__init__(raw_model_params)
        self.checkpoint_path = self.model_params.out_dir
        #  training_data_dir and validation_data_dir are used by the driver
        self.training_data_dir = self.model_params.train_file
        self.validation_data_dir = self.model_params.dev_file
        self.best_checkpoint = train_flow_helper.get_export_model_variable_dir(self.checkpoint_path)

    def train(self,
              training_data_dir,
              validation_data_dir,
              metadata_file,
              checkpoint_path,
              execution_context,
              schema_params):
        # Delegate to super class
        detext_driver.run_detext(self.model_params)

    def predict(self,
                output_dir,
                input_data_path,
                metadata_file,
                checkpoint_path,
                execution_context,
                schema_params):
        n_records = 0
        n_batch = 0
        # Predict on the dataset
        sharded_dataset_paths, file_level_sharding = shard_input_files(input_data_path,
                                                                       execution_context[constants.NUM_SHARDS],
                                                                       execution_context[constants.SHARD_INDEX])
        if file_level_sharding and len(sharded_dataset_paths) == 0:
            logger.info("No input dataset is found, returning...")
            return

        inference_dataset = input_fn_tfrecord(input_pattern=','.join(sharded_dataset_paths),  # noqa: E731
                                              batch_size=self.model_params.test_batch_size,
                                              mode=tf.estimator.ModeKeys.EVAL,
                                              feature_map=self.model_params.feature_map)

        self.model = train_model_helper.load_model_with_ckpt(
            parsing_utils.HParams(**self.model_params._asdict()),
            self.best_checkpoint)
        output = train_flow_helper.predict_with_additional_info(inference_dataset,
                                                                self.model,
                                                                self.model_params.feature_map)
        detext_writer = DetextWriter(schema_params=schema_params)
        shard_index = execution_context[constants.SHARD_INDEX]
        output_file = os.path.join(output_dir, f"part-{shard_index:05d}.avro")
        for batch_score in output:
            if n_batch == 0:
                with tf.io.gfile.GFile(output_file, 'wb') as f:
                    f.seekable = lambda: False
                    n_records, n_batch = detext_writer.save_batch(f, batch_score, output_file,
                                                                  n_records, n_batch)
            else:
                with tf.io.gfile.GFile(output_file, 'ab+') as f:
                    f.seek(0, 2)
                    f.seekable = lambda: True
                    f.readable = lambda: True
                    n_records, n_batch = detext_writer.save_batch(f, batch_score, output_file,
                                                                  n_records, n_batch)
        logger.info(f"{n_batch} records, e.g. {n_records} records inferenced")

    def export(self, output_model_dir):
        logger.info("Detext has built-in export operations during training")

    def _parse_parameters(self, raw_model_parameters):
        return DetextArg.__from_argv__(raw_model_parameters, error_on_unknown=False)
