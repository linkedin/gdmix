import logging
import os

import detext.run_detext as detext_driver
import detext.train.train as detext_train
import detext.utils.misc_utils as detext_utils
import tensorflow as tf
from detext.run_detext import DetextArg
from detext.train.data_fn import input_fn
from detext.utils import vocab_utils
from gdmix.models.api import Model
from gdmix.models.detext_writer import DetextWriter
from gdmix.util import constants
from gdmix.util.distribution_utils import shard_input_files
from tensorflow.contrib.training import HParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class FixedEffectDetextEstimatorModel(Model):
    """
    TF Estimator-based model class to support fixed-effect training.
    """
    __BEST = 'best_'

    def __init__(self, raw_model_params):
        super().__init__(raw_model_params)
        self.checkpoint_path = self.model_params.out_dir
        #  training_data_dir and validation_data_dir are used by the driver
        self.training_data_dir = self.model_params.train_file
        self.validation_data_dir = self.model_params.dev_file
        self.best_checkpoint = os.path.join(
            self.checkpoint_path, FixedEffectDetextEstimatorModel.__BEST + self.model_params.pmetric)

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

        inference_dataset = lambda: input_fn(input_pattern=','.join(sharded_dataset_paths),  # noqa: E731
                                             # DeText uses metadata_path
                                             metadata_path=self.model_params.metadata_path,
                                             batch_size=self.model_params.test_batch_size,
                                             mode=tf.estimator.ModeKeys.EVAL,
                                             vocab_table=vocab_utils.read_tf_vocab(
                                                 self.model_params.vocab_file, self.model_params.UNK),
                                             vocab_table_for_id_ftr=vocab_utils.read_tf_vocab(
                                                 self.model_params.vocab_file_for_id_ftr,
                                                 self.model_params.UNK_FOR_ID_FTR),
                                             feature_names=self.model_params.feature_names,
                                             CLS=self.model_params.CLS, SEP=self.model_params.SEP,
                                             PAD=self.model_params.PAD,
                                             PAD_FOR_ID_FTR=self.model_params.PAD_FOR_ID_FTR,
                                             max_len=self.model_params.max_len,
                                             min_len=self.model_params.min_len,
                                             cnn_filter_window_size=max(
                                                 self.model_params.filter_window_sizes)
                                             if self.model_params.ftr_ext == 'cnn' else 0)

        self.estimator_based_model = detext_train.get_estimator(detext_utils.extend_hparams(HParams(**self.model_params._asdict())),
                                                                strategy=None,  # local mode
                                                                best_checkpoint=self.best_checkpoint)
        output = self.estimator_based_model.predict(inference_dataset, yield_single_examples=False)
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
