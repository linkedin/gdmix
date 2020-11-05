import fastavro
import logging
import tensorflow as tf

from gdmix.util.io_utils import try_write_avro_blocks

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DetextWriter:
    """
    Helper class to run estimator style inference and write detext model and/or data
    """

    def __init__(self, schema_params):
        """
        Initialize the schema.

        Args:
            self: (todo): write your description
            schema_params: (dict): write your description
        """
        self.schema_params = schema_params

    def get_inference_output_avro_schema(self):
        """
        Gets inference schema.

        Args:
            self: (todo): write your description
        """
        schema = {
            'name': 'validation_result',
            'type': 'record',
            'fields': [
                {'name': self.schema_params.sample_id, 'type': 'long'},
                {'name': self.schema_params.sample_weight, 'type': 'float'},
                {'name': self.schema_params.label, 'type': 'int'},
                {'name': self.schema_params.prediction_score, 'type': 'float'}
            ],
        }
        return schema

    def append_validation_results(self, records, predicts, ids, labels, weights):
        """
        Applies a set of validation.

        Args:
            self: (todo): write your description
            records: (todo): write your description
            predicts: (todo): write your description
            ids: (list): write your description
            labels: (str): write your description
            weights: (array): write your description
        """
        batch_size = predicts.shape[0]
        assert predicts.shape[0] == ids.shape[0]
        assert predicts.shape[0] == labels.shape[0]
        assert predicts.shape[0] == weights.shape[0]
        for i in range(batch_size):
            # we only support pointwise training for detext
            # label is list of one scalar
            # score is also scalar
            record = {self.schema_params.prediction_score: predicts[i][0],
                      self.schema_params.sample_id: ids[i],
                      self.schema_params.label: int(labels[i][0]),
                      self.schema_params.sample_weight: weights[i]}
            records.append(record)
        return batch_size

    def save_batch(self, f, batch_score, output_file, n_records, n_batch):
        """
        Save a batch of - batch.

        Args:
            self: (todo): write your description
            f: (todo): write your description
            batch_score: (todo): write your description
            output_file: (str): write your description
            n_records: (int): write your description
            n_batch: (int): write your description
        """
        validation_results = []
        validation_schema = fastavro.parse_schema(self.get_inference_output_avro_schema())
        # save one batch of score
        try:
            predict_val = batch_score['scores']
            ids = batch_score['uid']
            labels = batch_score['label']
            weights = batch_score['weight']
            n_records += self.append_validation_results(validation_results,
                                                        predict_val,
                                                        ids,
                                                        labels,
                                                        weights)
            n_batch += 1
        except tf.errors.OutOfRangeError:
            logger.info(
                'Iterated through one batch. Finished evaluating work at batch {0}.'.format(n_batch))
        else:
            try_write_avro_blocks(f, validation_schema, validation_results, None,
                                  self.create_error_message(n_batch, output_file))
        return n_records, n_batch

    def create_error_message(self, n_batch, output_file):
        """
        Create an error message to the error message.

        Args:
            self: (todo): write your description
            n_batch: (str): write your description
            output_file: (str): write your description
        """
        err_msg = 'An error occurred while writing batch #{} to path {}'.format(
            n_batch, output_file)
        return err_msg
