import fastavro
import logging
import numpy as np
import tensorflow as tf
import time
from scipy.sparse import csr_matrix
from gdmix.util.io_utils import try_write_avro_blocks
from gdmix.util import constants
from gdmix.models.custom.scipy.utils import convert_to_training_jobs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PhotonMLWriter:
    """
    Helper class to run tf.Session() style inference and write model and/or data in Photon ML format
    """

    def __init__(self, schema_params):
        self.schema_params = schema_params
        self.inference_log_frequency = 100

    def get_inference_output_avro_schema(self, metadata, has_label, has_logits_per_coordinate, has_weight=False):
        schema = {
            'name': 'validation_result',
            'type': 'record',
            'fields': [
                {'name': self.schema_params[constants.SAMPLE_ID], 'type': 'long'},
                {'name': self.schema_params[constants.PREDICTION_SCORE], 'type': 'float'}
            ]
        }
        if has_label:
            schema.get('fields').append({'name': self.schema_params[constants.LABEL],
                                         'type': 'int'})
        if has_weight or metadata.get(self.schema_params[constants.SAMPLE_WEIGHT]) is not None:
            schema.get('fields').append({'name': self.schema_params[constants.SAMPLE_WEIGHT],
                                         'type': 'float'})
        if has_logits_per_coordinate:
            schema.get('fields').append({'name': self.schema_params[constants.PREDICTION_SCORE_PER_COORDINATE],
                                         'type': 'float'})
        return schema

    def append_validation_results(self, records, labels, predicts, metadata, predicts_per_coordinate=None):
        """
        Append validation results into records
        :param records:                 Records containing all the validation info
        :param labels:                  Ground truth label tensor
        :param predicts:                Prediction tensor
        :param metadata:                Metadata about the dataset
        :param predicts_per_coordinate: Prediction tensor without offset
        :return: Batch size
        """
        predicts = predicts.flatten()
        if labels is not None:
            labels = labels.flatten()
        if predicts_per_coordinate is not None:
            prediction_per_coordinate = predicts_per_coordinate.flatten()
        batch_size = predicts.size
        assert (batch_size == predicts.size)
        if self.schema_params[constants.SAMPLE_WEIGHT] in metadata:
            sample_weights = metadata[self.schema_params[constants.SAMPLE_WEIGHT]].flatten()
            assert (batch_size == sample_weights.size)
        if self.schema_params[constants.SAMPLE_ID] in metadata:
            sample_ids = metadata[self.schema_params[constants.SAMPLE_ID]].flatten()
            assert (batch_size == sample_ids.size)

        for i in range(batch_size):
            record = {self.schema_params[constants.PREDICTION_SCORE]: predicts[i]}
            if labels is not None:
                record[self.schema_params[constants.LABEL]] = labels[i]
            if predicts_per_coordinate is not None:
                record[self.schema_params[constants.PREDICTION_SCORE_PER_COORDINATE]] = prediction_per_coordinate[i]
            if self.schema_params[constants.SAMPLE_WEIGHT] in metadata:
                record[self.schema_params[constants.SAMPLE_WEIGHT]] = sample_weights[i]
            if self.schema_params[constants.SAMPLE_ID] in metadata:
                record[self.schema_params[constants.SAMPLE_ID]] = sample_ids[i]
            records.append(record)
        return batch_size

    def run_custom_scipy_re_inference(self, inference_dataset, model_coefficients, lr_model, metadata, tensor_metadata,
                                      output_file):
        """
        Run inference on custom LR RE model

        NOTE - currently this implementation only supports datasets with batch size = 1. A fix for supporting
        larger batch sizes will be added soon
        :param inference_dataset:      Dataset to run inference on
        :param model_coefficients:     Custom LR model coefficients
        :param lr_model:               Custom LR model object
        :param metadata:               Metadata for dataset
        :param tensor_metadata:        Processed metadata
        :param output_file:            Output AVRO file
        :return:    None
        """

        # Create dataset iterator
        iterator = tf.compat.v1.data.make_initializable_iterator(inference_dataset)
        features, labels = iterator.get_next()

        # Set up output schema
        validation_results = []
        has_label = self.schema_params[constants.LABEL] in labels
        has_logits_per_coordinate = True  # Always true for custom scipy-based LR
        has_weight = self.schema_params[constants.SAMPLE_WEIGHT] in [feature.name for feature in
                                                                     tensor_metadata.get_features()]
        validation_schema = fastavro.parse_schema(self.get_inference_output_avro_schema(metadata,
                                                                                        has_label,
                                                                                        has_logits_per_coordinate,
                                                                                        has_weight))

        num_features = next(filter(lambda x: x.name == self.schema_params[constants.FEATURE_BAGS][0],
                                   tensor_metadata.get_features())).shape[0]

        # Run session over the dataset, and write to output file
        with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(use_per_session_threads=True)) as sess:
            sess.run(iterator.initializer)
            n_batch = 0
            t0 = time.time()

            """
            For the first block, the file needs to be open in âwâ mode, while the
            rest of the blocks needs the âaâ mode. This restriction makes it
            necessary to open the files at least twice, one for the first block,
            one for the remaining. So itâs not possible to put them into the
            while loop within a file context.
            """

            with tf.io.gfile.GFile(output_file, 'wb') as f:
                f.seekable = lambda: False
                try:
                    # Extract and process raw entity data
                    features_val, labels_val = sess.run([features, labels])
                    # Re-use convert_to_training_jobs code to generate inference jobs
                    processed_inference_jobs = convert_to_training_jobs(features_val, labels_val,
                                                                        self.schema_params,
                                                                        num_features=num_features,
                                                                        enable_local_indexing=False)
                    # Run inference on all jobs
                    [self._run_batched_inference(lr_model, model_coefficients,
                                                 processed_inference_job, num_features, validation_results) for
                     processed_inference_job in processed_inference_jobs]
                    n_batch += 1
                except tf.errors.OutOfRangeError:
                    logger.info(
                        'Iterated through one batch. Finished evaluating work at batch {0}.'.format(n_batch))
                else:
                    try_write_avro_blocks(f, validation_schema, validation_results, None,
                                          self.create_error_message(n_batch, output_file))

            # write the remaining blocks
            with tf.io.gfile.GFile(output_file, 'ab+') as f:
                f.seek(0, 2)  # seek to the end of the file, 0 is offset, 2 means the end of file
                f.seekable = lambda: True
                f.readable = lambda: True
                # loop through each batch of data
                while True:
                    try:
                        # Extract and process raw entity data
                        features_val, labels_val = sess.run([features, labels])
                        # Re-use convert_to_training_jobs code to generate inference jobs
                        processed_inference_jobs = convert_to_training_jobs(features_val, labels_val,
                                                                            self.schema_params,
                                                                            num_features=num_features,
                                                                            enable_local_indexing=False)
                        # Run inference on all jobs
                        [self._run_batched_inference(lr_model, model_coefficients,
                                                     processed_inference_job, num_features, validation_results) for
                         processed_inference_job in processed_inference_jobs]
                        n_batch += 1
                    except tf.errors.OutOfRangeError:
                        logger.info(
                            'Iterated through all batches. Finished evaluating work at batch {0}.'.format(n_batch))
                        break
                    if n_batch % self.inference_log_frequency == 0:
                        delta_time = float(time.time() - t0)
                        local_speed = float(n_batch) / delta_time
                        suc_msg = "nbatch = {0}, deltaT = {1:0.2f} seconds, speed = {2:0.2f} batches/sec".format(
                            n_batch, delta_time, local_speed)
                        try_write_avro_blocks(f, validation_schema, validation_results, suc_msg,
                                              self.create_error_message(n_batch, output_file))
                if len(validation_results) > 0:
                    # save the final part
                    try_write_avro_blocks(f, validation_schema, validation_results, None,
                                          self.create_error_message(n_batch, output_file))

    def _run_batched_inference(self, lr_model, model_coefficients, processed_inference_job, num_features,
                               validation_results):
        if processed_inference_job.entity_id in model_coefficients:
            if self.schema_params[constants.ENABLE_LOCAL_INDEXING]:
                # Convert locally indexed weights to global space. Since global indices are shifted by one because of
                # the bias term, increase global index values by 1
                locally_indexed_custom_theta = model_coefficients[processed_inference_job.entity_id].training_result
                unique_global_indices = model_coefficients[processed_inference_job.entity_id].unique_global_indices + 1
                cols = np.hstack((0, unique_global_indices))
                rows = np.zeros(cols.shape, dtype=int)
                custom_theta = csr_matrix((locally_indexed_custom_theta, (rows, cols)), shape=(1, num_features+1)).T
            else:
                custom_theta = model_coefficients[processed_inference_job.entity_id].training_result

            logits = lr_model.predict_proba(X=processed_inference_job.X,
                                            offsets=processed_inference_job.offsets,
                                            custom_theta=custom_theta,
                                            return_logits=True)
        else:
            logits = processed_inference_job.offsets
        logits_per_coordinate = logits - processed_inference_job.offsets
        metadata_val = {self.schema_params[constants.SAMPLE_WEIGHT]: processed_inference_job.weights,
                        self.schema_params[constants.SAMPLE_ID]: processed_inference_job.ids}
        return self.append_validation_results(validation_results,
                                              processed_inference_job.y,
                                              logits, metadata_val, logits_per_coordinate)

    @staticmethod
    def create_error_message(n_batch, output_file) -> str:
        return f'An error occurred while writing batch #{n_batch} to path {output_file}'
