import logging
from collections import namedtuple
from multiprocessing.process import current_process

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

from gdmix.util.io_utils import dataset_reader

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a named tuple to represent training result
TrainingResult = namedtuple('TrainingResult', ('theta', 'unique_global_indices'))
Job = namedtuple('Job', 'entity_id X y offsets weights ids unique_global_indices theta')
_CONSUMER_LOGGING_FREQUENCY = 1000
INDICES_SUFFIX = '_indices'
VALUES_SUFFIX = '_values'


class TrainingJobConsumer:
    """Callable class to consume entity-based random effect training jobs"""
    def __init__(self, lr_model, name):
        self.name = f'Training: {name}'
        self.lr_model = lr_model
        self.job_count = 0

    def __call__(self, job: Job):
        """
        Call method to process a training job
        :param job:      training job to be processed
        :return: None
        """
        # Train model
        result = self.lr_model.fit(X=job.X,
                                   y=job.y,
                                   weights=job.weights,
                                   offsets=job.offsets,
                                   theta_initial=None if job.theta is None else job.theta)
        inc_count(self)
        return job.entity_id, TrainingResult(result[0], job.unique_global_indices)


class InferenceJobConsumer:
    """Callable class to consume entity-based random effect inference jobs"""
    def __init__(self, lr_model, num_features, schema_params, use_local_index, name):
        self.use_local_index = use_local_index
        self.name = f'Inference: {name}'
        self.num_features = num_features
        self.lr_model = lr_model
        self.schema_params = schema_params
        self.job_count = 0
        logger.info(f"InferenceJobConsumer with use_local_index = {self.use_local_index} created: {name!r}.")

    def _inference_results(self, labels, predicts, sample_weights, sample_ids, predicts_per_coordinate):
        """
        Append validation results into records
        :param labels:                  Ground truth label tensor
        :param predicts:                Prediction tensor
        :param predicts_per_coordinate: Prediction tensor without offset
        :return: Records containing all the inference info
        """
        predicts = predicts.flatten()
        if labels is not None:
            labels = labels.flatten()
        if predicts_per_coordinate is not None:
            predicts_per_coordinate = predicts_per_coordinate.flatten()
        batch_size = predicts.size
        params = self.schema_params
        records = []
        for i in range(batch_size):
            record = {params.prediction_score_column_name: predicts[i], params.weight_column_name: sample_weights[i], params.uid_column_name: sample_ids[i]}
            if labels is not None:
                record[params.label_column_name] = labels[i]
            if predicts_per_coordinate is not None:
                record[params.prediction_score_per_coordinate_column_name] = predicts_per_coordinate[i]
            records.append(record)
        return records

    def __call__(self, job: Job):
        """ Call method to process an inference job """
        if job.theta is None:
            logits = job.offsets
        else:
            if self.use_local_index:
                # Convert locally indexed weights to global space. Since global indices are shifted by one because of
                # the bias term, increase global index values by 1
                locally_indexed_custom_theta = job.theta
                unique_global_indices = job.unique_global_indices + 1
                cols = np.hstack((0, unique_global_indices))
                rows = np.zeros(cols.shape, dtype=int)
                custom_theta = csr_matrix((locally_indexed_custom_theta, (rows, cols)), shape=(1, self.num_features + 1)).T
            else:
                custom_theta = job.theta

            logits = self.lr_model.predict_proba(X=job.X, offsets=job.offsets, custom_theta=custom_theta, return_logits=True)
        logits_per_coordinate = logits - job.offsets
        inc_count(self)
        return self._inference_results(job.y, logits, job.weights.flatten(), job.ids.flatten(), logits_per_coordinate)


def inc_count(job_consumer):
    job_consumer.job_count += 1
    if job_consumer.job_count % _CONSUMER_LOGGING_FREQUENCY == 0:
        logger.info(f"{current_process()}: completed {job_consumer.job_count} jobs so far for {job_consumer.name}.")


def prepare_jobs(batch_iterator, model_params, schema_params, num_features, model_weights: dict, gen_index_map: bool):
    """
    Utility method to take batches of TF grouped data and convert it into one or more Jobs.
    Useful for running training and inference
    :param batch_iterator:        TF dataset feature, label batch iterator
    :param model_params:          model parameters to aid in converting to Job objects
    :param schema_params:         schema parameters to aid in converting to Job objects
    :param num_features           Number of features in global space
    :param model_weights:         Model coefficients
    :param gen_index_map:         Generate local -> global index mapping if True
    :return:
    """
    logger.info(f"Kicking off job producer with gen_index_map = {gen_index_map}.")
    for features_val, labels_val in dataset_reader(batch_iterator()):
        # Extract number of entities in batch
        num_entities = features_val[model_params.partition_entity].shape[0]

        # Now, construct entity_id, X, y, offsets and weights
        X_index = 0
        y_index = 0
        for entity in range(num_entities):
            ids_indices = features_val[schema_params.uid_column_name].indices
            rows = ids_indices[np.where(ids_indices[:, 0] == entity)][:, 1]
            sample_count_from_ids = rows.size
            if model_params.feature_bag is None:
                # intercept only model
                assert(num_features == 1)
                sample_count = sample_count_from_ids
                values = np.zeros(sample_count)
                cols = np.zeros(sample_count, dtype=int)
            else:
                # Construct data matrix X. Slice portion of arrays from X_index through the number of rows for the entity
                features = features_val[model_params.feature_bag + INDICES_SUFFIX]
                indices = features.indices
                rows = indices[np.where(indices[:, 0] == entity)][:, 1]
                cols = features.values[X_index: X_index + len(rows)]
                values = features_val[model_params.feature_bag + VALUES_SUFFIX].values[X_index: X_index + len(rows)]

                # Get sample count
                sample_count = np.amax(rows) + 1

                # sanity check
                assert(sample_count == sample_count_from_ids)

            # Construct entity ID
            entity_id = str(features_val[model_params.partition_entity][entity])
            result = model_weights.get(entity_id, None)
            if gen_index_map:
                # Locally index the column values, and preserve mapping to global space
                unique_global_indices, locally_indexed_cols = np.unique(cols, return_inverse=True)
                X = coo_matrix((values, (rows, locally_indexed_cols)))
            else:
                unique_global_indices = result.unique_global_indices if result else None
                X = coo_matrix((values, (rows, cols)), shape=(sample_count, num_features))

            # Construct y, offsets, weights and ids. Slice portion of arrays from y_index through sample_count
            y = labels_val[schema_params.label_column_name].values[y_index: y_index + sample_count]
            offsets = features_val[model_params.offset].values[y_index: y_index + sample_count]
            weights = (features_val[schema_params.weight_column_name].values[y_index: y_index + sample_count]
                       if schema_params.weight_column_name in features_val else np.ones(sample_count))

            ids = features_val[schema_params.uid_column_name].values[y_index: y_index + sample_count]

            yield Job(entity_id, X, y, offsets, weights, ids, unique_global_indices,
                      theta=result.theta if result else None)

            # Update X_index and y_index
            y_index += sample_count
            X_index += len(rows)
