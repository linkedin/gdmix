import logging
from collections import namedtuple
from multiprocessing.managers import BaseProxy
from multiprocessing.process import current_process

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, issparse

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
    def __init__(self, lr_model, name, job_queue, enable_local_indexing):
        self.name = f'Training: {name}'
        self.lr_model = lr_model
        self.job_count = 0
        self.job_queue = job_queue
        self.enable_local_indexing = enable_local_indexing

    def __call__(self, job_id: str):
        """
        Call method to process a training job
        :param job_id: training job_id, a dummy input
        :return: (entity_id, TrainingResult)
        """
        # Train model
        job = self.job_queue.get()
        theta_length = job.X.shape[1] + 1  # +1 due to intercept
        result = self.lr_model.fit(X=job.X,
                                   y=job.y,
                                   weights=job.weights,
                                   offsets=job.offsets,
                                   theta_initial=self._densify_theta(job.theta, theta_length))
        inc_count(self)

        if self.enable_local_indexing:
            theta = result[0]
        else:
            # extract the values from result according to unique_global_indices.
            theta = self._sparsify_theta(result[0], job.unique_global_indices)
        return job.entity_id, TrainingResult(theta, job.unique_global_indices)

    def _densify_theta(self, theta, length):
        """
        Convert theta to dense vector if it is not so already.
        It will be used as initial value for the L-BFGS optimizer.
        If the input theta is None, an all-zero vector will be returned.
        :param theta: input theta, can be None, sparse or dense vector.
        :param length: expect length of the return vector.
        :return: a dense numpy array representing the initial coefficients.
        """
        if theta is None:
            return np.zeros(length)

        if issparse(theta):
            theta = theta.toarray().squeeze()
        elif not isinstance(theta, np.ndarray):
            raise ValueError("Unknown data type, expecting sparse matrix or numpy array")

        assert theta.shape == (length,), f"Dimension mismatch, theta " \
                                         f"shape: {theta.shape}, expected shape ({length},)"
        return theta

    def _sparsify_theta(self, theta, indices_without_intercept):
        """
        Exract the relevant values from theta according to the indices.
        This undoes the densification at the input to L-BFGS optimizer.
        :param theta: input coefficient vector
        :param indices_without_intercept: indices for the relevant (non-padding) elements.
        :return: a numpy array containing the elements specified by the input indices.
        """
        indices = [0] + [x+1 for x in indices_without_intercept]  # account for the intercept.
        return np.array([theta[u] for u in indices])


class InferenceJobConsumer:
    """Callable class to consume entity-based random effect inference jobs"""
    def __init__(self, lr_model, num_features, schema_params, name, job_queue):
        self.name = f'Inference: {name}'
        self.num_features = num_features
        self.lr_model = lr_model
        self.schema_params = schema_params
        self.job_count = 0
        self.job_queue = job_queue

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
            record = {params.prediction_score_column_name: predicts[i], params.weight_column_name: sample_weights[i],
                      params.uid_column_name: sample_ids[i]}
            if labels is not None:
                record[params.label_column_name] = labels[i]
            if predicts_per_coordinate is not None:
                record[params.prediction_score_per_coordinate_column_name] = predicts_per_coordinate[i]
            records.append(record)
        return records

    def __call__(self, job_id: str):
        """
        Call method to process an inference jo
        :param job_id: inference job_id, a dummy input
        :return: records corresponding to an input batch
        """
        job = self.job_queue.get()
        if job.theta is None:
            logits = job.offsets
        else:
            logits = self.lr_model.predict_proba(X=job.X, offsets=job.offsets, custom_theta=job.theta,
                                                 return_logits=True)
        logits_per_coordinate = logits - job.offsets
        inc_count(self)
        return self._inference_results(job.y, logits, job.weights.flatten(), job.ids.flatten(), logits_per_coordinate)


def inc_count(job_consumer):
    job_consumer.job_count += 1
    if job_consumer.job_count % _CONSUMER_LOGGING_FREQUENCY == 0:
        logger.info(f"{current_process()}: completed {job_consumer.job_count} jobs so far for {job_consumer.name}.")


def prepare_jobs(batch_iterator, model_params, schema_params, num_features, model_weights: dict,
                 enable_local_indexing: bool, job_queue: BaseProxy):
    """
    Utility method to take batches of TF grouped data and convert it into one or more Jobs.
    Useful for running training and inference
    :param batch_iterator:        TF dataset feature, label batch iterator
    :param model_params:          model parameters to aid in converting to Job objects
    :param schema_params:         schema parameters to aid in converting to Job objects
    :param num_features           Number of features in global space
    :param model_weights:         Model coefficients
    :param enable_local_indexing: Whether to index the features locally instead of use global indices
    :param job_queue:             A managed queue containing the generated jobs
    :return: a generator of entity_ids.

    The feature_bag is represented in sparse tensor format. Take per_member feature bag for example.
    The following batch has three records, two belonging to member #0 and one belonging to member #1.
        member #0 has two records
            per_member_indices = [[0, 7, 60, 80, 95], [34, 57]]
            per_member_values = [[1.0, 2.0, 3.0, 5.0, 6.6], [1.0, 2.0]]
        member #1 has one record
            per_member_indices = [[10, 11]]
            per_member_values = [[-3.5, 2.3]]
    The batch combines both members' records:
        per_member_indices = [[[0, 7, 60, 80, 95], [34, 57]], [[10, 11]]]
        per_member_values = [[[1.0, 2.0, 3.0, 5.0, 6.6], [1.0, 2.0]], [[-3.5, 2.3]]]
    Tensorflow representation of the batch above:
        SparseTensorValue(indices=array(
        [[0, 0, 0],
        [0, 0, 1],
        [0, 0, 2],
        [0, 0, 3],
        [0, 0, 4],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1]]), values=array([ 1. ,  2. ,  3. ,  5. ,  6.6,  1. ,  2. , -3.5,  2.3],
        dtype=float32), dense_shape=array([2, 2, 5]))
        Note the first dimension is the batch dimension.
    """
    logger.info(f"Kicking off job producer with enable_local_indexing = {enable_local_indexing}.")
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
            raw_entity_id = features_val[model_params.partition_entity][entity]
            if isinstance(raw_entity_id, bytes):
                entity_id = raw_entity_id.decode('utf-8')
            else:
                entity_id = str(raw_entity_id)
            result = model_weights.get(entity_id, None)

            # generate index map
            unique_global_indices, locally_indexed_cols = np.unique(cols, return_inverse=True)

            if enable_local_indexing:
                # Use local indices to represent the data matrix.
                X = coo_matrix((values, (rows, locally_indexed_cols)))
            else:
                # Use global indices to represent the data matrix.
                X = coo_matrix((values, (rows, cols)), shape=(sample_count, num_features))

            # Construct y, offsets, weights and ids. Slice portion of arrays from y_index through sample_count
            y = labels_val[schema_params.label_column_name].values[y_index: y_index + sample_count]
            offsets = features_val[model_params.offset].values[y_index: y_index + sample_count]
            weights = (features_val[schema_params.weight_column_name].values[y_index: y_index + sample_count]
                       if schema_params.weight_column_name in features_val else np.ones(sample_count))

            ids = features_val[schema_params.uid_column_name].values[y_index: y_index + sample_count]

            # If a prior model exists, get the coefficients to warm start the training.
            # Note the prior model may have fewer or more features than the current dataset.
            theta = None
            if result:
                prior_model = {u: v for u, v in zip(result.unique_global_indices, result.theta[1:])}
                model_rows = [0]  # intercept index
                model_values = [result.theta[0]]  # intercept value
                for i, u in enumerate(unique_global_indices):
                    if u in prior_model:
                        r = i if enable_local_indexing else u
                        model_rows.append(1 + r)  # +1 since intercept is the first element.
                        model_values.append(prior_model[u])
                model_cols = [0]*len(model_rows)
                if enable_local_indexing:
                    theta = csr_matrix((model_values, (model_rows, model_cols)),
                                       shape=(len(unique_global_indices) + 1, 1))
                else:
                    theta = csr_matrix((model_values, (model_rows, model_cols)), shape=(num_features + 1, 1))
            job = Job(entity_id, X, y, offsets, weights, ids, unique_global_indices, theta=theta)
            job_queue.put(job)
            # use entity_id as a token, it may not be unique
            yield entity_id

            # Update X_index and y_index
            y_index += sample_count
            X_index += len(rows)
