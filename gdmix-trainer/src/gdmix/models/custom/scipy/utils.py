import numpy as np
from collections import namedtuple
from scipy.sparse import coo_matrix

# Create a named tuple to represent training jobs
TrainingJob = namedtuple("TrainingJob", "entity_id X y offsets weights ids unique_global_indices")

INDICES_SUFFIX = "_indices"
VALUES_SUFFIX = "_values"


def convert_to_training_jobs(features_val, labels_val, conversion_params, num_features, enable_local_indexing=False):
    """
    Utility method to take a batch of TF grouped data and convert it into one or more TrainingJobs.
    Useful for running training and inference
    :param features_val:          TF dataset feature values
    :param labels_val:            TF dataset label values
    :param conversion_params:     Conversion parameters to aid in converting to TrainingJob objects
    :param num_features           Number of features in global space
    :param enable_local_indexing: Enable local indexing
    :return:
    """
    # Extract number of entities in batch
    num_entities = features_val[conversion_params.partition_entity].shape[0]

    # Now, construct entity_id, X, y, offsets, uids and weights
    X_index = 0
    y_index = 0
    for entity in range(num_entities):
        # Construct entity ID
        entity_id = features_val[conversion_params.partition_entity][entity]

        # Construct data matrix X. Slice portion of arrays from X_index through the number of rows for the entity
        indices = features_val[conversion_params.feature_bags[0] + INDICES_SUFFIX].indices
        rows = indices[np.where(indices[:, 0] == entity)][:, 1]
        values = features_val[conversion_params.feature_bags[0] + VALUES_SUFFIX].values[X_index: X_index + len(rows)]
        cols = features_val[conversion_params.feature_bags[0] + INDICES_SUFFIX].values[X_index: X_index + len(rows)]

        # Get sample count
        sample_count = np.amax(rows) + 1

        if enable_local_indexing:
            # Locally index the column values, and preserve mapping to global space
            unique_global_indices, locally_indexed_cols = np.unique(cols, return_inverse=True)
            X = coo_matrix((values, (rows, locally_indexed_cols)))
        else:
            unique_global_indices = None
            X = coo_matrix((values, (rows, cols)), shape=(sample_count, num_features))

        # Construct y, offsets, weights and ids. Slice portion of arrays from y_index through sample_count
        y = labels_val[conversion_params.label].values[y_index: y_index + sample_count]
        offsets = features_val[conversion_params.offset].values[y_index: y_index + sample_count]
        weights = (features_val[conversion_params.sample_weight].values[y_index: y_index + sample_count] if conversion_params.sample_weight in features_val else
                   np.ones(sample_count))

        ids = features_val[conversion_params.sample_id].values[y_index: y_index + sample_count]

        yield TrainingJob(entity_id=entity_id, X=X, y=y, offsets=offsets, weights=weights, ids=ids, unique_global_indices=unique_global_indices)

        # Update X_index and y_index
        y_index += sample_count
        X_index += len(rows)
