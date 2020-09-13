import logging
import numpy as np
from scipy.sparse import csr_matrix

from gdmix.util.io_utils import dataset_reader
from gdmix.models.custom.scipy.utils import convert_to_training_jobs

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class PhotonMLWriter:
    """
    Helper class to run tf.Session() style inference and write model and/or data in Photon ML format
    """
    INFERENCE_LOG_FREQUENCY = 100

    def __init__(self, lr_model, model_coefficients, num_features, schema_params):
        self.num_features = num_features
        self.lr_model = lr_model
        self.model_coefficients = model_coefficients
        self.schema_params = schema_params

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
        if self.schema_params.sample_weight in metadata:
            sample_weights = metadata[self.schema_params.sample_weight].flatten()
            assert (batch_size == sample_weights.size)
        if self.schema_params.sample_id in metadata:
            sample_ids = metadata[self.schema_params.sample_id].flatten()
            assert (batch_size == sample_ids.size)

        for i in range(batch_size):
            record = {self.schema_params.prediction_score: predicts[i]}
            if labels is not None:
                record[self.schema_params.label] = labels[i]
            if predicts_per_coordinate is not None:
                record[self.schema_params.prediction_score_per_coordinate] = prediction_per_coordinate[i]
            if self.schema_params.sample_weight in metadata:
                record[self.schema_params.sample_weight] = sample_weights[i]
            if self.schema_params.sample_id in metadata:
                record[self.schema_params.sample_id] = sample_ids[i]
            records.append(record)

    def get_batch_iterator(self, iterator):
        """Create an batch generator with TF iterator"""
        def batch_generator():
            for features_val, labels_val in dataset_reader(iterator=iterator):
                # Re-use convert_to_training_jobs code to generate inference jobs
                processed_inference_jobs = convert_to_training_jobs(features_val, labels_val,
                                                                    self.schema_params,
                                                                    num_features=self.num_features,
                                                                    enable_local_indexing=False)
                validation_results = []
                # Run inference on all jobs
                for processed_inference_job in processed_inference_jobs:
                    self._run_batched_inference(processed_inference_job, validation_results, self.schema_params.enable_local_indexing)
                for r in validation_results:
                    yield r
        return batch_generator()

    def _run_batched_inference(self, processed_inference_job, validation_results, local_indexing):
        coeff = self.model_coefficients.get(processed_inference_job.entity_id, None)
        if coeff:
            if local_indexing:
                # Convert locally indexed weights to global space. Since global indices are shifted by one because of
                # the bias term, increase global index values by 1
                locally_indexed_custom_theta = coeff.training_result
                unique_global_indices = coeff.unique_global_indices + 1
                cols = np.hstack((0, unique_global_indices))
                rows = np.zeros(cols.shape, dtype=int)
                custom_theta = csr_matrix((locally_indexed_custom_theta, (rows, cols)), shape=(1, self.num_features+1)).T
            else:
                custom_theta = coeff.training_result

            logits = self.lr_model.predict_proba(X=processed_inference_job.X,
                                                 offsets=processed_inference_job.offsets,
                                                 custom_theta=custom_theta,
                                                 return_logits=True)
        else:
            logits = processed_inference_job.offsets
        logits_per_coordinate = logits - processed_inference_job.offsets
        metadata_val = {self.schema_params.sample_weight: processed_inference_job.weights,
                        self.schema_params.sample_id: processed_inference_job.ids}
        self.append_validation_results(validation_results, processed_inference_job.y, logits, metadata_val, logits_per_coordinate)
