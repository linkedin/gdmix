import logging
import numpy as np
from collections import namedtuple
from gdmix.models.custom.binary_logistic_regression import BinaryLogisticRegressionTrainer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a named tuple to represent training result
TrainingResult = namedtuple("TrainingResult", "training_result unique_global_indices")


class TrainingJobConsumer:
    """
    Callable class to consume entity-based random effect training jobs from a shared queue
    """
    _CONSUMER_LOGGING_FREQUENCY = 1000

    def __init__(self, consumer_id, regularize_bias=False, lambda_l2=1.0, tolerance=1e-8, num_of_curvature_pairs=10,
                 num_iterations=100):
        self.consumer_id = consumer_id
        self.lr_trainer = BinaryLogisticRegressionTrainer(regularize_bias=regularize_bias, lambda_l2=lambda_l2,
                                                          precision=tolerance/np.finfo(float).eps,
                                                          num_lbfgs_corrections=num_of_curvature_pairs,
                                                          max_iter=num_iterations)
        self.processed_counter = 0

    def __call__(self, training_job_queue, training_results_dict, get_timeout_in_seconds=300):
        """
        Call method to read training jobs off of a shared queue
        :param training_job_queue:      Shared multiprocessing job queue
        :param training_results_dict:   Shared dictionary to store training results
        :param get_timeout_in_seconds:   Timeout (in seconds) for retrieving items off the shared job queue
        :return: None
        """
        logger.info("Kicking off training job consumer with ID : {}".format(self.consumer_id))
        while True:
            # Extract TrainingJob object
            training_job = training_job_queue.get(True, get_timeout_in_seconds)
            # If producer is done producing jobs, terminate consumer
            if training_job is None:
                logger.info("Terminating consumer {}".format(self.consumer_id))
                break

            # Train model
            training_result = self.lr_trainer.fit(X=training_job.X,
                                                  y=training_job.y,
                                                  weights=training_job.weights,
                                                  offsets=training_job.offsets)
            # Map trained model to entity ID
            training_results_dict[training_job.entity_id] = TrainingResult(training_result=training_result[0],
                                                                           unique_global_indices=training_job.
                                                                           unique_global_indices)

            self.processed_counter += 1
            if self.processed_counter % TrainingJobConsumer._CONSUMER_LOGGING_FREQUENCY == 0:
                logger.info("Consumer job {} has completed {} training jobs so far".format(self.consumer_id,
                                                                                           self.processed_counter))
