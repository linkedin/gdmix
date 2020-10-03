import json
import os
import logging
import tensorflow as tf
from gdmix.drivers.driver import Driver
from gdmix.util import constants
from gdmix.util.distribution_utils import remove_tf_config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RandomEffectDriver(Driver):
    """
    Driver class to support random-effect training.
    """
    _RANDOM_EFFECT_PARTITION_DIR_PREFIX = "partitionId="

    def __init__(self, base_training_params, model):
        super().__init__(base_training_params, model, constants.RANDOM_EFFECT)

    def _validate_params(self):
        assert self.base_training_params.model_type == constants.LOGISTIC_REGRESSION, \
            "Random effect supports logistic_regression only"
        assert self.base_training_params.partition_list_file is not None, \
            "Random effect requires partition list file"

    def _setup_cluster(self):
        logger.info("Setting up cluster parameters for random effect training")
        tf_config = os.environ.get(constants.TF_CONFIG)
        if not tf_config:
            # setup local mode
            execution_context = {constants.TASK_TYPE: 'worker',
                                 constants.TASK_INDEX: 0,
                                 constants.CLUSTER_SPEC: None,
                                 constants.NUM_WORKERS: 1,
                                 constants.NUM_SHARDS: 1,
                                 constants.SHARD_INDEX: 0,
                                 constants.IS_CHIEF: True}
            return execution_context
        tf_config_json = json.loads(tf_config)

        cluster = tf_config_json.get('cluster')
        execution_context = {constants.TASK_TYPE: tf_config_json.get('task', {}).get('type'),
                             constants.TASK_INDEX: tf_config_json.get('task', {}).get('index'),
                             # Random effect runs in local mode
                             constants.CLUSTER_SPEC: None,
                             constants.NUM_WORKERS: tf.train.ClusterSpec(cluster).num_tasks(constants.WORKER),
                             constants.NUM_SHARDS: 1,
                             constants.SHARD_INDEX: 0,
                             constants.IS_CHIEF: tf_config_json.get('task', {}).get('index') == 0}
        if execution_context[constants.TASK_TYPE] is None or execution_context[constants.TASK_INDEX] is None:
            raise Exception('No job name found')
        if execution_context[constants.NUM_WORKERS] < 1:
            raise Exception('No worker found')
        # Since random effect runs in local mode, set TF_CONFIG to {}
        remove_tf_config()
        return execution_context

    def _get_partition_list(self):
        with tf.io.gfile.GFile(self.base_training_params.partition_list_file) as f:
            line = f.readline()
        all_partitions = [int(l) for l in line.split(',')]
        num_partitions = len(all_partitions)
        indices = list(range(self.execution_context[constants.TASK_INDEX], num_partitions,
                             self.execution_context[constants.NUM_WORKERS]))
        partition_index_list = [all_partitions[i] for i in indices]
        return partition_index_list

    def _anchor_directory(self, directory_path, partition_index):
        # For random effect, directories should be anchored by attaching partition information
        return os.path.join(directory_path,
                            RandomEffectDriver._RANDOM_EFFECT_PARTITION_DIR_PREFIX + str(partition_index))
