import json
import os
import logging
import tensorflow as tf
from gdmix.drivers.driver import Driver
from gdmix.util import constants
from gdmix.util.distribution_utils import remove_tf_config

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)


class FixedEffectDriver(Driver):
    """
    Driver class to support fixed-effect training.
    """

    def __init__(self, base_training_params, model):
        super().__init__(base_training_params, model, constants.FIXED_EFFECT)

    def _validate_params(self):
        pass

    def _setup_cluster(self):
        logger.info("Setting up cluster parameters for fixed effect training")
        tf_config = os.environ.get(constants.TF_CONFIG)
        if not tf_config:
            # setup local mode
            execution_context = {constants.TASK_TYPE: 'worker',
                                 constants.TASK_INDEX: 0,
                                 constants.CLUSTER_SPEC: {"worker": ["localhost:2222"]},
                                 constants.NUM_WORKERS: 1,
                                 constants.NUM_SHARDS: 1,
                                 constants.SHARD_INDEX: 0,
                                 constants.IS_CHIEF: True}
            return execution_context
        tf_config_json = json.loads(tf_config)
        cluster = tf_config_json.get('cluster')
        if self.base_training_params.action == constants.ACTION_INFERENCE:
            # Inference / prediction / validation runs in local mode.
            cluster_spec = None
        else:
            cluster_spec = tf.train.ClusterSpec(cluster)
        execution_context = {constants.TASK_TYPE: tf_config_json.get('task', {}).get('type'),
                             constants.TASK_INDEX: tf_config_json.get('task', {}).get('index'),
                             constants.CLUSTER_SPEC: cluster_spec,
                             constants.NUM_WORKERS: tf.train.ClusterSpec(cluster).num_tasks(constants.WORKER),
                             constants.NUM_SHARDS: tf.train.ClusterSpec(cluster).num_tasks(constants.WORKER),
                             constants.SHARD_INDEX: tf_config_json.get('task', {}).get('index'),
                             constants.IS_CHIEF: tf_config_json.get('task', {}).get('index') == 0}
        if execution_context[constants.TASK_TYPE] is None or execution_context[constants.TASK_INDEX] is None:
            raise Exception('No job name found')
        if execution_context[constants.NUM_WORKERS] < 1:
            raise Exception('No worker found')
        if cluster_spec is None:
            # Remove TF_CONFIG if cluster_spec is none.
            remove_tf_config()
        return execution_context

    def _get_partition_list(self):
        # For fixed effect training, partition index is the same as task index
        return [self.execution_context[constants.TASK_INDEX]]

    def _anchor_directory(self, directory_path, partition_index):
        # For fixed effect, anchoring using partition_index is not required
        assert partition_index >= 0
        return directory_path
