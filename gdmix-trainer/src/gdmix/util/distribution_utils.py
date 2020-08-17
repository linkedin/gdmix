import os
import logging
import tensorflow as tf

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def shard_input_files(input_path, num_shards, shard_index):

    """List input files in the input_path, then shard them such that
    a worker only takes a portion of the input files. If the number
    of files are less than the number of shards, file-level sharding indicator
    is set, each shard gets one file if the shard_index is less than
    the number of files, otherwise empty list is returned.

    Input path is a directory or a directory + file pattern.
    Possible values for input_path:
    1) a directory: hdfs://namespace.com/jobs/bert/trainData
    2) a filename pattern: /user/data/*.tfrecord
    :param input_path: the path where the training dataset is located. it can be
           a directory or a filename pattern.
    :param num_shards: Total number of shards.
    :param shard_index: The index of the current worker.
    :return: A tuple, (a list of files belonging to the shard and a boolean
             suggesting whether sample level sharding is needed.
    """
    assert((shard_index >= 0) and (num_shards >= 1) and (num_shards > shard_index))
    if tf.compat.v1.gfile.IsDirectory(input_path):
        input_files = tf.io.gfile.glob(os.path.join(input_path, '*'))
    else:  # This is a file or file pattern
        input_files = tf.io.gfile.glob(input_path)
    # sort the file so that all workers see the same order.
    input_files = sorted(input_files)
    n = len(input_files)
    # there should be at least one file
    assert(n > 0), "{} is empty".format(input_files)
    if n < num_shards:
        if shard_index < n:
            return [input_files[shard_index]], True
        else:
            return [], True
    else:
        return [input_files[i]
                for i in range(shard_index, n, num_shards)], False


def remove_tf_config():
    tf_config = os.environ.pop('TF_CONFIG', '')

    if tf_config:
        logger.info("====== removing the following tf config environmental variable =======")
        logger.info(tf_config)
        logger.info("======================================================================")
