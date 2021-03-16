import os
import logging
import tensorflow as tf

from functools import partial
from gdmix.io.dataset_metadata import DatasetMetadata
from gdmix.util import constants
from gdmix.util.distribution_utils import shard_input_files
from gdmix.util.io_utils import low_rpc_call_glob

GZIP = "GZIP"
ZLIB = "ZLIB"
GZIP_SUFFIX = ".gz"
ZLIB_SUFFIX = ".deflate"
UNCOMPRESSED = ""

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _unpack_one_element_list(scalar_or_list):
    """If the value is a list and it only has one value, we unpack it;
    otherwise, we keep the list. This is used for size parameter inside
    tf.io.SparseFeature.
    """
    if isinstance(scalar_or_list, list) and len(scalar_or_list) == 1:
        return scalar_or_list[0]
    else:
        return scalar_or_list


def _get_features_and_labels_info(metadata_file):
    """
    Get the metadata information for features and labels
    :param metadata_file: input tensor metadata file
    :return: feature metadata namedtuple and label metadata namedtuple
    """
    metadata = DatasetMetadata(metadata_file)
    return metadata.get_features(), metadata.get_labels()


def _splits_label_and_features(example, label_tensors):
    """
    Split the features and labels in a tfrecord example or sequence example.
    :param example: tfrecord example or sequence of example
    :param label_tensors: metadata of label tensors
    :return: feature partition and label partition of the example / sequence example.
    """
    label_names = [x.name for x in label_tensors]
    return example, {label_key: example.pop(label_key) for label_key in label_names}


def _if_files_exist(file_pattern):
    """
    Check if the files with the given perfix and suffix file pattern exist
    :param file_pattern: the file pattern to be searched for
    :return: True if files exist, False otherwise.
    """
    num_files = len(low_rpc_call_glob(file_pattern))
    return num_files > 0


def _identify_tfrecord_file_compression_type(filename):
    """
    Find the compression type of the file from its name.
    :param filename: the file name to be identified.
    :return: a string representing compression type: "", "ZLIB" or "GZIP".
             See https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset for details.
    """
    suffix = filename.split('.')[-1]
    if suffix == ZLIB_SUFFIX[1:]:
        compression_type = ZLIB
    elif suffix == GZIP_SUFFIX[1:]:
        compression_type = GZIP
    else:
        # When the suffix is neither ".deflate" nor "gz", we assume it's uncompressed.
        # Since there is no standard naming for uncompressed files, it could be
        # ".tfrecord", ".tfrecords" or any other suffix, we leave the TFRecordDataset to
        # make a decision.
        compression_type = UNCOMPRESSED
    if compression_type == UNCOMPRESSED:
        logger.info(f"Identified uncompressed type")
    else:
        logger.info(f"Identified {compression_type} compression type")
    return compression_type


def _convert_dir_to_filename_pattern(file_path, filename_pattern="*"):
    """
    Convert a directory to a filename under that directory with a given pattern.
    For example if the directory is /src/data and the filename_pattern is "*.avro",
    then it returns "/src/data/*.avro". If the input file_path is not a directory, it
    returns the input file_path
    :param file_path: input file path
    :param filename_pattern: filename patterns with "*" or "?"
    :return: (file_path + filename_pattern if file_path is a directory, compression_type)
    """
    compression_type = UNCOMPRESSED
    if tf.io.gfile.isdir(file_path):
        # Handle TFRecord file patterns, check compressed files
        if filename_pattern == constants.TFRECORD_GLOB_PATTERN:
            uncompressed_filename_pattern = os.path.join(file_path, filename_pattern)
            zlib_filename_pattern = uncompressed_filename_pattern + ZLIB_SUFFIX
            gzip_filename_pattern = uncompressed_filename_pattern + GZIP_SUFFIX
            if _if_files_exist(uncompressed_filename_pattern):
                input_filename_pattern = uncompressed_filename_pattern
            elif _if_files_exist(zlib_filename_pattern):
                input_filename_pattern = zlib_filename_pattern
                compression_type = ZLIB
            elif _if_files_exist(gzip_filename_pattern):
                input_filename_pattern = gzip_filename_pattern
                compression_type = GZIP
            else:
                # append filename pattern to the input directory
                input_filename_pattern = os.path.join(file_path, filename_pattern)
    else:
        # Return the input file_path
        input_filename_pattern = file_path
        # Use filename_pattern to check compression type
        if filename_pattern == constants.TFRECORD_GLOB_PATTERN:
            compression_type = _identify_tfrecord_file_compression_type(input_filename_pattern)
    if compression_type == UNCOMPRESSED:
        logger.info(f"Found uncompressed type")
    else:
        logger.info(f"Found {compression_type} compression type")
    return input_filename_pattern, compression_type


def per_record_input_fn(input_path_or_files, metadata_file, num_shards, shard_index,
                        batch_size, data_format, build_features=True, custom_input_fn=None):
    """
    Input function for per-record dataset. In the dataset, the records are individual examples.
    Batch size adds extra dimension on top of that.
    :param input_path_or_files: input directory, file pattern or a list of files
    :param metadata_file: tensor metadata file
    :param num_shards: number of shards
    :param shard_index: the index of this worker
    :param batch_size: batch size
    :param data_format: tfrecord or avro
    :param build_features: whether to parse the record into feature and labels.
    :param custom_input_fn: full name "package.module.fn" for the external custom input_fn.
    :return: a batched dataset.
    """
    if data_format == constants.TFRECORD:
        logger.info("using {} dataset".format(constants.TFRECORD))

        def build_features(tensors):
            """
            Create features from metadata, used to deserialize the tfrecord.
            :param tensors: list of metadata for all tensors.
            :return: tfrecord features
            """
            tf_features = {}
            for feature in tensors:
                if feature.isSparse:
                    # If this is a sparse tensor, we process indices and values separately.
                    # Note in the metadata, we don't see _indices and _values,
                    # only the feature name.
                    tf_features[feature.name] = tf.io.SparseFeature(
                        index_key=f"{feature.name}_{DatasetMetadata.INDICES}",
                        value_key=f"{feature.name}_{DatasetMetadata.VALUES}",
                        dtype=DatasetMetadata.map_int(feature.dtype),
                        size=_unpack_one_element_list(feature.shape)
                    )
                else:
                    tf_features[feature.name] = tf.io.FixedLenFeature(
                        shape=feature.shape, dtype=DatasetMetadata.map_int(feature.dtype))
            return tf_features

        def map_fn(serialized, feature_tensors, label_tensors):
            """
            Deserialize TF records to features. This is done after batching since we are using
            tf.io.parse_example
            :param serialized: Serialized TF records
            :param feature_tensors: list of feature tensors
            :param label_tensors: list of label tensors
            :return: (features, labels) tuple where each of them is a map.
            """
            tensors = feature_tensors + label_tensors
            tf_features = build_features(tensors)
            example = tf.io.parse_example(
                serialized,
                tf_features,
                example_names=None,
                name=None
            )
            # then split features from labels
            return _splits_label_and_features(example, label_tensors)

        # Get shard input files
        if isinstance(input_path_or_files, str):
            input_filename_pattern, compression_type = _convert_dir_to_filename_pattern(
                input_path_or_files, constants.TFRECORD_GLOB_PATTERN)
            input_files, _ = shard_input_files(input_filename_pattern, num_shards, shard_index)
        elif isinstance(input_path_or_files, list):
            input_files = input_path_or_files
            compression_type = _identify_tfrecord_file_compression_type(input_files[0])

        # Get metadata
        feature_tensors, label_tensors = _get_features_and_labels_info(metadata_file)

        # Batching
        dataset = tf.data.TFRecordDataset(input_files, compression_type=compression_type) \
            .batch(batch_size, drop_remainder=False)

        if build_features:
            # Deserialize to features
            dataset = dataset.map(partial(map_fn, feature_tensors=feature_tensors,
                                          label_tensors=label_tensors),
                                  num_parallel_calls=16)
    elif custom_input_fn is not None:
        logger.info("loading {} dataset by {}".format(data_format, custom_input_fn))
        import importlib
        module_name, fn_name = custom_input_fn.rsplit('.', 1)
        dataset_module = importlib.import_module(module_name)
        dataset = getattr(dataset_module, fn_name)(input_path_or_files, metadata_file, num_shards,
                                                   shard_index, batch_size, data_format)
    else:
        raise Exception("Unknown data format :{}".format(data_format))
    return dataset


def per_entity_grouped_input_fn(input_path, metadata_file, num_shards, shard_index,
                                batch_size, data_format, entity_name, custom_input_fn=None):
    """
    Input function for per-entity grouped dataset. In the dataset, the records are grouped based on
    entity Id. each feature is a vector except the entity Id which is a scalar. Batch size adds extra
    dimension on top of that.
    :param input_path: input directory or file pattern.
    :param metadata_file: tensor metadata file
    :param num_shards: number of shards
    :param shard_index: the index of this worker
    :param batch_size: batch size
    :param data_format: tfrecord or avro
    :param entity_name: the name of the entity which is used to group the records.
    :param custom_input_fn: full name "package.module.fn" for the external custom input_fn.
    :return: a batched dataset.
    """

    if data_format == constants.TFRECORD:
        logger.info(f"using {constants.TFRECORD} dataset")

        # Build features
        def build_features(tensors, entity_name):
            """
            Create features from metadata, used to deserialize the tfrecord.
            :param tensors: list of metadata for all tensors.
            :param entity_name: entity by which the records are grouped.
            :return: a tuple of context_features and sequence_features
            """
            sequence_features = dict()
            context_features = dict()
            for tensor in tensors:
                tensor_dtype = DatasetMetadata.map_int(tensor.dtype)
                if tensor.name == entity_name:
                    # entity_name column is a scalar
                    context_features[entity_name] = tf.io.FixedLenFeature(shape=[], dtype=tensor_dtype)
                else:
                    if tensor.isSparse:
                        # If this is a sparse tensor, we process indices and values separately.
                        # Note in the metadata, we don't see _indices and _values,
                        # only the feature name.
                        indices_name = f"{tensor.name}_{DatasetMetadata.INDICES}"
                        values_name = f"{tensor.name}_{DatasetMetadata.VALUES}"
                        sequence_features[indices_name] = tf.io.VarLenFeature(dtype=tf.int64)
                        sequence_features[values_name] = tf.io.VarLenFeature(dtype=tensor_dtype)
                    else:
                        context_features[tensor.name] = tf.io.VarLenFeature(dtype=tensor_dtype)
            if len(sequence_features) == 0:
                sequence_features = None
            if len(context_features) == 0:
                context_features = None
            return context_features, sequence_features

        def map_fn(serialized, feature_tensors, label_tensors, entity_name):
            """
            Map serialized tfrecord to a dict of tensors.
            :param serialized: serialized tfrecord
            :param feature_tensors: list of metadata for features
            :param label_tensors: list of metadata for labels
            :param entity_name: entity by which the records are grouped.
            :return: A tuple (features, labels)
            """
            tensors = feature_tensors + label_tensors
            context_features, sequence_features = build_features(tensors, entity_name)
            example = tf.io.parse_sequence_example(
                serialized,
                context_features=context_features,
                sequence_features=sequence_features,
                example_names=None,
                name=None
            )

            # Split features from labels
            context, sequence = example[0], example[1]
            sequence.update(context)
            return _splits_label_and_features(sequence, label_tensors)

        # Get shard input files
        input_filename_pattern, compression_type = _convert_dir_to_filename_pattern(
            input_path, constants.TFRECORD_GLOB_PATTERN)
        input_files, _ = shard_input_files(input_filename_pattern, num_shards, shard_index)

        # Get metadata
        feature_tensors, label_tensors = _get_features_and_labels_info(metadata_file)

        # Check if entity_name is one of the features
        feature_names = [x.name for x in feature_tensors]
        if entity_name not in feature_names:
            raise ValueError(f"entity name {entity_name} is not found among the features")

        # Batching
        dataset = tf.data.TFRecordDataset(input_files, compression_type=compression_type) \
            .batch(batch_size, drop_remainder=False)

        # Deserialize to features
        dataset = dataset.map(partial(map_fn, feature_tensors=feature_tensors,
                                      label_tensors=label_tensors,
                                      entity_name=entity_name),
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)

    elif custom_input_fn:
        logger.info(f"loading {data_format} dataset by {custom_input_fn}")
        import importlib
        module_name, fn_name = custom_input_fn.rsplit('.', 1)
        dataset_module = importlib.import_module(module_name)
        dataset = getattr(dataset_module, fn_name)(input_path, metadata_file, num_shards,
                                                   shard_index, batch_size, entity_name,
                                                   data_format)
    else:
        raise Exception(f"Unknown data format : {data_format}")
    return dataset
