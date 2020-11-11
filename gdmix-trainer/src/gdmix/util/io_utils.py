import collections
import csv
import itertools
import json
import logging
import os
import time
from typing import Iterator

import fastavro
import numpy as np
import tensorflow as tf

from gdmix.models.schemas import BAYESIAN_LINEAR_MODEL_SCHEMA

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

INTERCEPT = "(INTERCEPT)"


def try_write_avro_blocks(f, schema, records, suc_msg=None, err_msg=None):
    """
    write a block into avro file. This is used continuously when the whole file does not fit in memory.

    :param f: file handle.
    :param schema: avro schema used by the writer.
    :param records: a set of records to be written to the avro file.
    :param suc_msg: message to print when write succeeds.
    :param err_msg: message to print when write fails.
    :return: none
    """
    try:
        fastavro.writer(f, schema, records)
        if suc_msg:
            logger.info(suc_msg)
    except Exception as exp:
        if err_msg:
            logger.error(exp)
            logger.error(err_msg)
        raise


def load_linear_models_from_avro(model_file, feature_file):
    """
    Load linear models from avro files.
    The models are in photon-ml format.
    Intercept is moved to the end of the coefficient array.
    :param model_file: Model avro file, photon-ml format
    :param feature_file: A file containing all features of the model (intercept excluded)
    :return:
    """

    def get_one_model_weights(model_record, feature_map):
        """
        Load a single model from avro record
        :param model_record: photon-ml LR model in avro record format
        :param feature_map: feature name to index map
        :return: a numpy array of the model coefficients, intercept is at the end. Elements are in np.float64.
        """
        num_features = 0 if feature_map is None else len(feature_map)
        model_coefficients = np.zeros(num_features+1, dtype=np.float64)
        for ntv in model_record["means"]:
            name, term, value = ntv['name'], ntv['term'], np.float64(ntv['value'])
            if name == INTERCEPT and term == '':
                model_coefficients[num_features] = value  # Intercept at the end.
            elif feature_map is not None:
                feature_index = feature_map.get((name, term), None)
                if feature_index is not None:  # Take only the features that in the current training dataset.
                    model_coefficients[feature_index] = value
        return model_coefficients

    if feature_file is None:
        feature_map = None
    else:
        feature_map = get_feature_map(feature_file)
    with tf.io.gfile.GFile(model_file, 'rb') as fo:
        avro_reader = fastavro.reader(fo)
        return tuple(get_one_model_weights(record, feature_map) for record in avro_reader)


def add_dummy_weight(models):
    """
    This function adds a dummy weight 0.0 to the first element of the weight vector.
    It should be only used for the intercept-only model where no feature is present.
    :param models: the models with intercept only.
    :return: models with zero prepend to the intercept.
    """
    def process_one_model(model):
        model_coefficients = np.zeros(2, dtype=np.float64)
        model_coefficients[1] = model[0]
        return model_coefficients
    return tuple(process_one_model(m) for m in models)


def gen_one_avro_model(model_id, model_class, weight_indices, weight_values, bias, feature_list):
    """
    generate the record for one LR model in photon-ml avro format
    :param model_id: model id
    :param model_class: model class
    :param weight_indices: LR weight vector indices
    :param weight_values: LR weight vector values
    :param bias: the bias/offset/intercept
    :param feature_list: corresponding feature names
    :return: a model in avro format
    """
    record = {u'name': INTERCEPT, u'term': '', u'value': bias}
    records = {u'modelId': model_id, u'modelClass': model_class, u'means': [record], u'lossFunction': ""}
    if weight_indices is not None and weight_values is not None:
        for w_i, w_v in zip(weight_indices.flatten(), weight_values.flatten()):
            feat = feature_list[w_i]
            name, term = feat[0], feat[1]
            record = {u'name': name, u'term': term, u'value': w_v}
            records[u'means'].append(record)
    return records


def export_linear_model_to_avro(model_ids,
                                list_of_weight_indices,
                                list_of_weight_values,
                                biases,
                                feature_file,
                                output_file,
                                model_log_interval=1000,
                                model_class="com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel"):
    """
    Export random effect logistic regression model in avro format for photon-ml to consume
    :param model_ids:               a list of model ids used in generated avro file
    :param list_of_weight_indices:  list of indices for entity-specific model weights
    :param list_of_weight_values:   list of values for entity-specific model weights
    :param biases:                  list of entity bias terms
    :param feature_file:            a file containing all the features, typically generated by avro2tf.
    :param output_file:             full file path for the generated avro file.
    :param model_log_interval:      write model every model_log_interval models.
    :param model_class:             the model class defined by photon-ml.
    :return: None
    """
    # STEP [1] - Read feature list
    feature_list = read_feature_list(feature_file) if feature_file else None

    # STEP [2] - Read number of features and models
    num_models = len(biases)
    logger.info(f"To save {num_models} models.")
    if feature_file:
        logger.info(f"Found {len(feature_list)} features in {feature_file}")

    # STEP [3]
    schema = fastavro.parse_schema(json.loads(BAYESIAN_LINEAR_MODEL_SCHEMA))

    def gen_records():
        if list_of_weight_indices is None or list_of_weight_values is None or feature_list is None:
            for i in range(num_models):
                yield gen_one_avro_model(str(model_ids[i]), model_class, None, None, biases[i], feature_list)
        else:
            for i in range(num_models):
                yield gen_one_avro_model(str(model_ids[i]), model_class, list_of_weight_indices[i],
                                         list_of_weight_values[i], biases[i], feature_list)
    batched_write_avro(gen_records(), output_file, schema, model_log_interval)
    logger.info(f"dumped {num_models} models to avro file at {output_file}.")


def read_feature_list(feature_file):
    """
    Get feature names from the feature file.
    Note: intercept is not included here since it is not part of the raw data.
    :param feature_file: user provided feature file, each row is a "name,term" feature name
    :return: list of feature (name, term) tuple
    """
    result = []
    with tf.io.gfile.GFile(feature_file) as f:
        f.seekable = lambda: False
        for row in csv.reader(f):
            assert len(row) == 2, f"Each feature name should have exactly name and term only, but I got {row}."
            result.append(tuple(row))
    return result


def get_feature_map(feature_file):
    """
    Get feature (name, term) -> index map.
    The index of a feature is the position of the feature in the file.
    The index starts from zero.
    :param feature_file: The file containing a list of features.
    :return: a dict of feature (name, term) and its index.
    """
    return {feature: index for index, feature in enumerate(read_feature_list(feature_file))}


def read_json_file(file_path: str):
    """ Load a json file from a path.

    :param file_path: Path string to json file.
    :return: dict. The decoded json object.

    Raises IOError if path does not exist.
    Raises ValueError if load fails.
    """

    if not tf.io.gfile.exists(file_path):
        raise IOError(f"Path {file_path!r} does not exist.")
    try:
        with tf.io.gfile.GFile(file_path) as json_file:
            return json.load(json_file)
    except Exception as e:
        raise ValueError(f"Failed loading file {file_path!r}.") from e


def copy_files(input_files, output_dir):
    """
    Copy a list of files to the output directory.
    The destination files will be overwritten.
    :param input_files: a list of files
    :param output_dir: output directory
    :return: the list of copied files
    """

    logger.info("Copy files to local")
    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.mkdir(output_dir)
    start_time = time.time()
    copied_files = []
    for f in input_files:
        fname = os.path.join(output_dir, os.path.basename(f))
        tf.io.gfile.copy(f, fname, overwrite=True)
        copied_files.append(fname)
    logger.info(f"Files copied to Local: {copied_files}")
    logger.info(f"--- {time.time() - start_time} seconds ---")
    return copied_files


def namedtuple_with_defaults(typename, field_names, defaults=()):
    """
    Namedtuple with default values is supported since 3.7, wrap it to be compatible with version <= 3.6
    :param typename: the type name of the namedtuple
    :param field_names: the field names of the namedtuple
    :param defaults: the default values of the namedtuple
    :return: namedtuple with defaults
    """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    prototype = T(**defaults) if isinstance(defaults, collections.Mapping) else T(*defaults)
    T.__new__.__defaults__ = tuple(prototype)
    return T


def batched_write_avro(records: Iterator, output_file, schema, write_frequency=1000, batch_size=1024):
    """ For the first block, the file needs to be open in âwâ mode, while the
        rest of the blocks needs the âaâ mode. This restriction makes it
        necessary to open the files at least twice, one for the first block,
        one for the remaining. So itâs not possible to put them into the
        while loop within a file context.  """
    f = None
    t0 = time.time()
    n_batch = 0
    logger.info(f"Writing to {output_file} with batch size of {batch_size}.")
    try:
        for batch in _chunked_iterator(records, batch_size):
            if n_batch == 0:
                with tf.io.gfile.GFile(output_file, 'wb') as f0:  # Create the file in 'w' mode
                    f0.seekable = lambda: False
                    try_write_avro_blocks(f0, schema, batch, None, create_error_message(n_batch, output_file))
                f = tf.io.gfile.GFile(output_file, 'ab+')  # reopen the file in 'a' mode for later writes
                f.seekable = f.readable = lambda: True
                f.seek(0, 2)  # seek to the end of the file, 0 is offset, 2 means the end of file
            else:
                try_write_avro_blocks(f, schema, batch, None, create_error_message(n_batch, output_file))
            n_batch += 1
            if n_batch % write_frequency == 0:
                delta_time = time.time() - t0
                logger.info(f"nbatch = {n_batch}, deltaT = {delta_time:0.2f} seconds, speed = {n_batch / delta_time :0.2f} batches/sec")
        logger.info(f"Finished writing to {output_file}.")
    finally:
        f and f.close()


def _chunked_iterator(iterator: Iterator, chuck_size):
    while True:
        chunk_it = itertools.islice(iterator, chuck_size)
        try:
            first_el = next(chunk_it)
            yield itertools.chain((first_el,), chunk_it)
        except StopIteration:
            return


def create_error_message(n_batch, output_file) -> str:
    return f'An error occurred while writing batch #{n_batch} to path {output_file}'


def dataset_reader(iterator):
    # Iterate through TF dataset in a throttled manner
    # (Forking after the TensorFlow runtime creates internal threads is unsafe, use config provided in this
    # link -
    # https://github.com/tensorflow/tensorflow/issues/14442)
    with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(use_per_session_threads=True)) as sess:
        sess.run(iterator.initializer)
        while True:
            try:
                # Extract and process raw entity data
                yield sess.run(iterator.get_next())
            except tf.errors.OutOfRangeError:
                break


def get_inference_output_avro_schema(metadata, has_logits_per_coordinate, schema_params, has_weight=False):
    fields = [{'name': schema_params.uid_column_name, 'type': 'long'}, {'name': schema_params.prediction_score_column_name, 'type': 'float'},
              {'name': schema_params.label_column_name, 'type': ['null', 'int'], "default": None}]
    if has_weight or metadata.get(schema_params.weight_column_name) is not None:
        fields.append({'name': schema_params.weight_column_name, 'type': 'float'})
    if has_logits_per_coordinate:
        fields.append({'name': schema_params.prediction_score_per_coordinate_column_name, 'type': 'float'})
    return {'name': 'validation_result', 'type': 'record', 'fields': fields}
