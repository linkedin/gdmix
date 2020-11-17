import zipfile
import urllib.request
import shutil
import json
import numpy as np
import os
import pandas as pd
import pickle
import re
import shutil
import string
import tensorflow as tf
import argparse
import sys

GENRE = ['unknown', 'Action', 'Adventure', 'Animation',
         'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
         'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi',
         'Thriller', 'War', 'Western']
USER_FEATURE_COLUMNS = ['age', 'gender', 'occupation']
MOVIE_FEATURE_COLUMNS = GENRE + ['release_date']
USER_FEATURE_VALUES = ['age', 'M', 'F', 'administrator', 'artist', 'doctor', 'educator', 'engineer',
                       'entertainment', 'executive', 'healthcare', 'homemaker', 'lawyer',
                       'librarian', 'marketing', 'none', 'other', 'programmer', 'retired',
                       'salesman', 'scientist', 'student', 'technician', 'writer']
MOVIE_FEATURE_VALUES = MOVIE_FEATURE_COLUMNS

DETEXT_LABEL = 'label'
DETEXT_QUERY = 'doc_query'
DETEXT_WIDE_IDX = 'wide_ftrs_sp_idx'
DETEXT_WIDE_VAL = 'wide_ftrs_sp_val'

GLOBAL_FEATURE_VALUES = USER_FEATURE_VALUES + MOVIE_FEATURE_VALUES
GLOBAL_INDEX_MAP = {GLOBAL_FEATURE_VALUES[i]: i for i in range(len(GLOBAL_FEATURE_VALUES))}

USER_INDEX_MAP = {USER_FEATURE_VALUES[i]: i for i in range(len(USER_FEATURE_VALUES))}

MOVIE_INDEX_MAP = {MOVIE_FEATURE_VALUES[i]: i for i in range(len(MOVIE_FEATURE_VALUES))}


def get_parser():
    """ Creates an argument parser.  """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--url',
        type=str,
        required=False,
        default="http://files.grouplens.org/datasets/movielens/ml-100k.zip",
        help='url to download movielens data')
    parser.add_argument(
        '--dest_path',
        type=str,
        required=False,
        default="./",
        help='local path to download and save processed movieLens data')
    return parser


def download_movieLens(url, dest_path):
    """ Download movie lens data. """
    file_name = url.split('/')[-1]
    with urllib.request.urlopen(url) as response, open(file_name, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
        with zipfile.ZipFile(file_name) as zf:
            zf.extractall(dest_path)
    os.remove(file_name)
    dest_dir = os.path.join(dest_path, os.path.splitext(file_name)[0])
    return dest_dir

def cleanup_text(input):
    # remove release data which is at the end of the title e.g. (1995)
    input = re.sub(r'\([^)]*\)$', '', input)
    tokens = []
    for d in input.split():
        t = d.strip().lower()
        t = re.sub(r'([%s])' % re.escape(string.punctuation), r' ', t)
        t = re.sub(r'\\.', r' ', t)
        t = re.sub(r'\s+', r' ', t)
        t = t.strip()
        tokens.append(t)
    sentence = ' '.join(tokens)
    return sentence


def read_data(data_dir):
    """Read the data using pandas. """
    data_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    # drop timestamp
    data = pd.read_csv(os.path.join(data_dir, 'u.data'), '\t', names=data_names).drop('timestamp', 1)
    item_names = ['movie_id', 'title', 'release_date', 'video_release_date',
                  'IMDb_URL'] + GENRE
    items = pd.read_csv(os.path.join(data_dir, 'u.item'), '|', names=item_names, encoding = "ISO-8859-1")
    # clean up title
    items = items.drop(['video_release_date', 'IMDb_URL'], 1)
    items['title'] = items['title'].apply(cleanup_text)
    user_names = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    # drop zip_code
    users = pd.read_csv(os.path.join(data_dir, 'u.user'), '|', names=user_names).drop('zip_code', 1)
    return data, items, users


def process_data(data, items, users):
    """Process data such as join features by id, normalize feature values, et.al. """
    # normalized release_date by 2020
    items['release_date'] = items['release_date'].apply(lambda x: float(str(x).split('-')[-1]) / 2000.0)
    # normalized the age:
    users['age'] = users['age'] / 100.0
    # append uid to data
    data.insert(0, 'uid', range(len(data)))
    # binarize the score, use 3.0 as threshold.
    data['response'] = data['rating'].apply(lambda x: 1 if x > 3.0 else 0)
    data = data.drop('rating', 1)
    # create per_movie dataset, per-movie features are user features
    per_movie = data.join(users.set_index('user_id'), on='user_id', lsuffix='', rsuffix='_other')
    # create per_user dataset, per_user features are movie features
    per_user = data.join(items.set_index('movie_id'), 'movie_id', lsuffix='', rsuffix='_other')
    # join to create the global_data
    global_data = per_user.join(users.set_index('user_id'), 'user_id', lsuffix='', rsuffix='_other')
    global_data.sort_values(by=['uid'], inplace=True)
    per_movie.sort_values(by=['uid'], inplace=True)
    per_user.sort_values(by=['uid'], inplace=True)
    return global_data, per_movie, per_user


def tensorize_data(data, kept_columns, numeric_columns, categorical_columns, index_map, feature_bag):
    """Tensorize data so that they can be consumed by Tensorflow dataset API. """
    indices_name = f'{feature_bag}_indices'
    values_name = f'{feature_bag}_values'
    tdata = data.filter(kept_columns, axis=1)
    index_column = []
    value_column = []
    for i, row in enumerate(data.itertuples()):
        indices, values = [], []
        for key in row._fields:
            value = getattr(row, key)
            if key in numeric_columns:
                index = index_map[key]
            elif key in categorical_columns:
                index = index_map[value]
                value = 1.0
            else:
                continue
            if abs(value) > 1e-6:
                indices.append(index)
                values.append(float(value))
        # sort the indices
        indices, values = zip(*sorted(zip(indices, values)))
        index_column.append(indices)
        value_column.append(values)
    tdata[indices_name] = index_column
    tdata[values_name] = value_column
    return tdata


def generate_or_load_train_test_masks(data, train_percentage, output_file=None):
    """Generate masks for spliting the dataset into train and test by given percentage. """
    try:
        with open(output_file, 'rb') as f:
            return pickle.load(f)
        print(f'loaded data split mask from {output_file}')
    except FileNotFoundError:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < train_percentage]
        neg_mask = [not x for x in mask]
        if output_file:
            with open(output_file, 'wb') as f:
                pickle.dump((mask, neg_mask), f)
        return mask, neg_mask


def split_train_test(data, masks):
    """Split dataset into train and test datasets. """
    mask, neg_mask = masks[0], masks[1]
    train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data


def prepare_dir(directory):
    shutil.rmtree(directory, ignore_errors=True)
    os.makedirs(directory)


def save_tfrecord(data, feature_bag, other_features_map, label_name, output_file, detext=False):
    """Serialize data to TF record."""

    def get_example(row, feature_bag, other_features_map, label_name, detext):
        if detext:
            indices_name = DETEXT_WIDE_IDX
            values_name = DETEXT_WIDE_VAL
        else:
            indices_name = f'{feature_bag}_indices'
            values_name = f'{feature_bag}_values'
        tfr_features = {
            indices_name: tf.train.Feature(int64_list=tf.train.Int64List(
                value=getattr(row, indices_name))),
            values_name: tf.train.Feature(float_list=tf.train.FloatList(
                value=getattr(row, values_name))),
        }
        if detext:
            tfr_features[label_name] = tf.train.Feature(float_list=tf.train.FloatList(
                value=[getattr(row, label_name)]))
        else:
            tfr_features[label_name] = tf.train.Feature(int64_list=tf.train.Int64List(
                value=[getattr(row, label_name)]))
        for key in other_features_map:
            feature_type = other_features_map[key]
            feature = None
            if feature_type in [tf.int16, tf.int32, tf.int64]:
                feature = tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[getattr(row, key)]))
            elif feature_type == tf.float:
                feature = tf.train.Feature(float_list=tf.train.FloatList(
                    value=getattr(row, key)))
            else:
                print(f'unknown type {feature_type}')
            if feature:
                tfr_features[key] = feature
        if detext:
            title = str.encode(getattr(row, DETEXT_QUERY))
            tfr_features[DETEXT_QUERY] = tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[title]))
        features = tf.train.Features(feature=tfr_features)
        return tf.train.Example(features=features)

    with tf.io.TFRecordWriter(output_file) as writer:
        for row in data.itertuples():
            example = get_example(row, feature_bag, other_features_map, label_name, detext)
            writer.write(example.SerializeToString())


def create_metadata(metadata, outfile):
    """Dump metadata for deserialization."""
    with open(outfile, 'w') as f:
        json.dump(metadata, f)


def write_feature_list(features, outfile):
    """Write feature list to file."""
    with open(outfile, 'w') as f:
        for feature in features:
            f.write(f'{feature},\n')


def convert_to_detext(indf):
    """Change column names to meet DeText's requirement. """

    def increase_one(x):
        a = np.array(x) + 1
        return a.tolist()

    outdf = indf.rename(columns={'response': DETEXT_LABEL, 'title': DETEXT_QUERY,
                                 'global_indices': DETEXT_WIDE_IDX, 'global_values': DETEXT_WIDE_VAL})
    outdf[DETEXT_WIDE_IDX] = outdf[DETEXT_WIDE_IDX].apply(increase_one)
    return outdf


def gen_vocab(titles, outfile):
    """Generate vocabulary for the training dataset. """
    vocab = set(['[PAD]', '[UNK]'])
    for t in titles:
        for w in t.split():
            vocab.add(w)
    # Write vocab to file
    with open(outfile, 'w', encoding='utf-8') as f:
        for v in vocab:
            f.write(f'{v}\n')

def get_metadatas():
    """ Meta data to transform to TensorFlow tensor format. """
    global_metadata = {
        "features": [{
            "name": "global",
            "dtype": "float",
            "shape": [len(GLOBAL_FEATURE_VALUES)],
            "isSparse": True
        }, {
            "name": "uid",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }, {
            "name": "movie_id",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }, {
            "name": "user_id",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }],
        "labels" : [{
            "name": "response",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }]
    }

    per_user_metadata = {
        "features": [{
            "name": "per_user",
            "dtype": "float",
            "shape": [len(MOVIE_FEATURE_VALUES)],
            "isSparse": True
        }, {
            "name": "uid",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }, {
            "name": "movie_id",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }, {
            "name": "user_id",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }],
        "labels" : [{
            "name": "response",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }]
    }

    per_movie_metadata = {
        "features": [{
            "name": "per_movie",
            "dtype": "float",
            "shape": [len(USER_FEATURE_VALUES)],
            "isSparse": True
        }, {
            "name": "uid",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }, {
            "name": "movie_id",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }, {
            "name": "user_id",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }],
        "labels" : [{
            "name": "response",
            "dtype": "int",
            "shape": [],
            "isSparse": False
        }]
    }
    return (global_metadata, per_user_metadata, per_movie_metadata)


def data_process(input_dir, base_output_dir):
    """ process data to meet GDMix's need. """
    output_dir = os.path.join(base_output_dir, "movieLens")
    data, items, users = read_data(input_dir)
    global_data, per_movie, per_user = process_data(data, items, users)
    kept_columns = ['user_id', 'movie_id', 'uid', 'response', 'title']
    per_user_numerical_features = MOVIE_FEATURE_COLUMNS
    per_user_categorical_features = []
    per_movie_numerical_features = ['age']
    per_movie_categorical_features = ['gender', 'occupation']
    global_numerical_features = per_user_numerical_features + per_movie_numerical_features
    global_categorical_features = per_user_categorical_features + per_movie_categorical_features


    # transform to tensors
    ts_global_data = tensorize_data(global_data,
                                    kept_columns,
                                    global_numerical_features,
                                    global_categorical_features,
                                    GLOBAL_INDEX_MAP,
                                    "global")

    ts_per_user_data = tensorize_data(per_user, kept_columns, per_user_numerical_features,
                                  per_user_categorical_features, MOVIE_INDEX_MAP, "per_user")

    ts_per_movie_data = tensorize_data(per_movie, kept_columns, per_movie_numerical_features,
                                   per_movie_categorical_features, USER_INDEX_MAP, "per_movie")

    # Split data into training and testing
    mask_file = os.path.join(base_output_dir, "mask.pkl")
    masks = generate_or_load_train_test_masks(ts_global_data, 0.8, mask_file)
    global_train, global_test = split_train_test(ts_global_data, masks)
    per_user_train, per_user_test = split_train_test(ts_per_user_data, masks)
    per_movie_train, per_movie_test = split_train_test(ts_per_movie_data, masks)

    # Serialize data to TF record, create metadata for GDMix training
    other_features_map = {'user_id': tf.int32, 'movie_id': tf.int32, 'uid': tf.int64}
    datasets = [[global_train, global_test],
                [per_user_train, per_user_test],
                [per_movie_train, per_movie_test]]
    feature_bag = ['global', 'per_user', 'per_movie']
    global_metadata, per_user_metadata, per_movie_metadata = get_metadatas()
    metadata_list = [global_metadata, per_user_metadata, per_movie_metadata]
    features = [GLOBAL_FEATURE_VALUES, MOVIE_FEATURE_VALUES, USER_FEATURE_VALUES]

    for i in range(3):
        train_dir = os.path.join(output_dir, f'{feature_bag[i]}/trainingData')
        validation_dir = os.path.join(output_dir, f'{feature_bag[i]}/validationData')
        metadata_dir = os.path.join(output_dir, f'{feature_bag[i]}/metadata')
        feature_dir = os.path.join(output_dir, f'{feature_bag[i]}/featureList')

        prepare_dir(train_dir)
        prepare_dir(validation_dir)
        prepare_dir(metadata_dir)
        prepare_dir(feature_dir)

        save_tfrecord(datasets[i][0], feature_bag[i], other_features_map, "response",
                      os.path.join(train_dir, f'train_data.tfrecord'), False)
        save_tfrecord(datasets[i][1], feature_bag[i], other_features_map, "response",
                      os.path.join(validation_dir, f'validation_data.tfrecord'), False)
        metadata = metadata_list[i]
        metadata['numberOfTrainingSamples'] = len(datasets[i][0])
        create_metadata(metadata,
                        os.path.join(metadata_dir, f'tensor_metadata.json'))
        write_feature_list(features[i], os.path.join(feature_dir, feature_bag[i]))

    # Transform global model data to meet DeText's requirement
    detext_dir = os.path.join(output_dir, 'detext')
    # train data
    detext_train = convert_to_detext(global_train)
    training_dir = os.path.join(detext_dir, 'trainingData')
    prepare_dir(training_dir)
    save_tfrecord(detext_train, None, other_features_map, DETEXT_LABEL,
                  os.path.join(training_dir, 'train_data.tfrecord'), True)
    # validation data
    detext_test = convert_to_detext(global_test)
    validation_dir = os.path.join(detext_dir, 'validationData')
    prepare_dir(validation_dir)
    save_tfrecord(detext_test, None, other_features_map, DETEXT_LABEL,
                  os.path.join(validation_dir, 'test_data.tfrecord'), True)

    # generate vocab file
    gen_vocab(items['title'], os.path.join(detext_dir, 'vocab.txt'))

    return output_dir


def main(args=None):
    parser = get_parser()
    args = parser.parse_args()
    input_dir = download_movieLens(args.url, args.dest_path)
    output_dir = data_process(input_dir, args.dest_path)
    print("movieLens data is processed and saved to", output_dir)


if __name__ == "__main__":
    main()
