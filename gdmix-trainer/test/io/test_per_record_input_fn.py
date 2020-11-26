import numpy as np
import os
import shutil
import tensorflow as tf
import tempfile

from gdmix.io.input_data_pipeline import per_record_input_fn, GZIP, GZIP_SUFFIX, ZLIB, ZLIB_SUFFIX
from gdmix.util import constants

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

num_records = 10
np.random.seed(0)
labels = np.random.randint(2, size=num_records, dtype=np.int32)
weight_shape = 5
weight_length = np.random.randint(weight_shape-1, size=num_records)+1
weight_values = [np.random.random(l).astype(np.float32) for l in weight_length]
weight_indices = [np.sort(np.random.choice(
    weight_shape, l, replace=False).astype(np.int64)) for l in weight_length]
f1 = np.arange(num_records, dtype=np.float32) + 10.0
json_string = """
{
  "numberOfTrainingSamples": 20,
  "features": [
    {
      "name": "weight",
      "dtype": "float",
      "shape": [5],
      "isSparse": true
    },
    {
      "name": "f1",
      "dtype": "float",
      "shape": [],
      "isSparse": false
    }
  ],
  "labels": [
    {
      "name": "response",
      "dtype": "int",
      "shape": [],
      "isSparse": false
    }
  ]
}
"""


class TestPerRecordInputFn(tf.test.TestCase):
    """Test per_record_input_fn."""
    def setUp(self):
        self.test_uncompressed_data_dir = tempfile.mkdtemp()
        self.test_zlib_data_dir = tempfile.mkdtemp()
        self.test_gzip_data_dir = tempfile.mkdtemp()
        self.fd, self.test_metadata_file = tempfile.mkstemp()
        self.num_shards = 2
        self.shard_index = 1
        self.batch_size = 2

        # generate tf record files
        self.generate_tfrecords(labels, weight_indices, weight_values, f1,
                                self.num_shards, self.test_uncompressed_data_dir)
        self.generate_tfrecords(labels, weight_indices, weight_values, f1,
                                self.num_shards, self.test_zlib_data_dir, ZLIB)
        self.generate_tfrecords(labels, weight_indices, weight_values, f1,
                                self.num_shards, self.test_gzip_data_dir, GZIP)
        # generate meatadata file
        self.generate_metadata(json_string, self.test_metadata_file)

    def tearDown(self):
        shutil.rmtree(self.test_uncompressed_data_dir)
        shutil.rmtree(self.test_zlib_data_dir)
        shutil.rmtree(self.test_gzip_data_dir)
        os.close(self.fd)
        os.remove(self.test_metadata_file)

    @staticmethod
    def generate_tfrecords(label_tensor, weight_indices_tensor,
                           weight_value_tensor, f1_tensor,
                           num_shards, output_dir, compression_type=None):
        """
        Create tfrecords from a few tensors
        :param label_tensor: The tensor representing labels
        :param weight_indices_tensor: The indices for the weight (sparse) feature
        :param weight_value_tensor: The values for the weight (sparse) feature
        :param f1_tensor: A feature tensor
        :param num_shards: The number of shards
        :param output_dir: The output directory where the tfrecord files are saved.
        :param compression_type: None (uncompressed), ZLIB or GZIP
        :return: None
        """
        if compression_type == GZIP:
            suffix = GZIP_SUFFIX
        elif compression_type == ZLIB:
            suffix = ZLIB_SUFFIX
        else:
            suffix = None

        def get_example(w_i, w_v, f, l):
            features = tf.train.Features(feature={
                'weight_indices': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=w_i)),
                'weight_values': tf.train.Feature(float_list=tf.train.FloatList(
                    value=w_v)),
                'f1': tf.train.Feature(float_list=tf.train.FloatList(
                    value=[f])),
                'response': tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[l]))
            })
            return tf.train.Example(features=features)

        for s in range(num_shards):
            if suffix:
                filename = f'data_{s}.tfrecord{suffix}'
            else:
                filename = f'data_{s}.tfrecord'
            output_filename = os.path.join(output_dir, filename)
            with tf.io.TFRecordWriter(output_filename, options=compression_type) as writer:
                for i in range(len(label_tensor)):
                    example = get_example(weight_indices_tensor[i],
                                          weight_value_tensor[i] + s,
                                          f1_tensor[i] + s,
                                          label_tensor[i] + s)
                    writer.write(example.SerializeToString())

    @staticmethod
    def generate_metadata(metadata, output_file):
        """
        Create metadata file from a Json string.
        :param metadata:
        :param output_file:
        :return:
        """
        with open(output_file, 'w') as f:
            f.write(metadata)

    @staticmethod
    def generate_sparse_tensors(indices, values, shape, batch_size, shard_idx):
        """
        Generate sparse tensor from indices and values.
        This is used to check the ones generated from dataset
        :param indices: The tensor for all the indices
        :param values: The tensor for all the values
        :param shape: The dense shape of the sparse tensor
        :param batch_size: batch size
        :param shard_idx: shard index, added to the value
        :return: A list of sparse tensors
        """
        length = len(indices)
        assert(length % batch_size == 0)
        sparse_tensors = []
        for i in range(length // batch_size):
            sparse_indices = []
            sparse_values = []
            for j in range(batch_size):
                row_idx = i * batch_size + j
                curr_indices = indices[row_idx]
                curr_values = values[row_idx] + shard_idx
                for k in range(len(curr_indices)):
                    sparse_indices.append([j, curr_indices[k]])
                sparse_values += curr_values.tolist()
            sparse_tensors.append(tf.sparse.SparseTensor(indices=sparse_indices,
                                                         values=sparse_values,
                                                         dense_shape=[batch_size, shape]))
        return sparse_tensors

    def _test_input_fn(self, data_dir):
        """
        Test training dataset.
        :return: None
        """
        batch_size = self.batch_size
        d = per_record_input_fn(data_dir,
                                self.test_metadata_file,
                                self.num_shards,
                                self.shard_index,
                                batch_size,
                                constants.TFRECORD)
        d_iter = tf.compat.v1.data.make_one_shot_iterator(d)
        item = d_iter.get_next()
        i = 0
        sparse_tensors = self.generate_sparse_tensors(weight_indices,
                                                      weight_values, 5, batch_size, self.shard_index)
        with self.session() as sess:
            sparse_tensors_val = sess.run(sparse_tensors)
            try:
                while True:
                    features, response = sess.run(item)
                    self.assertAllEqual(features['weight'].values, sparse_tensors_val[i].values)
                    self.assertAllEqual(features['weight'].indices, sparse_tensors_val[i].indices)
                    self.assertAllEqual(features['f1'], f1[i*batch_size:(i+1)*batch_size]
                                        + self.shard_index)
                    self.assertAllEqual(response['response'], labels[i*batch_size:(i+1)*batch_size]
                                        + self.shard_index)
                    i += 1
            except tf.errors.OutOfRangeError:
                pass
        self.assertEqual(i, num_records // self.batch_size)

    def test_uncompressed_input_fn(self):
        self._test_input_fn(self.test_uncompressed_data_dir)

    def test_zlib_input_fn(self):
        self._test_input_fn(self.test_zlib_data_dir)

    def test_gzip_input_fn(self):
        self._test_input_fn(self.test_gzip_data_dir)


if __name__ == '__main__':
    tf.test.main()
