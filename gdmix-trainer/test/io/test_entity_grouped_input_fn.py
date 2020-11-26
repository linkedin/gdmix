import os
import tempfile
import tensorflow as tf

from gdmix.io.input_data_pipeline import per_entity_grouped_input_fn, GZIP, GZIP_SUFFIX, ZLIB, ZLIB_SUFFIX
from gdmix.util import constants

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class TestPerEntityGroupedInputFn(tf.test.TestCase):
    """Test per_entity_grouped_input_fn."""
    def setUp(self):
        self._base_dir = tempfile.mkdtemp()
        self._metadata_file = os.path.join(self._base_dir, "data.json")
        self._uncompressed_input_file = os.path.join(self._base_dir, "data.tfrecord")
        self._zlib_input_file = os.path.join(self._base_dir, f"data.tfrecord.{ZLIB_SUFFIX}")
        self._gzip_input_file = os.path.join(self._base_dir, f"data.tfrecord.{GZIP_SUFFIX}")
        metadata = '{"features":[' \
                   '{"name": "global", "dtype": "float", "shape": [100], "isSparse": true},' \
                   '{"name": "weight", "dtype": "float", "shape": [], "isSparse": false},' \
                   '{"name": "uid", "dtype": "int", "shape": [], "isSparse": false},' \
                   '{"name": "member_id", "dtype": "int", "shape": [], "isSparse": false}],' \
                   '"labels" :[' \
                   '{"name": "label", "dtype": "int", "shape": [], "isSparse": false}]}'
        # Write the meta data
        with open(self._metadata_file, 'w') as f:
            f.write(metadata)

        # Prepare the member data, this is grouped
        # there are two members, first member has 2 records, 2nd member has 1 record.
        # global_values: two members
        global_values = [[[1.0, 2.0, 3.0, 5.0, 6.6], [1.0, 2.0]], [[-3.5, 2.3]]]

        # global_indices: two members
        global_indices = [[[0, 7, 60, 80, 95], [34, 57]], [[10, 11]]]

        # Two weights corresponding to two members.
        weights = [[1.0, 2.0], [1.0]]

        # Two uids corresponding to two members.
        uids = [[10, 20], [23]]

        # MemberId for 2 members.
        # note member_id is a scalar
        member_ids = [100034, 100]

        # Labels for 2 members.
        labels = [[0, 1], [1]]

        # Set up the expected values
        self._expected_dict = {
            'global_values': global_values,
            'global_indices': global_indices,
            'weight': weights,
            'uid': uids,
            'label': labels,
            'member_id': member_ids
        }

        # Create feature
        examples = []
        for i in range(2):
            example = tf.train.SequenceExample(
                # member_ids is a scala per entity
                context=tf.train.Features(feature={
                    'member_id': tf.train.Feature(int64_list=tf.train.Int64List(value=[member_ids[i]])),
                    'weight': tf.train.Feature(float_list=tf.train.FloatList(value=weights[i])),
                    'uid': tf.train.Feature(int64_list=tf.train.Int64List(value=uids[i])),
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels[i]))
                }),
                # other fields are lists of variable length (variable number of records)
                feature_lists=tf.train.FeatureLists(feature_list={
                    'global_values': tf.train.FeatureList(feature=[
                        tf.train.Feature(float_list=tf.train.FloatList(value=x)) for x in global_values[i]
                    ]),
                    'global_indices': tf.train.FeatureList(feature=[
                        tf.train.Feature(int64_list=tf.train.Int64List(value=x)) for x in global_indices[i]
                    ])
                }))
            examples.append(example)
        with tf.io.TFRecordWriter(self._uncompressed_input_file) as writer:
            for i in range(2):
                writer.write(examples[i].SerializeToString())

        with tf.io.TFRecordWriter(self._zlib_input_file, options=ZLIB) as writer:
            for i in range(2):
                writer.write(examples[i].SerializeToString())

        with tf.io.TFRecordWriter(self._gzip_input_file, options=GZIP) as writer:
            for i in range(2):
                writer.write(examples[i].SerializeToString())

    def tearDown(self):
        tf.io.gfile.rmtree(self._base_dir)

    def test_entity_name_not_in_features(self):
        self.assertRaises(ValueError, per_entity_grouped_input_fn, self._uncompressed_input_file, self._metadata_file,
                          1, 0, 2, constants.TFRECORD, "random_feature")

    def _test_per_entity_group_input_fn(self, input_file):
        dataset = per_entity_grouped_input_fn(input_file, self._metadata_file,
                                              num_shards=1, shard_index=0,
                                              batch_size=2, data_format=constants.TFRECORD,
                                              entity_name="member_id")
        features, labels = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()

        def sparse_to_list_of_lists(spare_tensor_value):
            """
            This function convert SparseTensorValue to list of lists.
            The input_fn returns SparseTensorValue except the member_id. For example,
            global_values = [[[1.0, 2.0, 3.0, 5.0, 6.6], [1.0, 2.0]], [[-3.5, 2.3]]]
            SparseTensorValue(indices=array(
            [[0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 0, 4],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1]]), values=array([ 1. ,  2. ,  3. ,  5. ,  6.6,  1. ,  2. , -3.5,  2.3],
            dtype=float32), dense_shape=array([2, 2, 5]))
            Note the first dimension is the batch dim.
            This function convert the SparseTensorValue above back to the list of lists:
            [[[1.0, 2.0, 3.0, 5.0, 6.6], [1.0, 2.0]], [[-3.5, 2.3]]]
            :param spare_tensor_value: input SparseTensorValue
            :return: list of lists representation of the same tensor.
            """
            indices = spare_tensor_value.indices
            values = spare_tensor_value.values
            dense_shape = spare_tensor_value.dense_shape
            # 2D or higher, 1st dimension is batch
            max_dim = len(dense_shape)
            assert(max_dim > 1)
            indices_values = list(zip(indices, values))
            root = list()
            curr_dim = 0
            start = 0
            end = len(indices_values)
            get_list(indices_values, root, start, end, curr_dim, max_dim)
            return root

        def get_list(indices_values, parent, start, end, curr_dim, max_dim):
            if curr_dim == max_dim - 1:
                for i in range(start, end):
                    parent.append(indices_values[i][1])
                return
            prev_start = start
            for i in range(start, end):
                if i == end - 1 or indices_values[i][0][curr_dim] != \
                        indices_values[i+1][0][curr_dim]:
                    curr_list = list()
                    parent.append(curr_list)
                    get_list(indices_values, curr_list, prev_start, i+1, curr_dim+1, max_dim)
                    prev_start = i + 1

        with tf.compat.v1.Session() as session:
            features_val, labels_val = session.run([features, labels])
            # member_ids is a dense array, we can check directly.
            self.assertAllEqual(self._expected_dict['member_id'], features_val['member_id'])

            # check labels
            self.assertSequenceAlmostEqual(self._expected_dict["label"],
                                           sparse_to_list_of_lists(
                                               labels_val["label"]), places=5)
            # check 2d lists
            keys_2d = ['weight', 'uid']
            for key in keys_2d:
                self.assertSequenceAlmostEqual(self._expected_dict[key],
                                               sparse_to_list_of_lists(
                                                   features_val[key]), places=5)
            # check 3d lists
            keys_3d = ['global_values', 'global_indices']
            for key in keys_3d:
                expected = self._expected_dict[key]
                actual = sparse_to_list_of_lists(features_val[key])
                for i in range(2):
                    assert(len(expected[i]) == len(actual[i]))
                    for j in range(len(expected[i])):
                        self.assertSequenceAlmostEqual(expected[i][j], actual[i][j], places=5)

    def test_uncompressed_file(self):
        self._test_per_entity_group_input_fn(self._uncompressed_input_file)

    def test_zlib_file(self):
        self._test_per_entity_group_input_fn(self._zlib_input_file)

    def test_gzip_file(self):
        self._test_per_entity_group_input_fn(self._gzip_input_file)


if __name__ == '__main__':
    tf.test.main()
