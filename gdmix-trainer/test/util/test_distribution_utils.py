import os
import tempfile
import tensorflow as tf

from gdmix.util.distribution_utils import shard_input_files


class TestDistributionUtils(tf.test.TestCase):
    """
    Test distribution utils
    """

    def setUp(self):
        self._base_dir = tempfile.mkdtemp()
        for i in range(10):
            with open(os.path.join(self._base_dir, f'{i}.avro'), 'w') as f:
                f.write("test")
        for i in range(10):
            with open(os.path.join(self._base_dir, f'{i}.tfrecord'), 'w') as f:
                f.write("test")

    def tearDown(self):
        tf.io.gfile.rmtree(self._base_dir)

    def test_shard_input_files_with_wrong_params(self):
        with self.assertRaises(AssertionError):
            shard_input_files(self._base_dir, 1, 2)
        with self.assertRaises(AssertionError):
            shard_input_files(self._base_dir, -1, -2)
        with self.assertRaises(tf.errors.NotFoundError):
            shard_input_files(os.path.join(self._base_dir, "nowhere/nofile"), 3, 2)

    def test_shard_input_files_with_directory(self):
        shard_files, _ = shard_input_files(self._base_dir, 2, 0)
        expected_files = [os.path.join(self._base_dir, f'{i}.avro') for i in range(10)]
        self.assertAllEqual(shard_files, expected_files)

    def test_shard_input_file_with_filename_pattern(self):
        input_file_pattern = os.path.join(self._base_dir, "*.tfrecord")
        shard_files, indicator = shard_input_files(input_file_pattern, 3, 1)
        expected_files = [os.path.join(self._base_dir, f'{i}.tfrecord') for i in range(1, 10, 3)]
        self.assertAllEqual(shard_files, expected_files)
        self.assertFalse(indicator)

    def test_shard_input_file_with_more_shards(self):
        input_file_pattern = os.path.join(self._base_dir, "*.tfrecord")
        shard_files, indicator = shard_input_files(input_file_pattern, 20, 1)
        expected_files = [os.path.join(self._base_dir, '1.tfrecord')]
        self.assertAllEqual(shard_files, expected_files)
        self.assertTrue(indicator)
        shard_files, indicator = shard_input_files(input_file_pattern, 20, 19)
        self.assertEqual(len(shard_files), 0)
        self.assertTrue(indicator)
