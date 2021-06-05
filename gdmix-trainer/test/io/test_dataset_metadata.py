import os
import tensorflow as tf

from gdmix.io.dataset_metadata import DatasetMetadata

test_metadata_file = os.path.join(os.getcwd(), "test/resources/metadata")


class TestDatasetMetadata(tf.test.TestCase):
    """Test DatasetMetadata class."""
    dummy_metadata = DatasetMetadata(os.path.join(
        test_metadata_file, "valid_metadata.json"))
    feature_names = ["weight", "f1"]
    label_names = ["response"]

    def test_feature_names(self):
        self.assertEqual(self.dummy_metadata.get_feature_names(), self.feature_names)

    def test_label_names(self):
        self.assertEqual(self.dummy_metadata.get_label_names(), self.label_names)

    def test_invalid_type(self):
        msg_pattern = r"User provided dtype \'.*\' is not supported. Supported types are \'.*\'."
        with self.assertRaises(ValueError, msg=msg_pattern):
            DatasetMetadata(os.path.join(test_metadata_file, "invalid_type.json"))

    def test_invalid_name(self):
        msg_pattern = r"Feature name can not be None and must be str"
        with self.assertRaises(ValueError, msg=msg_pattern):
            DatasetMetadata(os.path.join(test_metadata_file, "invalid_name.json"))

    def test_invalid_shape(self):
        msg_pattern = r"Feature shape can not be None and must be a list"
        with self.assertRaises(ValueError, msg=msg_pattern):
            DatasetMetadata(os.path.join(test_metadata_file, "invalid_shape.json"))

    def test_duplicated_names(self):
        msg_pattern = r"The following tensor names in your metadata appears more than once:\['weight', 'response'\]"
        with self.assertRaises(ValueError, msg=msg_pattern):
            DatasetMetadata(os.path.join(test_metadata_file, "duplicated_names.json"))

    def test_map_int(self):
        int_dtypes = [tf.int8, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int16, tf.int32, tf.int64]
        for id in int_dtypes:
            assert tf.int64 == DatasetMetadata.map_int(id)
        assert tf.float32 == DatasetMetadata.map_int(tf.float32)
        assert tf.float16 == DatasetMetadata.map_int(tf.float16)
        assert tf.float64 == DatasetMetadata.map_int(tf.float64)
        assert tf.string == DatasetMetadata.map_int(tf.string)


if __name__ == '__main__':
    tf.test.main()
