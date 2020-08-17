import numpy as np
import os
import tensorflow as tf
from gdmix.util.io_utils import gen_one_avro_model


class TestUtils(tf.test.TestCase):
    """
    Test the functions in io_utils.py
    """
    tmp_path = os.path.join(os.path.expanduser("~"), 'tmp')
    test_resources = os.path.join(os.getcwd(), "test", "resources")

    def test_gen_one_avro_model(self):
        """
        Test avro model generation.
        :return: None
        """
        model_id = '1234'
        model_class = 'com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel'
        weights = np.array([[1.2, 3.4, 5.6]])
        weight_indices = np.arange(3)
        bias = np.array([7.8])
        feature_list = [['f1', 't1'], ['f2', ''], ['f3', 't3']]
        records_avro = gen_one_avro_model(model_id, model_class, weight_indices, weights, bias, feature_list)
        records = {u'modelId': model_id, u'modelClass': model_class, u'means': [
            {u'name': '(INTERCEPT)', u'term': '', u'value': 7.8},
            {u'name': 'f1', u'term': 't1', u'value': 1.2},
            {u'name': 'f2', u'term': '', u'value': 3.4},
            {u'name': 'f3', u'term': 't3', u'value': 5.6}
        ], u'lossFunction': ""}
        self.assertDictEqual(records_avro, records)


if __name__ == '__main__':
    tf.test.main()
