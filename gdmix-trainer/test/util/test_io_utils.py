import csv
import numpy as np
import os
import tempfile
import tensorflow as tf

from gdmix.util.io_utils import export_linear_model_to_avro, get_feature_map, gen_one_avro_model,\
    load_linear_models_from_avro, read_feature_list


class TestIoUtils(tf.test.TestCase):
    """
    Test io utils
    """

    def setUp(self):
        self.base_dir = tempfile.mkdtemp()
        self.name_term_tuple = [('a,a', 'x'), ('b', ''), ('c', 'z,z')]
        self.feature_file = os.path.join(self.base_dir, 'feature.txt')
        self.short_feature_file = os.path.join(self.base_dir, 'short_feature.txt')
        self.model_file = os.path.join(self.base_dir, 'model.avro')
        self.weight_indices = [np.array([0, 2]), np.array([0, 1, 2])]
        self.weight_values = [np.array([0.1, 0.2]), np.array([1.1, 0.3, 0.4])]
        self.biases = np.array([0.5, -0.7])
        self.expected_models = [[0.1, 0, 0.2, 0.5], [1.1, 0.3, 0.4, -0.7]]
        self.expected_short_models = [[0.1, 0, 0.5], [1.1, 0.3, -0.7]]

        with open(self.feature_file, 'w') as f:
            csvwriter = csv.writer(f)
            for name, term in self.name_term_tuple:
                csvwriter.writerow([name, term])

        # feature list with one fewer feature.
        with open(self.short_feature_file, 'w') as f:
            csvwriter = csv.writer(f)
            for i in range(len(self.name_term_tuple)-1):
                csvwriter.writerow(list(self.name_term_tuple[i]))

        export_linear_model_to_avro(model_ids=["model 1", "model 2"],
                                    list_of_weight_indices=self.weight_indices,
                                    list_of_weight_values=self.weight_values,
                                    biases=self.biases,
                                    feature_file=self.feature_file,
                                    output_file=self.model_file)

    def tearDown(self):
        tf.io.gfile.rmtree(self.base_dir)

    def testReadFeatureList(self):
        feature_list = read_feature_list(self.feature_file)
        self.assertAllEqual(feature_list, self.name_term_tuple)

    def testGetFeatureMap(self):
        feature_map = get_feature_map(self.feature_file)
        for index, feature in enumerate(self.name_term_tuple):
            self.assertEqual(index, feature_map[feature])

    def testLoadModel(self):
        models = load_linear_models_from_avro(self.model_file, self.feature_file)
        for model, expected in zip(models, self.expected_models):
            self.assertAllEqual(model, expected)
        short_models = load_linear_models_from_avro(self.model_file, self.short_feature_file)
        for i in range(len(short_models)):
            self.assertAllEqual(short_models[i], self.expected_short_models[i])

    def testGenOneAvroModel(self):
        """
        Test avro model generation.
        :return: None
        """
        model_id = '1234'
        model_class = 'com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel'
        weights = np.array([[1.2, 3.4, 5.6]])
        weight_indices = np.arange(3)
        bias = 7.8
        feature_list = [('f1,2', 't1'), ('f2', ''), ('f3', 't3,3')]

        records_avro = gen_one_avro_model(model_id, model_class, weight_indices, weights, bias, feature_list)
        records = {u'modelId': model_id, u'modelClass': model_class, u'means': [
            {u'name': '(INTERCEPT)', u'term': '', u'value': 7.8},
            {u'name': 'f1,2', u'term': 't1', u'value': 1.2},
            {u'name': 'f2', u'term': '', u'value': 3.4},
            {u'name': 'f3', u'term': 't3,3', u'value': 5.6}
        ], u'lossFunction': ""}
        self.assertDictEqual(records_avro, records)
