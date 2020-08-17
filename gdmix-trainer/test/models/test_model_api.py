import tensorflow as tf

from gdmix.models.api import Model


class ConcreteModel(Model):
    """Derived model class"""

    def train(self,
              training_data_path,
              validation_data_path,
              metadata_file,
              checkpoint_path,
              execution_context,
              schema_params):
        super(ConcreteModel, self).train(training_data_path,
                                         validation_data_path,
                                         metadata_file,
                                         checkpoint_path,
                                         execution_context,
                                         schema_params)

    def predict(self,
                output_dir,
                input_data_path,
                metadata_file,
                checkpoint_path,
                execution_context,
                schema_params):
        super(ConcreteModel, self).predict(output_dir,
                                           input_data_path,
                                           metadata_file,
                                           checkpoint_path,
                                           execution_context,
                                           schema_params)

    def export(self, output_model_dir):
        super(ConcreteModel, self).export(output_model_dir)

    def _parse_parameters(self, raw_model_parameters):
        pass


class TestAbstractModel(tf.test.TestCase):
    """Test abstract Model class."""

    raw_model_parameters = None
    concrete_model = ConcreteModel(raw_model_parameters)

    def test_train(self):
        self.assertRaises(NotImplementedError, self.concrete_model.train, None, None, None, None, None, None)

    def test_predict(self):
        self.assertRaises(NotImplementedError, self.concrete_model.predict, None, None, None, None, None, None)

    def test_export(self):
        self.assertRaises(NotImplementedError, self.concrete_model.export, None)


if __name__ == '__main__':
    tf.test.main()
