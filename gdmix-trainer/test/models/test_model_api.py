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
        """
        Train training model.

        Args:
            self: (todo): write your description
            training_data_path: (str): write your description
            validation_data_path: (str): write your description
            metadata_file: (str): write your description
            checkpoint_path: (str): write your description
            execution_context: (todo): write your description
            schema_params: (dict): write your description
        """
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
        """
        Predict the model.

        Args:
            self: (array): write your description
            output_dir: (str): write your description
            input_data_path: (str): write your description
            metadata_file: (str): write your description
            checkpoint_path: (str): write your description
            execution_context: (array): write your description
            schema_params: (array): write your description
        """
        super(ConcreteModel, self).predict(output_dir,
                                           input_data_path,
                                           metadata_file,
                                           checkpoint_path,
                                           execution_context,
                                           schema_params)

    def export(self, output_model_dir):
        """
        Export model output directory.

        Args:
            self: (todo): write your description
            output_model_dir: (str): write your description
        """
        super(ConcreteModel, self).export(output_model_dir)

    def _parse_parameters(self, raw_model_parameters):
        """
        Parses the raw_parametereter.

        Args:
            self: (todo): write your description
            raw_model_parameters: (str): write your description
        """
        pass


class TestAbstractModel(tf.test.TestCase):
    """Test abstract Model class."""

    raw_model_parameters = None
    concrete_model = ConcreteModel(raw_model_parameters)

    def test_train(self):
        """
        Train the model.

        Args:
            self: (todo): write your description
        """
        self.assertRaises(NotImplementedError, self.concrete_model.train, None, None, None, None, None, None)

    def test_predict(self):
        """
        Predict the test.

        Args:
            self: (todo): write your description
        """
        self.assertRaises(NotImplementedError, self.concrete_model.predict, None, None, None, None, None, None)

    def test_export(self):
        """
        Test if the test.

        Args:
            self: (todo): write your description
        """
        self.assertRaises(NotImplementedError, self.concrete_model.export, None)


if __name__ == '__main__':
    tf.test.main()
