import abc


class Model(abc.ABC):
    """
    Abstract class. Must be subclassed to support fixed or random effect training.

    The deriving subclasses can rely on different model frameworks like TF Estimator, session-based training, Keras etc.

    Supports the following functionality:

    1) Encapsulates model graph specific to fixed or random effect
    3) Interfaces with the underlying training framework.
    3) Wires together model graph, loss functions, optimizers and metrics
    4) Exposes APIs for model compiling, training, prediction, export to be used in the business logic of the driver
    """
    def __init__(self, raw_model_params):
        self.model_params = self._parse_parameters(raw_model_params)
        self.metadata_file = None
        self.checkpoint_path = None
        self.training_data_dir = None
        self.validation_data_dir = None

    @abc.abstractmethod
    def train(self,
              training_data_dir,
              validation_data_dir,
              metadata_file,
              checkpoint_path,
              execution_context,
              schema_params):
        """
        Fit/train the model
        The interface should use internal model parameters `model_params` for model-specifc params.
        The data path arguments' values could be different from what users specify in the config
        because gdmix internally partitions the data into different chunks for distributed training/inference.
        :param training_data_dir   the path to training data
        :param validation_data_dir the path to validation data
        :param metadata_file        the path to tensor metadata file
        :param checkpoint_path      the path to designated savedmodel/checkpoint directory
        :param execution_context    the tensorflow cluster setup
        :param schema_params        parameters for schema field keyword definition
        :return:    None
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def predict(self,
                output_dir,
                input_data_path,
                metadata_file,
                checkpoint_path,
                execution_context,
                schema_params):
        """
        Run inference on a provided dataset
        :param output_dir:          the path to which inference output should be written
        :param input_data_path      the path to validation data
        :param metadata_file        the path to tensor metadata file
        :param checkpoint_path      the path to designated savedmodel/checkpoint directory
        :param execution_context    the tensorflow cluster setup
        :param schema_params:       parameters for schema field keyword definition
        :return:    None
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def export(self, output_model_dir):
        """
        Export TF model into the SavedModel format
        :param output_model_dir:    model directory where model should be exported
        :return:    None
        """
        raise NotImplementedError('Must be implemented in subclasses.')

    @abc.abstractmethod
    def _parse_parameters(self, raw_model_parameters):
        """
        Parse model-specific parameters. This excludes generic parameters like path to training set, optimization algo
        etc. which are necessary for all models
        :param raw_model_parameters:     TF Dataset object
        :return:   Parsed dict of model-specific arguments
        """
        raise NotImplementedError('Must be implemented in subclasses.')
