import tensorflow as tf
from gdmix.util.io_utils import read_json_file, namedtuple_with_defaults


class DatasetMetadata:
    """Abstract Metadata class from which all dataset metadata classes derive"""

    # define mapping of dtype in meta data to dtype in TensorFlow
    TO_TF_DTYPE = {
        'int': tf.int32,
        'long': tf.int64,
        'float': tf.float32,
        'double': tf.float64,
        'bytes': tf.string,
        'string': tf.string
    }

    TF_INT_DTYPES = {tf.int8, tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int16, tf.int32, tf.int64}

    FEATURES = "features"
    LABELS = "labels"
    INDICES = "indices"
    VALUES = "values"
    NUMBER_OF_TRAINING_SAMPLES = "numberOfTrainingSamples"
    SUPPORTED_TYPES = frozenset(['int', 'long', 'float', 'double', 'bytes', 'string'])
    METADATA_FIELDS = frozenset(["name", "dtype", "shape", "isSparse"])
    METADATA_FIELD_DEFAULT_VALUES = (None, None, None, False)
    MetadataInfo = namedtuple_with_defaults("MetadataInfo", METADATA_FIELDS, defaults=METADATA_FIELD_DEFAULT_VALUES)

    def __init__(self, path_or_metadata):
        """
        Take a metadata str or dict to build up the tensor metadata infos

        :param path_or_metadata: Path to the metadata file or a JSON dict
        corresponding to the metadata
        """
        # ensure m is dict
        if isinstance(path_or_metadata, str):
            try:
                path_or_metadata = read_json_file(path_or_metadata)
            except Exception as err:
                raise ("Input of type str must be a valid JSON file. {}".format(err))

        # ensure features and labels are list
        if not isinstance(path_or_metadata.get(self.FEATURES, []), list):
            raise TypeError("Features must be a list. Type {} detected."
                            .format(type(path_or_metadata[self.FEATURES])))
        if not isinstance(path_or_metadata.get(self.LABELS, []), list):
            raise TypeError("Labels must be a list. Type {} detected."
                            .format(type(path_or_metadata[self.LABELS])))

        def parseMetadata(key):
            tensors = {}
            for entity in path_or_metadata.get(key, []):
                name = entity["name"]
                # Check if there are duplicated names in the metadata
                if name in tensors:
                    raise ValueError("Tensor name in your metadata appears more than once:{}".format(name))
                tensors[name] = self._build_metadata_info(entity.copy())
            return tensors

        try:
            feature_tensors = parseMetadata(self.FEATURES)
            label_tensors = parseMetadata(self.LABELS)
        except (TypeError, ValueError) as err:
            raise ValueError("Invalid field: {}".format(err))

        self._tensors = {**feature_tensors, **label_tensors}
        self._features = list(feature_tensors.values())
        self._labels = list(label_tensors.values())
        self._feature_names = list(feature_tensors.keys())
        self._label_names = list(label_tensors.keys())
        self._number_of_training_samples = path_or_metadata.get(
            "numberOfTrainingSamples", -1)

    @classmethod
    def _build_metadata_info(cls, metadata_dict):
        """
        Create namedtuple from metadata dict
        :param metadata_dict: the metadata in dict form
        :return: metadata namedtuple
        """
        if not cls.METADATA_FIELDS.issubset(metadata_dict.keys()):
            raise ValueError("Required metadata fields are {0}. "
                             "Proved fields are {1}".format(",".join(
                              cls.METADATA_FIELDS), ",".join(metadata_dict.keys())))
        metadata_obj = cls.MetadataInfo(**metadata_dict)
        if metadata_obj.name is None or not isinstance(metadata_obj.name, str):
            raise ValueError("Feature name can not be None and must be str")
        if metadata_obj.dtype not in cls.SUPPORTED_TYPES:
            raise ValueError("User provided dtype '{}' is not supported. "
                             "Supported types are '{}'.".format(
                              metadata_obj.dtype, list(cls.SUPPORTED_TYPES)))
        metadata_obj = metadata_obj._replace(dtype=cls.TO_TF_DTYPE[metadata_obj.dtype])
        if metadata_obj.shape is None or not isinstance(metadata_obj.shape, list):
            raise ValueError("Feature shape can not be None and must be a list")
        return metadata_obj

    def get_features(self):
        return self._features.copy()

    def get_labels(self):
        return self._labels.copy()

    def get_label_names(self):
        return self._label_names.copy()

    def get_feature_names(self):
        return self._feature_names.copy()

    def get_feature_shape(self, feature_name):
        return next(filter(lambda x: x.name == feature_name, self.get_features())).shape

    def get_tensors(self):
        return self._tensors.copy()

    def get_number_of_training_samples(self):
        return self._number_of_training_samples

    @staticmethod
    def map_int(in_dtype):
        """
        TFRecord features only support three data types:
        1. tf.float32
        2. tf.int64
        3. tf.string
        This function maps int32 and int16 to int64 and
        leave other types intact.
        :param in_dtype: Input TF data type
        :return: Mapped TF data type
        """

        if in_dtype in DatasetMetadata.TF_INT_DTYPES:
            return tf.int64
        else:
            return in_dtype
