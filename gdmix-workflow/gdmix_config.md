
# GDMix Configs
To train fixed effect and random effect models using GDMix, users need to provide a GDMix config, which consists of configs for
fixed-effect and random-effect models. For distributed training, computing resource
configs for Tensorflow and Spark jobs are needed.

GDMix config examples for movieLens with a fixed-effect `global` model and two random effect `per-user` and `per-movie` models are available in directory `examples/movielens-100k`:
  - [lr-single-node-movieLens.config](examples/movielens-100k/lr-single-node-movieLens.config): train logistic regression models for the `global`, `per-user` and `per-movie` models
  - [lr-distributed-movieLens.config](examples/movielens-100k/lr-distributed-movieLens.config): same as above plus resource config for distributed training
  - [detext-single-node-movieLens.config](examples/movielens-100k/detext-single-node-movieLens.config): train a deep and wide neural network model for the `global` and logistic regression models for the `per-user` and `per-movie`
  - [detext-distributed-movieLens.config](examples/movielens-100k/detext-distributed-movieLens.config): same as above plus resource config for distributed training

## Logistic regression models
### Fixed-effect config
Required fields:
  - **name**: name of the model. String.
  - **training_data_dir**: path to training data directory. String.
  - **input_column_names**: input column names for label, unique id, weight and feature bag(the collection of features). String map.
  - **prediction_score_column_name**: column name for prediction score. String.
  - **metadata_file**: path to an input data tensor metadata file. String.
  - **feature_file**: path to a feature list file for outputing model in name-term-value format. String.

Optional fields:
  - **validation_data_dir**: path to validation data directory. String, default is "".
  - **regularize_bias**: whether to regularize the intercept. Ususally we don't put regularization on intercept since it is an important feature. Boolean, default is false.
  - **l2_reg_weight**: weight of L2 regularization for each feature bag. Float, default is 0.001.
  - **optimizer**: optimizer used in the training, currently support LBFGS only. Map, default values are {"name": "LBFGS", "params": [{ "lbfgs_tolerance": 1.0e-7, "num_of_lbfgs_iterations": 100, "num_of_lbfgs_curvature_pairs": 10 }] }
  - **metric**: metric of the model. String, support "AUC" and "NDCG" Default is "AUC".
  - **position_k**: the position to compute the truncated ndcg, only needed when metric is "NDCG". Integer, default is 1.
  - **copy_to_local**: whether copy training data to worker's local disk. Boolean, default is true.

### Random-effect config
Required fields include all fields from fixed-effect config plus:
  - **partition_entity**: the column name used to partition data in order to improve random effect model training parallelism. String.
  - **num_partitions**: number of partitions. Integer.

Optional fields include all fields from fixed-effect config plus:
  - **max_training_queue_size**: maximum number of training queue size in the producer/consumer model. The trainer is implemented in a producer/consumer model. The producer reads data from hard drive, then the consumers solve the optimization problem for each entity. The blocking queue synchcronizes both sides. Integer, default is 10.
  - **num_of_consumers**: the number of consumers (processes that optimizes the models). This specifies the parallelism inside a trainer. Integer, default is 2.
  - **enable_local_indexing**: whether to enable local indexing. Some dataset has large global feature space, but small per entity feature space. For example the total features in a dataset could be on the order of millions, but each member has only hundreds of features.  We should re-index the features to save memory footprint and increase the training efficiency. Boolean, default is true.

## Neural network model supported by DeText for fixed-effect
### Fixed-effect config
Please refer to DeText training manual [TRAINING.md](https://github.com/linkedin/detext/blob/master/TRAINING.md) for available parameters for the config, the config is a collection of key/value pair, a DeText config example for movieLens data can be found at [detext-single-node-movieLens.config](examples/movielens-100k/detext-single-node-movieLens.config)
