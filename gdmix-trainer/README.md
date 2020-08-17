# GDMix

## What is it
Generalized Deep [Mixed Model](https://en.wikipedia.org/wiki/Mixed_model) (GDMix) is a framework to train non-linear fixed effect and random effect models. This kind of models are widely used in personalization of search and recommender systems. This project is an extension of our early effort on generalized linear models [Photon ML](https://github.com/linkedin/photon-ml). It is implemented in Tensorflow, Scipy and Spark.

The current version of GDMix supports logistic regression and [DeText](https://github.com/linkedin/detext) models for the fixed effect, then logistic regression for the random effects. In the future, we may support deep models for random effects if the increase complexity can be justified by improvement in relevance metrics.

## Supported models
### Logistic regression
As a basic classification model, logistic regression finds wide usage in search and recommender systems due to its model simplicity and training efficiency. Our implementation uses Tensorflow for data reading and gradient computation, and utilizes L-BFGS solver from Scipy. This combination takes advantage of the versatility of Tensorflow and fast convergence of L-BFGS. This mode is functionally equivalent to Photon-ML but with improved efficiency. Our internal tests show about 10% to 40% training speed improvement on various datasets.

### DeText models
DeText is a framework for ranking with emphasis on textual features. GDMix supports DeText training natively as a global model. A user can specify a fixed effect model type as DeText then provide the network specifications. GDMix will train and score it automatically and connect the model to the subsequent random effect models. Currently only the pointwise loss function from DeText is allowed to be connected with the logistic regression random effect models.

### Other models
GDMix can work with any deep learning fixed effect models. The interface between GDMix and other models is at the file I/O. A user can train a model outside GDMix, then score the training data with the model and save the scores in files, which are the input to the GDMix random effect training. This enables the user to train random effect models based on scores from a custom  fixed effect model that is not natively supported by GDMix.

## Training efficiency
For logistic regression models, the training efficiency is achieved by parallel training. Since the fixed effect model is usually trained on a large amount of data, synchronous training based on Tensorflow all-reduce operation is utilized. Each worker takes a portion of the training data and compute the local gradient. The gradients are aggregated then fed to the L-BFGS solver. The training dataset for each random effect model is usually small, however the number of models (e.g. individual models for all LinkedIn members) can be on the order of hundred of millions. This requires a partitioning and parallel training strategy, where each worker is responsible for a portion of the population and all the workers train their assigned models independently and simultaneously.

For DeText models, efficiency is achieved by either Tensorflow based parameter server asynchronous distributed training or Horovod based synchronous distributed training.
