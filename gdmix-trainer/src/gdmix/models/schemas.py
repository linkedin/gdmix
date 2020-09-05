# photon-ml output model format

BAYESIAN_LINEAR_MODEL_SCHEMA = """
{
  "type" : "record",
  "name" : "BayesianLinearModelAvro",
  "namespace" : "com.linkedin.photon.avro.generated",
  "doc" : "a generic schema to describe a Bayesian linear model with means and variances",
  "fields" : [ {
    "name" : "modelId",
    "type" : "string"
  }, {
    "name" : "modelClass",
    "type" : [ "null", "string" ],
    "doc" : "The fully-qualified class name of enclosing GLM model class. E.g.: com.linkedin.photon.ml.supervised.classification.LogisticRegressionModel",
    "default" : null
  }, {
    "name" : "means",
    "type" : {
      "type" : "array",
      "items" : {
        "type" : "record",
        "name" : "NameTermValueAvro",
        "doc" : "A tuple of name, term and value. Used as feature or model coefficient",
        "fields" : [ {
          "name" : "name",
          "type" : "string"
        }, {
          "name" : "term",
          "type" : "string"
        }, {
          "name" : "value",
          "type" : "double"
        } ]
      }
    }
  }, {
    "name" : "variances",
    "type" : [ "null", {
      "type" : "array",
      "items" : "NameTermValueAvro"
    } ],
    "default" : null
  }, {
    "name" : "lossFunction",
    "type" : [ "null", "string" ],
    "doc" : "The loss function used for training as the class name. E.g.: com.linkedin.photon.ml.function.LogisticLossFunction",
    "default" : null
  } ]
}
"""
