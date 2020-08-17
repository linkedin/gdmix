package com.linkedin.gdmix.configs

import com.fasterxml.jackson.module.scala.JsonScalaEnumeration

/**
 * Case class for the dataset metadata
 *
 * @param numberOfTrainingSamples Number of training samples
 * @param features Tensor metadata of a sequence of features
 * @param labels Tensor metadata of a sequence of labels
 */
case class DatasetMetadata(
  numberOfTrainingSamples: Option[Long] = None,
  features: Seq[TensorMetadata],
  labels: Option[Seq[TensorMetadata]] = None)

/**
 * Case class for the tensor metadata
 *
 * @param name Name of a tensor
 * @param dtype Data type of a tensor
 * @param shape Shape of a tensor
 * @param isSparse If it is a sparse tensor
 */
case class TensorMetadata(
  name: String,
  @JsonScalaEnumeration(classOf[DataTypeRef]) dtype: DataType.DataType,
  shape: Seq[Int],
  isSparse: Boolean = false
)
