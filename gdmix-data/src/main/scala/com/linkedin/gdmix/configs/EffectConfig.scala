package com.linkedin.gdmix.configs

import com.fasterxml.jackson.core.`type`.TypeReference
import com.fasterxml.jackson.module.scala.JsonScalaEnumeration

/**
 * Enumeration to represent different data type of tensors
 */
object DataType extends Enumeration {
  type DataType = Value
  val string, int, long, double, float, byte = Value
}

/**
 * This class is a workaround to Scala's Enumeration so that we can use
 * DataType.DataType as a type in the definition. See the example
 * in the ColumnConfig definition below, where the member "dtype" is defined as
 * DataType.DataType.
 * For details:
 * https://github.com/FasterXML/jackson-module-scala/wiki/Enumerations
 */
class DataTypeRef extends TypeReference[DataType.type]

/**
 * Case class for fixed or random effect config
 *
 * @param isRandomEffect Whether this is a random effect.
 * @param coordinateName Coordinate name for a fixed effect or a random effect.
 * @param perEntityName Entity name that random effect is based on. null for fixed effect.
 * @param labels A sequence of label column names.
 * @param columnConfigList A sequence of column configs.
 */
case class EffectConfig (
  isRandomEffect: Boolean,
  coordinateName: String,
  perEntityName: Option[String]=None,
  labels: Option[Seq[String]]=None,
  columnConfigList: Seq[ColumnConfig]
) extends Ordered[EffectConfig] {
  require(
    columnConfigList.nonEmpty,
    s"Please specify at least one column"
  )
  if (isRandomEffect) {
    require(!perEntityName.isEmpty)
  } else {
    require(perEntityName.isEmpty)
  }

  // We want the configs to be sorted such that the fixed effect precedes random effects.
  // This property is used in name-term-value to sparse/dense tensor conversion function.
  def compare (that: EffectConfig) = this.isRandomEffect.compare(that.isRandomEffect)
}

/**
 * Case class for column config
 *
 * @param name Column name.
 * @param dtype Intended data type after conversion.
 * @param shape The data shape of the column.
 * @param isInputNTV Whether this is in name-term-value format.
 * @param isOutputSparse Whether the output tensor should be a sparse tensor.
 * @param sharedFeatureSpace Whether the name-term shared the feature space, only used
 *                           for random effect data conversion.
 */

case class ColumnConfig(
  name: String,
  @JsonScalaEnumeration(classOf[DataTypeRef]) dtype: DataType.DataType,
  shape: Seq[Int] = Seq(),
  isInputNTV: Option[Boolean] = None,
  isOutputSparse: Option[Boolean] = None,
  sharedFeatureSpace: Option[Boolean] = None)
