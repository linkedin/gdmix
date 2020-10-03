package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.configs.{DataType, EffectConfig, ColumnConfig}
import org.json4s.DefaultFormats
import org.json4s.ext.EnumNameSerializer
import org.json4s.jackson.JsonMethods.parse

/**
 * Parser for fixed or random effect configuration [[EffectConfig]]
 *
 */
object EffectConfigParser {

  /**
   * Get the fixed or random effect config with sanity check
   *
   * @param jsonString JSON format of a list of EffectConfig.
   * @return a list of EffectConfig
   */
  def getEffectConfigList(jsonString: String): Seq[EffectConfig] = {
    // Define implicit JSON4S default format
    implicit val formats = DefaultFormats + new EnumNameSerializer(DataType)
    // Use JSON4S to parse and extract a list of EffectConfig.
    val configList = parse(jsonString).extract[Seq[EffectConfig]]
    sanityCheck(configList)
  }

  /**
   * Sanity check the list of EffectConfig.
   * Throw an exception when there are more than 1 fixed effect.
   *
   * @param EffectConfig A sequence of EffectConfig to be checked
   * @return A sequence of EffectConfig
   */
  private def sanityCheck(configList: Seq[EffectConfig]): Seq[EffectConfig] = {

    // A sequence of EffectConfig that represent a dataset should only have one fixed-effect.
    val numFixedEffect = configList.foldLeft(0)((accum, config) => accum + (if (config.isRandomEffect) 0 else 1))
    if (numFixedEffect > 1) {
      throw new IllegalArgumentException(s"There should be only 1 fixed effect, but $numFixedEffect are present")
    }

    // Check indivdual EffectConfig
    val checkedConfigList = configList.map(config => checkEffectConfig(config))

    // Sort the configs such that the fixed effect is at the beginning.
    checkedConfigList.sorted
  }

  /**
   * Check the content of an EffectConfig.
   *
   * @param config The EffectConfig to be checked
   * @return An EffectConfig with missing column info added.
   */
  private def checkEffectConfig(config: EffectConfig): EffectConfig = {

    val columnNames = config.columnConfigList.map(column => column.name).toSet

    // Check if the labels are in columnConfig
    if (config.labels != None) {
      config.labels.get.map {
        label =>
          if (!columnNames.contains(label)) {
            throw new IllegalArgumentException(s"Label $label is not in column names")
          }
      }
    }

    // Check if perEntityName is in columnConfig
    if (config.perEntityName != None) {
      val entityName = config.perEntityName.get
      if (!columnNames.contains(entityName)) {
        throw new IllegalArgumentException(s"EntityName $entityName is not in column names")
      }
    }

    config.copy(columnConfigList = config.columnConfigList.map(column => checkColumnConfig(column)))
  }

  /**
   * Check the content of a ColumnConfig. Fill in the default values if possible.
   * Throw an exception when column expression and column configuration both exist in input feature information
   *
   * @param columnConfig ColumnConfig to be checked
   * @return A ColumnConfig with missing values filled with default
   */
  private def checkColumnConfig(columnConfig: ColumnConfig): ColumnConfig = {

    val isInputNTV = setDefaultBoolean(columnConfig.isInputNTV, false)
    val isOutputSparse = setDefaultBoolean(columnConfig.isOutputSparse, false)
    val sharedFeatureSpace = setDefaultBoolean(columnConfig.sharedFeatureSpace, true)

    if ((isInputNTV.get) && (columnConfig.dtype != DataType.float)) {
      throw new IllegalArgumentException(s"Name-Term-Value format output datatype must be float")
    }

    columnConfig.copy(
      isInputNTV = isInputNTV,
      isOutputSparse = isOutputSparse,
      sharedFeatureSpace = sharedFeatureSpace)
  }

  /**
   * A utility function that fill a Boolean value if missing from an option
   *
   * @param some an option value.
   * @param value a boolean value used as the default.
   * @return A Some with the default value when the input is None.
   */
  private def setDefaultBoolean(some: Option[Boolean], value: Boolean): Option[Boolean] = {
    if (some == None) Some(value) else some
  }
}
