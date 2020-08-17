package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.configs.{ColumnConfig, DataType, EffectConfig}
import com.linkedin.gdmix.utils.IoUtils.readFile
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

/**
 * Unit tests for [[EffectConfigParser]].
 */
class EffectConfigParserTest {

  final val CONFIG_FILE_WITH_TWO_FIXED_EFFECTS = "configs/ConfigWithTwoFixedEffects.json"
  final val EFFECT_CONFIG_FILE = "configs/EffectConfigs.json"
  final val LABEL_NOT_IN_COLUMNS_FILE = "configs/LabelNotInColumns.json"
  final val ENTITY_NOT_IN_COLUMNS_FILE = "configs/EntityNotInColumns.json"

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testTwoFixedEffect(): Unit = {
    val configJson = readFile(null, CONFIG_FILE_WITH_TWO_FIXED_EFFECTS, true)
    EffectConfigParser.getEffectConfigList(configJson)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testLabelNotInColumns(): Unit = {
    val configJson = readFile(null, CONFIG_FILE_WITH_TWO_FIXED_EFFECTS, true)
    EffectConfigParser.getEffectConfigList(configJson)
  }

  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testEntityNotInColumns(): Unit = {
    val configJson = readFile(null, CONFIG_FILE_WITH_TWO_FIXED_EFFECTS, true)
    EffectConfigParser.getEffectConfigList(configJson)
  }

  @Test
  def testEffectConfigParser(): Unit = {
    val configJson = readFile(null, EFFECT_CONFIG_FILE, true)
    val parsedConfigList = EffectConfigParser.getEffectConfigList(configJson)

    // construct expected configs
    val fixedEffectCol1 = ColumnConfig("response", DataType.int, Seq(), Some(false), Some(false), Some(true))
    val fixedEffectCol2 = ColumnConfig("label", DataType.float, Seq(), Some(false), Some(false), Some(true))
    val fixedEffectCol3 = ColumnConfig("global", DataType.float, Seq(12), Some(false), Some(false), Some(true))

    val randomEffectCol1 = ColumnConfig("weight", DataType.float, Seq(), Some(false), Some(false), Some(false))
    val randomEffectCol2 = fixedEffectCol1
    val randomEffectCol3 = ColumnConfig("memberId", DataType.string, Seq(), Some(false), Some(false), Some(true))
    val randomEffectCol4 = ColumnConfig("per_member", DataType.float, Seq(2, 3), Some(false), Some(false), Some(true))

    val fixedEffect = EffectConfig(
      false,
      "fixed-effect",
      None,
      Some(Seq("response", "label")),
      Seq(fixedEffectCol1, fixedEffectCol2, fixedEffectCol3))

    val randomEffect = EffectConfig(
      true,
      "per-member",
      Some("memberId"),
      Some(Seq("response")),
      Seq(randomEffectCol1, randomEffectCol2, randomEffectCol3, randomEffectCol4))

    assertEquals(parsedConfigList, Seq(fixedEffect, randomEffect))
  }
}
