package com.linkedin.gdmix.configs

import org.testng.Assert.assertEquals
import org.testng.annotations.Test

/**
 * Unit tests for [[EffectConfig]].
 */
@Test
class EffectConfigTest {
  def testSort(): Unit = {
    val col1 = ColumnConfig("feature", DataType.float, Seq(1, 2), Some(true), Some(true), Some(false))
    val col2 = ColumnConfig("global", DataType.int)
    val first = EffectConfig(true, "per-member", Some("memberId"), Some(Seq("label")), Seq(col1, col2))
    val second = EffectConfig(false, "global", None,  Some(Seq("label", "response")), Seq(col2))
    val configs = Seq(first, second)
    assertEquals(configs.sorted, Seq(second, first))
  }
}
