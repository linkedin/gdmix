package com.linkedin.gdmix.parsers

import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert.assertEquals

/**
 * Unit tests for AreaUnderROCCurveEvaluatorParser.
 */
class AreaUnderROCCurveEvaluatorParserTest {

  @DataProvider
  def dataCompleteArgs(): Array[Array[Any]] = {
    Array(
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // miss metricsInputDir
      Array(
        Seq(
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore")),
      // miss outputMetricFile
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore")),
      // miss labelColumnName
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--predictionColumnName", "predictionScore")),
      // miss predictionColumnName
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = AreaUnderROCCurveEvaluatorParser.parse(completeArgs)
    val expectedParams = AreaUnderROCCurveEvaluatorParams(
      metricsInputDir = "global/validationScore",
      outputMetricFile = "global/metric/0",
      labelColumnName = "response",
      predictionColumnName = "predictionScore"
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
    AreaUnderROCCurveEvaluatorParser.parse(inCompleteArgs)
  }
}
