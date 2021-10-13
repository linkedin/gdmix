package com.linkedin.gdmix.parsers

import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert.assertEquals

/**
 * Unit tests for EvaluatorParser.
 */
class EvaluatorParserTest {

  @DataProvider
  def dataCompleteArgs(): Array[Array[Any]] = {
    Array(
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore",
          "--metricName", "auc")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // miss metricsInputDir
      Array(
        Seq(
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore",
          "--metricName", "auc")),
      // miss outputMetricFile
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore",
          "--metricName", "auc")),
      // miss labelColumnName
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--predictionColumnName", "predictionScore",
          "--metricName", "auc")),
      // miss predictionColumnName
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response",
          "--metricName", "auc")),
      // miss metricName
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore")),
      // metricName not supported
      Array(
        Seq(
          "--metricsInputDir", "global/validationScore",
          "--outputMetricFile", "global/metric/0",
          "--labelColumnName", "response",
          "--predictionColumnName", "predictionScore",
          "--metricName", "UnsupportedMetric"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = EvaluatorParser.parse(completeArgs)
    val expectedParams = EvaluatorParams(
      metricsInputDir = "global/validationScore",
      outputMetricFile = "global/metric/0",
      labelColumnName = "response",
      predictionColumnName = "predictionScore",
      metricName = "auc"
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
    EvaluatorParser.parse(inCompleteArgs)
  }
}
