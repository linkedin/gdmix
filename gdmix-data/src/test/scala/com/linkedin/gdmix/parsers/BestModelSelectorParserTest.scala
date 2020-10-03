package com.linkedin.gdmix.parsers

import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert.assertEquals

/**
 * Unit tests for BestModelSelectorParser.
 */
class BestModelSelectorParserTest {

  @DataProvider
  def dataCompleteArgs(): Array[Array[Any]] = {
    Array(
      Array(
        Seq(
          "--inputMetricsPaths", "gdmix/0/per-job/metric/0;gdmix/1/per-job/metric/0",
          "--inputModelPaths", "gdmix/0/per-job/model_output;gdmix/1/per-job/model_output",
          "--outputBestMetricsPath", "gdmix/best/metric",
          "--outputBestModelPath", "gdmix/best/model",
          "--hyperparameters", "eyIwIjogWyJnbG9iYWw6YmF0Y2hfc2l6",
          "--evalMetric", "auc",
          "--copyBestOutput", "True")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // copyBestOutput = true but no inputModelPaths
      Array(
        Seq(
          "--inputMetricsPaths", "gdmix/0/per-job/metric/0;gdmix/1/per-job/metric/0",
          "--outputBestMetricsPath", "gdmix/best/metric",
          "--outputBestModelPath", "gdmix/best/model",
          "--hyperparameters", "eyIwIjogWyJnbG9iYWw6YmF0Y2hfc2l6",
          "--evalMetric", "auc",
          "--copyBestOutput", "True")),
      // copyBestOutput is true but no outputBestMetricsPath
      Array(
        Seq(
          "--inputMetricsPaths", "gdmix/0/per-job/metric/0;gdmix/1/per-job/metric/0",
          "--inputModelPaths", "gdmix/0/per-job/model_output;gdmix/1/per-job/model_output",
          "--outputBestModelPath", "gdmix/best/model",
          "--hyperparameters", "eyIwIjogWyJnbG9iYWw6YmF0Y2hfc2l6",
          "--evalMetric", "auc",
          "--copyBestOutput", "True")),
      // miss inputMetricsPaths
      Array(
        Seq(
          "--outputBestModelPath", "gdmix/best/model",
          "--hyperparameters", "eyIwIjogWyJnbG9iYWw6YmF0Y2hfc2l6",
          "--evalMetric", "auc")),
      // miss outputBestModelPath
      Array(
        Seq(
          "--inputMetricsPaths", "gdmix/0/per-job/metric/0;gdmix/1/per-job/metric/0",
          "--hyperparameters", "eyIwIjogWyJnbG9iYWw6YmF0Y2hfc2l6",
          "--evalMetric", "auc")),
      // miss hyperparameters
      Array(
        Seq(
          "--inputMetricsPaths", "gdmix/0/per-job/metric/0;gdmix/1/per-job/metric/0",
          "--outputBestModelPath", "gdmix/best/model",
          "--evalMetric", "auc")),
      // miss evalMetric
      Array(
        Seq(
          "--inputMetricsPaths", "gdmix/0/per-job/metric/0;gdmix/1/per-job/metric/0",
          "--outputBestModelPath", "gdmix/best/model",
          "--hyperparameters", "eyIwIjogWyJnbG9iYWw6YmF0Y2hfc2l6"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = BestModelSelectorParser.parse(completeArgs)
    val expectedParams = BestModelSelectorParams(
      inputMetricsPaths = Seq("gdmix/0/per-job/metric/0", "gdmix/1/per-job/metric/0"),
      outputBestModelPath = "gdmix/best/model",
      evalMetric = "auc",
      hyperparameters = "eyIwIjogWyJnbG9iYWw6YmF0Y2hfc2l6",
      outputBestMetricsPath = Some("gdmix/best/metric"),
      inputModelPaths = Some("gdmix/0/per-job/model_output;gdmix/1/per-job/model_output"),
      copyBestOutput = true
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
    BestModelSelectorParser.parse(inCompleteArgs)
  }
}
