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
          "--inputPath", "global/validationScore",
          "--outputPath", "global/metric/0",
          "--labelName", "response",
          "--scoreName", "predictionScore")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // miss inputPath
      Array(
        Seq(
          "--outputPath", "global/metric/0",
          "--labelName", "response",
          "--scoreName", "predictionScore")),
      // miss outputPath
      Array(
        Seq(
          "--inputPath", "global/validationScore",
          "--labelName", "response",
          "--scoreName", "predictionScore")),
      // miss labelName
      Array(
        Seq(
          "--inputPath", "global/validationScore",
          "--outputPath", "global/metric/0",
          "--scoreName", "predictionScore")),
      // miss scoreName
      Array(
        Seq(
          "--inputPath", "global/validationScore",
          "--outputPath", "global/metric/0",
          "--labelName", "response"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = AreaUnderROCCurveEvaluatorParser.parse(completeArgs)
    val expectedParams = AreaUnderROCCurveEvaluatorParams(
      inputPath = "global/validationScore",
      outputPath = "global/metric/0",
      labelName = "response",
      scoreName = "predictionScore"
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
    def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
      AreaUnderROCCurveEvaluatorParser.parse(inCompleteArgs)
  }
}
