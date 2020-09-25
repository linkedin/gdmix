package com.linkedin.gdmix.parsers

import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert.assertEquals

/**
 * Unit tests for OffsetUpdaterParser.
 */
class OffsetUpdaterParserTest {

  @DataProvider
  def dataCompleteArgs(): Array[Array[Any]] = {
    Array(
      Array(
        Seq(
          "--trainInputDataPath", "fixed-effect/trainingData",
          "--trainInputScorePath", "jymbii_lr/per-job/trainingScore",
          "--trainPerCoordinateScorePath", "jymbii_lr/global/trainingScore",
          "--trainOutputDataPath", "jymbii_lr/global/updatedTrainingData",
          "--validationInputDataPath", "fixed-effect/validationData",
          "--validationInputScorePath", "jymbii_lr/per-job/validationScore",
          "--validationPerCoordinateScorePath", "jymbii_lr/global/validationScore",
          "--validationOutputDataPath", "jymbii_lr/global/updatedValidationData")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // miss trainInputDataPath
      Array(
        Seq(
          "--trainInputScorePath", "jymbii_lr/per-job/trainingScore",
          "--trainPerCoordinateScorePath", "jymbii_lr/global/trainingScore",
          "--trainOutputDataPath", "jymbii_lr/global/updatedTrainingData")),
      // miss trainInputScorePath
      Array(
        Seq(
          "--trainInputDataPath", "fixed-effect/trainingData",
          "--trainPerCoordinateScorePath", "jymbii_lr/global/trainingScore",
          "--trainOutputDataPath", "jymbii_lr/global/updatedTrainingData")),
      // miss trainPerCoordinateScorePath
      Array(
        Seq(
          "--trainInputDataPath", "fixed-effect/trainingData",
          "--trainInputScorePath", "jymbii_lr/per-job/trainingScore",
          "--trainOutputDataPath", "jymbii_lr/global/updatedTrainingData")),
      // miss trainOutputDataPath
      Array(
        Seq(
          "--trainInputDataPath", "fixed-effect/trainingData",
          "--trainInputScorePath", "jymbii_lr/per-job/trainingScore",
          "--trainPerCoordinateScorePath", "jymbii_lr/global/trainingScore"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = OffsetUpdaterParser.parse(completeArgs)
    val expectedParams = OffsetUpdaterParams(
      trainInputDataPath = "fixed-effect/trainingData",
      trainInputScorePath = "jymbii_lr/per-job/trainingScore",
      trainPerCoordinateScorePath = "jymbii_lr/global/trainingScore",
      trainOutputDataPath = "jymbii_lr/global/updatedTrainingData",
      validationInputDataPath = Some("fixed-effect/validationData"),
      validationInputScorePath = Some("jymbii_lr/per-job/validationScore"),
      validationPerCoordinateScorePath = Some("jymbii_lr/global/validationScore"),
      validationOutputDataPath = Some("jymbii_lr/global/updatedValidationData")
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
    def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
      OffsetUpdaterParser.parse(inCompleteArgs)
  }
}
