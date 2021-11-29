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
          "--trainingDataDir", "fixed-effect/trainingData",
          "--trainingScoreDir", "jymbii_lr/per-job/trainingScore",
          "--trainingScorePerCoordinateDir", "jymbii_lr/global/trainingScore",
          "--outputTrainingDataDir", "jymbii_lr/global/updatedTrainingData",
          "--validationDataDir", "fixed-effect/validationData",
          "--validationScoreDir", "jymbii_lr/per-job/validationScore",
          "--validationScorePerCoordinateDir", "jymbii_lr/global/validationScore",
          "--outputValidationDataDir", "jymbii_lr/global/updatedValidationData",
          "--numPartitions", "10")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // miss trainingDataDir
      Array(
        Seq(
          "--trainingScoreDir", "jymbii_lr/per-job/trainingScore",
          "--trainingScorePerCoordinateDir", "jymbii_lr/global/trainingScore",
          "--outputTrainingDataDir", "jymbii_lr/global/updatedTrainingData")),
      // miss trainingScoreDir
      Array(
        Seq(
          "--trainingDataDir", "fixed-effect/trainingData",
          "--trainingScorePerCoordinateDir", "jymbii_lr/global/trainingScore",
          "--outputTrainingDataDir", "jymbii_lr/global/updatedTrainingData")),
      // miss outputTrainingDataDir
      Array(
        Seq(
          "--trainingDataDir", "fixed-effect/trainingData",
          "--trainingScoreDir", "jymbii_lr/per-job/trainingScore",
          "--trainingScorePerCoordinateDir", "jymbii_lr/global/trainingScore"))
    )
  }

  @DataProvider
  def dataIncorrectArgs(): Array[Array[Any]] = {

    Array(
      // negative numPartitions
      Array(
        Seq(
          "--trainingDataDir", "fixed-effect/trainingData",
          "--trainingScoreDir", "jymbii_lr/per-job/trainingScore",
          "--trainingScorePerCoordinateDir", "jymbii_lr/global/trainingScore",
          "--outputTrainingDataDir", "jymbii_lr/global/updatedTrainingData",
          "--numPartitions", "-10"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = OffsetUpdaterParser.parse(completeArgs)
    val expectedParams = OffsetUpdaterParams(
      trainingDataDir = "fixed-effect/trainingData",
      trainingScoreDir = "jymbii_lr/per-job/trainingScore",
      trainingScorePerCoordinateDir = Some("jymbii_lr/global/trainingScore"),
      outputTrainingDataDir = "jymbii_lr/global/updatedTrainingData",
      validationDataDir = Some("fixed-effect/validationData"),
      validationScoreDir = Some("jymbii_lr/per-job/validationScore"),
      validationScorePerCoordinateDir = Some("jymbii_lr/global/validationScore"),
      outputValidationDataDir = Some("jymbii_lr/global/updatedValidationData"),
      numPartitions = 10
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
    OffsetUpdaterParser.parse(inCompleteArgs)
  }

  @Test(dataProvider = "dataIncorrectArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testIncorrectArgs(inCorrectArgs: Seq[String]): Unit = {
    OffsetUpdaterParser.parse(inCorrectArgs)
  }
}
