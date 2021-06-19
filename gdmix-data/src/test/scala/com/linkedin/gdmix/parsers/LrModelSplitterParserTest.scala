package com.linkedin.gdmix.parsers

import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert.assertEquals

/**
 * Unit tests for LrModelSplitterParserTest.
 */
class LrModelSplitterParserTest {

  @DataProvider
  def dataCompleteArgs(): Array[Array[Any]] = {
    Array(
      Array(
        Seq(
          "--modelInputDir", "global/input",
          "--modelOutputDir", "global/output",
          "--numOutputFiles", "100")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // missing input path
      Array(
        Seq(
          "--modelOutputDir", "global/output",
          "--numOutputFiles", "100")),
      // missing output path
      Array(
        Seq(
          "--modelInputDir", "global/input",
          "--numOutputFiles", "100"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = LrModelSplitterParser.parse(completeArgs)
    val expectedParams = LrModelSplitterParams(
      modelInputDir = "global/input",
      modelOutputDir = "global/output",
      numOutputFiles = 100
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
    LrModelSplitterParser.parse(inCompleteArgs)
  }
}
