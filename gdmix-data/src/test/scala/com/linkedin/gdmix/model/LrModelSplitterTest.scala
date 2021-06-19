package com.linkedin.gdmix.model

import com.databricks.spark.avro._

import com.linkedin.gdmix.parsers.LrModelSplitterParser
import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.ConversionUtils.NameTermValueOptionDouble
import com.linkedin.gdmix.utils.{SharedSparkSession, TestUtils}

import org.apache.commons.io.FileUtils
import org.apache.spark.sql.{DataFrame, Row, SaveMode}
import org.apache.spark.sql.functions.typedLit
import org.apache.spark.sql.types._
import org.testng.Assert.assertTrue
import org.testng.annotations.{AfterTest, BeforeTest, Test}

import java.nio.file.{Files, Paths}

/**
 * Unit tests for [[LrModelSplitter]].
 */
class LrModelSplitterTest extends SharedSparkSession {


  final val TEST_DIR = Files.createTempDirectory("TestInput")
  final val INPUT_DIR = Files.createTempDirectory(TEST_DIR, "TestInput").toString
  final val OUTPUT_DIR = Files.createTempDirectory(TEST_DIR, "TestOutput").toString
  final val NUM_OUTPUT_FILES = 1

  import spark.implicits._

  @BeforeTest
  def beforeTest() {
    FileUtils.deleteDirectory(TEST_DIR.toFile)
  }

  @AfterTest
  def afterTest() {
    FileUtils.deleteDirectory(TEST_DIR.toFile)
  }


  /**
   * Unit test for [[splitModelId]].
   */
  @Test
  def testSplitModelId(): Unit = {
    // create input df
    val inputData = Seq(
      Row(Array(Row(s"m1${CROSS}f1", "t1", 0.4), Row(s"m1${CROSS}f3", "", 1.4),
        Row(s"m2${CROSS}f7", "t1", 0.5), Row(s"m2${CROSS}f9", "", -0.4)))
    )
    val inputSchema = StructType(List(StructField(MEANS, ArrayType(
      StructType(List(
        StructField(NAME, StringType, true),
        StructField(TERM, StringType, true),
        StructField(VALUE, DoubleType, true)
      )), true), true)))
    val inputDf = spark.createDataFrame(spark.sparkContext.parallelize(inputData), inputSchema)

    val actualDf = LrModelSplitter.splitModelId(MEANS, inputDf)

    val expectedData = Seq(
      Row("m1", Array(Row("f1", "t1", 0.4), Row("f3", "", 1.4))),
      Row("m2", Array(Row("f7", "t1", 0.5), Row("f9", "", -0.4)))
    )
    val expectedSchema = StructType(List(StructField(MODEL_ID, StringType, true),
      StructField(MEANS, ArrayType(StructType(List(StructField(NAME, StringType, true),
        StructField(TERM, StringType, true), StructField(VALUE, DoubleType, true))), true), true)))
    val expectedDf = spark.createDataFrame(spark.sparkContext.parallelize(expectedData), expectedSchema)
    TestUtils.equalSmallDataFrame(actualDf, expectedDf, MODEL_ID)
  }

  /**
   * Unit test for [[run]].
   * The input dataset has means only
   */
  @Test
  def testRunMeansOnly(): Unit = {
    // Parse params
    val inputDir = s"$INPUT_DIR/means_only"
    val outputDir = s"$OUTPUT_DIR/means_only"
    val cmdLine = Array(
      "--modelInputDir", inputDir,
      "--modelOutputDir", outputDir,
      "--numOutputFiles", "1")
    val params = LrModelSplitterParser.parse(cmdLine)
    // create input dataset
    val inputDf = Seq(
      (Seq(NameTermValueOptionDouble(s"m1${CROSS}f1", "", Some(0.3)),
        NameTermValueOptionDouble(s"m1${CROSS}f2", "t2", Some(-0.3))),
        LR_MODEL_CLASS, ""),
      (Seq(NameTermValueOptionDouble(s"m2${CROSS}f3", "t5", Some(2.1)),
        NameTermValueOptionDouble(s"m2${CROSS}f4", "t6", Some(4.5))),
        LR_MODEL_CLASS, ""))
      .toDF(MEANS, "modelClass", "lossFunction")
      .withColumn(VARIANCES, typedLit[Option[NameTermValueOptionDouble]](None))
    inputDf.write.mode(SaveMode.Overwrite).format(AVRO_FORMAT).save(inputDir)

    // run the test
    LrModelSplitter.run(spark, params)
    val actualDf = spark.read.avro(outputDir)

    // set expectation
    val expectedDf = Seq(
      (Seq(NameTermValueOptionDouble("f1", "", Some(0.3)),
        NameTermValueOptionDouble("f2", "t2", Some(-0.3))),
        LR_MODEL_CLASS, "", "m1"),
      (Seq(NameTermValueOptionDouble("f3", "t5", Some(2.1)),
        NameTermValueOptionDouble("f4", "t6", Some(4.5))),
        LR_MODEL_CLASS, "", "m2"))
      .toDF(MEANS, "modelClass", "lossFunction", MODEL_ID)
      .withColumn(VARIANCES, typedLit[Option[NameTermValueOptionDouble]](None))
    assertTrue(TestUtils.equalSmallDataFrame(actualDf, expectedDf, MODEL_ID))
  }

  /**
   * Unit test for [[run]].
   * The input dataset has means and variances
   */
  @Test
  def testRunMeansAndVariance(): Unit = {
    // Parse params
    val inputDir = s"$INPUT_DIR/means_variances"
    val outputDir = s"$OUTPUT_DIR/means_variances"
    val cmdLine = Array(
      "--modelInputDir", inputDir,
      "--modelOutputDir", outputDir,
      "--numOutputFiles", "1")
    val params = LrModelSplitterParser.parse(cmdLine)
    // create input dataset
    val inputDf = Seq(
      (Seq(NameTermValueOptionDouble(s"m1${CROSS}f1", "", Some(0.3)),
        NameTermValueOptionDouble(s"m1${CROSS}f2", "t2", Some(-0.3))),
        Seq(NameTermValueOptionDouble(s"m1${CROSS}f1", "", Some(1.0)),
          NameTermValueOptionDouble(s"m1${CROSS}f2", "t2", Some(2.1))),
        LR_MODEL_CLASS, ""),
      (Seq(NameTermValueOptionDouble(s"m2${CROSS}f3", "t5", Some(2.1)),
        NameTermValueOptionDouble(s"m2${CROSS}f4", "t6", Some(4.5))),
        Seq(NameTermValueOptionDouble(s"m2${CROSS}f3", "t5", Some(0.5)),
          NameTermValueOptionDouble(s"m2${CROSS}f4", "t6", Some(0.7))),
        LR_MODEL_CLASS, ""))
      .toDF(MEANS, VARIANCES, "modelClass", "lossFunction")
    inputDf.write.mode(SaveMode.Overwrite).format(AVRO_FORMAT).save(inputDir)

    // run the test
    LrModelSplitter.run(spark, params)
    val actualDf = spark.read.avro(outputDir)

    // set expectation
    val expectedDf = Seq(
      (Seq(NameTermValueOptionDouble("f1", "", Some(0.3)),
        NameTermValueOptionDouble("f2", "t2", Some(-0.3))),
        Seq(NameTermValueOptionDouble("f1", "", Some(1.0)),
          NameTermValueOptionDouble("f2", "t2", Some(2.1))),
        LR_MODEL_CLASS, "", "m1"),
      (Seq(NameTermValueOptionDouble("f3", "t5", Some(2.1)),
        NameTermValueOptionDouble("f4", "t6", Some(4.5))),
        Seq(NameTermValueOptionDouble("f3", "t5", Some(0.5)),
          NameTermValueOptionDouble("f4", "t6", Some(0.7))),
        LR_MODEL_CLASS, "", "m2"))
      .toDF(MEANS, VARIANCES, "modelClass", "lossFunction", MODEL_ID)
    assertTrue(TestUtils.equalSmallDataFrame(actualDf, expectedDf, MODEL_ID))
  }
}