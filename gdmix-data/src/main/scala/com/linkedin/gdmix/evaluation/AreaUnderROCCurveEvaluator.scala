package com.linkedin.gdmix.evaluation

import com.databricks.spark.avro._
import org.apache.commons.cli.{BasicParser, CommandLine, CommandLineParser, Options}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.{IoUtils, JsonUtils}

/**
 * Evaluator for area under the ROC curve (AUROC).
 */
object AreaUnderROCCurveEvaluator {


  // Create a Spark session.
  val spark: SparkSession = SparkSession
    .builder()
    .appName(getClass.getName)
    .getOrCreate()

  // Set up Hadoop file system.
  val hadoopJobConf = new JobConf()
  val fs: FileSystem = FileSystem.get(hadoopJobConf)

  /**
   * Compute area under ROC curve.
   *
   * @param df Input data frame
   * @param labelName Name of the label in the dataframe
   * @param scoreName Name of the score in the dataframe
   * @return Area under ROC curve.
   */
  def calculateAreaUnderROCCurve(df: DataFrame, labelName: String, scoreName: String): Double = {
    // Cast the columns.
    val scoreLabelDF = df.withColumn(scoreName, col(scoreName).cast("double"))
      .withColumn(labelName, col(labelName).cast("double"))
      .select(scoreName, labelName)

    // Map to (score, label).
    val scoreAndLabels = scoreLabelDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))

    // Compute auc.
    new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
  }

  def main(args: Array[String]): Unit = {

    // Define options.
    val options = new Options()
    options.addOption("inputPath", true, "input score path")
    options.addOption("outputPath", true, "output path")
    options.addOption("labelName", true, "ground truth label name")
    options.addOption("scoreName", true, "predicted score name")

    // Get the parser.
    val parser: CommandLineParser = new BasicParser()
    val cmd: CommandLine = parser.parse(options, args)

    // Parse the commandline option.
    val inputPath = cmd.getOptionValue("inputPath")
    val outputPath = cmd.getOptionValue("outputPath")
    val labelName = cmd.getOptionValue("labelName")
    val scoreName = cmd.getOptionValue("scoreName")

    // Sanity check.
    require(inputPath != null
      && outputPath != null
      && labelName != null
      && scoreName != null,
      "Incorrect number of input parameters")

    // Read file and cast the label and score to double.
    val df = spark.read.avro(inputPath)

    // Compute auc.
    val auc = calculateAreaUnderROCCurve(df, labelName, scoreName)

    // Convert to json and save to HDFS.
    val jsonResult = JsonUtils.toJsonString(Map("auc" -> auc))
    IoUtils.writeFile(fs, new Path(outputPath, EVAL_SUMMARY_JSON), jsonResult)
  }
}