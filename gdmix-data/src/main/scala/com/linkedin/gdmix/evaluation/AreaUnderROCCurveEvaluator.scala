package com.linkedin.gdmix.evaluation

import com.databricks.spark.avro._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

import com.linkedin.gdmix.parsers.AreaUnderROCCurveEvaluatorParser
import com.linkedin.gdmix.parsers.AreaUnderROCCurveEvaluatorParams
import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.{IoUtils, JsonUtils}


/**
 * Evaluator for area under the ROC curve (AUROC).
 */
object AreaUnderROCCurveEvaluator {
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

    val params = AreaUnderROCCurveEvaluatorParser.parse(args)

    // Create a Spark session.
    val spark: SparkSession = SparkSession
      .builder()
      .appName(getClass.getName)
      .getOrCreate()

    try {
      run(spark, params)
    } finally {
      spark.stop()
    }
  }

  def run(spark: SparkSession, params: AreaUnderROCCurveEvaluatorParams): Unit = {

    // Read file and cast the label and score to double.
    val df = spark.read.avro(params.metricsInputDir)

    // Compute auc.
    val auc = calculateAreaUnderROCCurve(df, params.labelColumnName, params.predictionColumnName)

    // Set up Hadoop file system.
    val hadoopJobConf = new JobConf()
    val fs: FileSystem = FileSystem.get(hadoopJobConf)

    // Convert to json and save to HDFS.
    val jsonResult = JsonUtils.toJsonString(Map("auc" -> auc))
    IoUtils.writeFile(fs, new Path(params.outputMetricFile, EVAL_SUMMARY_JSON), jsonResult)
  }
}