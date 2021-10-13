package com.linkedin.gdmix.evaluation

import com.databricks.spark.avro._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.mllib.evaluation.{BinaryClassificationMetrics, RegressionMetrics}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

import com.linkedin.gdmix.parsers.EvaluatorParams
import com.linkedin.gdmix.parsers.EvaluatorParser
import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.{IoUtils, JsonUtils}


/**
 * Metric evaluator.
 */
object Evaluator {
  /**
   * Compute evaluation metric based on the metric name.
   *
   * @param df         Input data frame
   * @param labelName  Name of the label in the dataframe
   * @param scoreName  Name of the score in the dataframe
   * @param metricName Name of the evaluation metric
   * @return evaluation metric (e.g, area under ROC curve, mean squared error, etc.)
   */
  def calculateMetric(df: DataFrame, labelName: String, scoreName: String, metricName: String): Double = {
    // Cast the columns.
    val scoreLabelDF = df.withColumn(scoreName, col(scoreName).cast("double"))
      .withColumn(labelName, col(labelName).cast("double"))
      .select(scoreName, labelName)

    // Map to (score, label).
    val scoreAndLabels = scoreLabelDF.rdd.map(row => (row.getDouble(0), row.getDouble(1)))

    // Compute evaluation metric.
    val metric = metricName match {
      case AUC => new BinaryClassificationMetrics(scoreAndLabels).areaUnderROC()
      case MSE => new RegressionMetrics(scoreAndLabels).meanSquaredError
      case _ => throw new IllegalArgumentException(s"Do not support metric ${metricName}, currently only support 'auc' and 'mse'.")
    }
    metric
  }

  def main(args: Array[String]): Unit = {

    val params = EvaluatorParser.parse(args)

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

  def run(spark: SparkSession, params: EvaluatorParams): Unit = {

    // Read file and cast the label and score to double.
    val df = spark.read.avro(params.metricsInputDir)

    // Compute evaluation metric.
    val metric = calculateMetric(df, params.labelColumnName, params.predictionColumnName, params.metricName)

    // Set up Hadoop file system.
    val hadoopJobConf = new JobConf()
    val fs: FileSystem = FileSystem.get(hadoopJobConf)

    // Convert to json and save to HDFS.
    val jsonResult = JsonUtils.toJsonString(Map(params.metricName -> metric))
    IoUtils.writeFile(fs, new Path(params.outputMetricFile, EVAL_SUMMARY_JSON), jsonResult)
  }
}