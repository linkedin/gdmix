package com.linkedin.gdmix.data

import scala.collection.mutable

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.sql.SparkSession

import com.linkedin.gdmix.utils.{IoUtils, JsonUtils}

/**
 * Object to select the best model from hyperparameter tuning
 */
object ModelSelector {

  def main(args: Array[String]): Unit = {

    // Read the input parameters.
    val inputMetricsPath = args(0)
    val inputModelPath = args(1)
    val inputScorePath = args(2)
    val outputBestMetricsPath = args(3)
    val outputBestModelPath = args(4)
    val outputBestScorePath = args(5)
    val evalMetric = args(6)
    val learningRatesString = args(7)
    val l2RegWeightsString = args(8)
    val copyBestOutput = args(9).toBoolean

    require(args.length == 10, "Incorrect number of input parameters")

    // Create a Spark session.
    val spark = SparkSession
      .builder()
      .appName(getClass.getName)
      .getOrCreate()

    val hadoopJobConf = new JobConf()
    val fs = FileSystem.get(hadoopJobConf)

    val learningRates = learningRatesString.split(",")
    val l2RegWeights = l2RegWeightsString.split(",")

    var maxMetric: Float = -1.0F
    var bestModelId: Int = -1
    var modelId: Int = 0

    // Indicate whether optimization is maximization (1) or minimization (-1).
    var optDirection = 1
    val configs: mutable.ListBuffer[Map[String, Any]] = new mutable.ListBuffer[Map[String, Any]]

    evalMetric match {
      case "auc" => optDirection = 1
      case "rmse" => optDirection = -1
      case _ => new IllegalArgumentException(s"Evaluation metric $evalMetric is not defined")
    }
    // Loops for reading the evaluation metrics and selecting the best model.
    for (learningRate <- learningRates) {
      for (l2RegWeight <- l2RegWeights) {
        val jsonString = IoUtils.readFile(fs, inputMetricsPath + s"/$modelId/evalSummary.json")
        val metricsMap: Map[String, Float] = JsonUtils.toMap[Float](jsonString)
        val metricOption = metricsMap.get(evalMetric)
        val metric = metricOption.getOrElse(throw new IllegalArgumentException(s"Couldn't find metric: $evalMetric"))

        if (metric * optDirection > maxMetric * optDirection) {
          maxMetric = metric
          bestModelId = modelId
        }

        // Put hypeparemeters into the recordMap
        val recordMap: Map[String, Any] = Map(
          "model index" -> modelId,
          "learning rate" -> learningRate,
          "l2 regularization weight" -> l2RegWeight,
          evalMetric -> metric)

        // Update configurations List
        configs += recordMap

        modelId = modelId + 1
      }
    }
    configs += Map("best model index" -> bestModelId)

    // Output hyperparameters and evaluation metrics
    val configsJsonString = JsonUtils.toJsonString(configs)
    val outputJsonFile = inputMetricsPath + "/evals.json"
    IoUtils.writeFile(fs, new Path(outputJsonFile), configsJsonString)

    if (copyBestOutput) {
      // Copy the best metrics log.
      val srcMetric = s"$inputMetricsPath/$bestModelId"
      IoUtils.copyDirectory(fs, hadoopJobConf, srcMetric, outputBestMetricsPath)

      // Copy the best model.
      val srcModel = s"$inputModelPath/$bestModelId"
      IoUtils.copyDirectory(fs, hadoopJobConf, srcModel, outputBestModelPath)

      // Copy the best score.
      val srcScore = s"$inputScorePath/$bestModelId"
      IoUtils.copyDirectory(fs, hadoopJobConf, srcScore, outputBestScorePath)
    }

    // Terminate Spark session.
    spark.stop()
  }
}
