package com.linkedin.gdmix.data

import java.nio.charset.StandardCharsets
import java.util.Base64

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.sql.SparkSession

import com.linkedin.gdmix.parsers.BestModelSelectorParser
import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.{IoUtils, JsonUtils}

/**
 * Object to select the best model from hyperparameter tuning.
 */
object BestModelSelector {

  def main(args: Array[String]): Unit = {

    val params = BestModelSelectorParser.parse(args)

    // Read the input parameters.
    // Update input parameters to match with the cloudflow implementation.
    // The paths `inputMetricsPaths` and `inputModelPaths` are joined together separated by semicolon
    // i.e. path0;path1;path2;path3 with respect to model index from 0 to 3.
    val inputMetricsPaths = params.inputMetricsPaths
    val evalMetric = params.evalMetric
    val outputBestModelPath = params.outputBestModelPath
    val inputModelPaths = params.inputModelPaths.getOrElse("").split(CONFIG_SPLITTER).map(_.trim)
    val outputBestMetricsPath = params.outputBestMetricsPath
    // hyperparameters is a encoded base64 string to avoid the parsing error in spark due to nested
    // fields coming from PDLs.
    val hyperparameters = params.hyperparameters
    val copyBestOutput = params.copyBestOutput

    // Create a Spark session.
    val spark = SparkSession.builder().appName(getClass.getName).getOrCreate()

    val hadoopJobConf = new JobConf()
    val fs = FileSystem.get(hadoopJobConf)
    val hparamMap = deserialize(hyperparameters)

    val modelSize = inputMetricsPaths.size
    require(modelSize == hparamMap.size, s"hyperparameters does not have desired $modelSize values.")
    if (copyBestOutput) {
      require(modelSize == inputModelPaths.size, s"inputModelPaths does not have desired $modelSize paths.")
    }

    var maxMetric: Float = -1.0F
    var bestModelId: Int = -1

    // Indicate whether optimization is maximization (1) or minimization (-1).
    var optDirection = 1

    evalMetric match {
      case AUC => optDirection = 1
      case RMSE => optDirection = -1
      case _ => new IllegalArgumentException(s"Evaluation metric $evalMetric is not defined")
    }

    inputMetricsPaths.zipWithIndex.foreach { case (metricPath, modelId) =>
      val jsonString = IoUtils.readFile(fs, s"$metricPath/$EVAL_SUMMARY_JSON")
      val metric = getMetric(jsonString, evalMetric)
      if (metric * optDirection > maxMetric * optDirection) {
        maxMetric = metric
        bestModelId = modelId
      }
    }

    val bestModelParam = JsonUtils.toJsonString(hparamMap(s"$bestModelId"))
    val configs = Map("best model index" -> bestModelId, "model params" -> bestModelParam)

    // Output hyperparameters and evaluation metrics
    val configsJsonString = JsonUtils.toJsonString(configs)
    val outputJsonFile = s"$outputBestModelPath/evals.json"
    IoUtils.writeFile(fs, new Path(outputJsonFile), configsJsonString)


    if (copyBestOutput) {
      // Copy the best metrics log.
      val srcMetric = inputMetricsPaths(bestModelId)
      IoUtils.copyDirectory(fs, hadoopJobConf, srcMetric, outputBestMetricsPath.get)

      // Copy the best model.
      val srcModel = inputModelPaths(bestModelId)
      IoUtils.copyDirectory(fs, hadoopJobConf, srcModel, outputBestModelPath)
    }

    // Terminate Spark session.
    spark.stop()
  }

  /**
   * Get map from the encoded base64 hyperparameter
   *
   * @param hparams Encoded base64 hyperparameter string
   * @return A map with index of model id, value of each hyper-parameter of the model
   */
  def deserialize(hparams: String): Map[String, Any] = {
    val decodedBytes = Base64.getDecoder.decode(hparams)
    val decodedString = new String(decodedBytes, StandardCharsets.UTF_8)
    JsonUtils.toMap[Any](decodedString)
  }

  /**
   * Get evaluation metric from a Json String
   *
   * @param evalSummary A Json String containing the evaluation metrics
   * @param evalMetric Metric to be chosen
   * @return Metric value
   */
  def getMetric(evalSummary: String, evalMetric:String): Float = {

    val metricsMap: Map[String, Float] = JsonUtils.toMap[Float](evalSummary)
    val metricOption = metricsMap.get(evalMetric)
    val metric = metricOption.getOrElse(throw new IllegalArgumentException(s"Couldn't find metric: $evalMetric"))

    metric
  }
}
