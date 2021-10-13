package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._

/**
 * Parameters for evaluation metric compute job.
 */
case class EvaluatorParams(
  metricsInputDir: String,
  outputMetricFile: String,
  labelColumnName: String,
  predictionColumnName: String,
  metricName: String
)

/**
 * Parser for evaluation metric compute job.
 */
object EvaluatorParser {
  private val evaluatorParser = new scopt.OptionParser[EvaluatorParams](
    "Parsing command line for evaluation metric compute job.") {

    opt[String]("metricsInputDir").action((x, p) => p.copy(metricsInputDir = x.trim))
      .required
      .text(
        """Required.
          |Input data path containing prediction and label column.""".stripMargin)

    opt[String]("outputMetricFile").action((x, p) => p.copy(outputMetricFile = x.trim))
      .required
      .text(
        """Required.
          |Output file for the computed evaluation metric.""".stripMargin)

    opt[String]("labelColumnName").action((x, p) => p.copy(labelColumnName = x.trim))
      .required
      .text(
        """Required.
          |Label column name.""".stripMargin)

    opt[String]("predictionColumnName").action((x, p) => p.copy(predictionColumnName = x.trim))
      .required
      .text(
        """Required.
          |prediction score column name.""".stripMargin)

    opt[String]("metricName").action((x, p) => p.copy(metricName = x.trim))
      .required
      .text(
        """Required.
          |evaluation metric name (current only support 'auc' and 'mse').""".stripMargin)

    checkConfig(p =>
      if (!List(AUC, MSE).contains(p.metricName)) {
        failure(s"${p.metricName} is not supported, should be in ['auc', 'mse'].")
      }
      else success)
  }

  def parse(args: Seq[String]): EvaluatorParams = {
    val emptyEvaluatorParams = EvaluatorParams(
      metricsInputDir = "",
      outputMetricFile = "",
      labelColumnName = "",
      predictionColumnName = "",
      metricName = ""
    )
    evaluatorParser.parse(args, emptyEvaluatorParams) match {
      case Some(params) => params
      case None => throw new IllegalArgumentException(
        s"Parsing the command line arguments failed.\n" +
          s"(${args.mkString(", ")}),\n${evaluatorParser.usage}")
    }
  }
}
