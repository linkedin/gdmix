package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._

/**
 * Parameters for AUC compute job.
 */
case class AreaUnderROCCurveEvaluatorParams(
  metricsInputDir: String,
  outputMetricFile: String,
  labelColumnName: String,
  predictionColumnName: String
)

/**
 * Parser for AUC compute job.
 */
object AreaUnderROCCurveEvaluatorParser {
  private val areaUnderROCCurveEvaluatorParser = new scopt.OptionParser[AreaUnderROCCurveEvaluatorParams](
    "Parsing command line for auc compute job.") {

    opt[String]("metricsInputDir").action((x, p) => p.copy(metricsInputDir = x.trim))
      .required
      .text(
        """Required.
          |Input data path containing prediction and label column.""".stripMargin)

    opt[String]("outputMetricFile").action((x, p) => p.copy(outputMetricFile = x.trim))
      .required
      .text(
        """Required.
          |Output file for the computed AUC.""".stripMargin)

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
  }

  def parse(args: Seq[String]): AreaUnderROCCurveEvaluatorParams = {
    val emptyAreaUnderROCCurveEvaluatorParams = AreaUnderROCCurveEvaluatorParams(
      metricsInputDir = "",
      outputMetricFile = "",
      labelColumnName = "",
      predictionColumnName = ""
    )
    areaUnderROCCurveEvaluatorParser.parse(args, emptyAreaUnderROCCurveEvaluatorParams) match {
      case Some(params) => params
      case None => throw new IllegalArgumentException(
        s"Parsing the command line arguments failed.\n" +
          s"(${args.mkString(", ")}),\n${areaUnderROCCurveEvaluatorParser.usage}")
    }
  }
}
