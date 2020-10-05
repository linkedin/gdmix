package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._

/**
 * Parameters for AUC compute job.
 */
case class AreaUnderROCCurveEvaluatorParams(
  inputPath: String,
  outputPath: String,
  labelName: String,
  scoreName: String
)

/**
 * Parser for AUC compute job.
 */
object AreaUnderROCCurveEvaluatorParser {
  private val areaUnderROCCurveEvaluatorParser = new scopt.OptionParser[AreaUnderROCCurveEvaluatorParams](
    "Parsing command line for auc compute job.") {

    opt[String]("inputPath").action((x, p) => p.copy(inputPath = x.trim))
      .required
      .text(
        """Required.
          |Input data path.""".stripMargin)

    opt[String]("outputPath").action((x, p) => p.copy(outputPath = x.trim))
      .required
      .text(
        """Required.
          |Output path for the computed AUC.""".stripMargin)

    opt[String]("labelName").action((x, p) => p.copy(labelName = x.trim))
      .required
      .text(
        """Required.
          |Label column name.""".stripMargin)

    opt[String]("scoreName").action((x, p) => p.copy(scoreName = x.trim))
      .required
      .text(
        """Required.
          |score column name.""".stripMargin)
  }

  def parse(args: Seq[String]): AreaUnderROCCurveEvaluatorParams = {
    val emptyAreaUnderROCCurveEvaluatorParams = AreaUnderROCCurveEvaluatorParams(
      inputPath = "",
      outputPath = "",
      labelName = "",
      scoreName = ""
    )
    areaUnderROCCurveEvaluatorParser.parse(args, emptyAreaUnderROCCurveEvaluatorParams) match {
      case Some(params) => params
      case None => throw new IllegalArgumentException(
        s"Parsing the command line arguments failed.\n" +
          s"(${args.mkString(", ")}),\n${areaUnderROCCurveEvaluatorParser.usage}")
    }
  }
}
