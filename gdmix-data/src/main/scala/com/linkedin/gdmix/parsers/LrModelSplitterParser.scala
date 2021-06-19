package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.IoUtils

/**
 * Parameters for LR model splitter job.
 */
case class LrModelSplitterParams(
  modelInputDir: String,
  modelOutputDir: String,
  numOutputFiles: Int = 200
)

/**
 * Parser for model splitting job.
 */
object LrModelSplitterParser {
  private val lrModelSplitterParser = new scopt.OptionParser[LrModelSplitterParams](
    "Parsing command line for model splitting job.") {

    opt[String]("modelInputDir").action((x, p) => p.copy(modelInputDir = x.trim))
      .required
      .text(
        """Required.
          |The path for input models.""".stripMargin)

    opt[String]("modelOutputDir").action((x, p) => p.copy(modelOutputDir = x.trim))
      .required
      .text(
        """Required.
          |The path for output models.""".stripMargin)

    opt[Int]("numOutputFiles").action((x, p) => p.copy(numOutputFiles = x))
      .optional
      .validate(
        x => if (x > 0) success else failure("Option --numPartitions must be > 0"))
      .text(
        """Optional.
          |Number of output files.""".stripMargin)

    checkConfig(p =>
      if (p.modelInputDir == "") {
        failure("Model input path can not be empty string.")
      }
      else success)

    checkConfig(p =>
      if (p.modelOutputDir == "") {
        failure("Model output path can not be empty string.")
      }
      else success)
  }

  def parse(args: Seq[String]): LrModelSplitterParams = {
    val emptyLrModelSplitterParams = LrModelSplitterParams(
      modelInputDir = "",
      modelOutputDir = ""
    )
    lrModelSplitterParser.parse(args, emptyLrModelSplitterParams) match {
      case Some(params) => params
      case None => throw new IllegalArgumentException(
        s"Parsing the command line arguments failed.\n" +
          s"(${args.mkString(", ")}),\n${lrModelSplitterParser.usage}")
    }
  }
}
