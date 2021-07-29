package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._

/**
 * Parameters for offset update job.
 */
case class OffsetUpdaterParams(
  trainingDataDir: String,
  trainingScoreDir: String,
  trainingScorePerCoordinateDir: String,
  outputTrainingDataDir: String,
  validationDataDir: Option[String] = None,
  validationScoreDir: Option[String] = None,
  validationScorePerCoordinateDir: Option[String] = None,
  outputValidationDataDir: Option[String] = None,
  predictionScoreColumnName: String = PREDICTION_SCORE,
  predictionScorePerCoordinateColumnName: String = PREDICTION_SCORE_PER_COORDINATE,
  dataFormat: String = AVRO,
  offsetColumnName: String = OFFSET,
  uidColumnName: String = UID,
  numPartitions: Int = 0
)

/**
 * Parser for offset update job.
 */
object OffsetUpdaterParser {
  private val offsetUpdaterParser = new scopt.OptionParser[OffsetUpdaterParams](
    "Parsing command line for offset updater job.") {

    opt[String]("trainingDataDir").action((x, p) => p.copy(trainingDataDir = x.trim))
      .required
      .text(
        """Required.
          |Training input dataset path.""".stripMargin)

    opt[String]("trainingScoreDir").action((x, p) => p.copy(trainingScoreDir = x.trim))
      .required
      .text(
        """Required.
          |Training input score path.""".stripMargin)

    opt[String]("trainingScorePerCoordinateDir").action((x, p) => p.copy(trainingScorePerCoordinateDir = x.trim))
      .required
      .text(
        """Required.
          |Path to the per-coordinate training score of the previous iteration.""".stripMargin)

    opt[String]("outputTrainingDataDir").action((x, p) => p.copy(outputTrainingDataDir = x.trim))
      .required
      .text(
        """Required.
          |Output path for training data.""".stripMargin)

    opt[String]("validationDataDir").action((x, p) => p.copy(validationDataDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input dataset path.""".stripMargin)


    opt[String]("validationScoreDir").action((x, p) => p.copy(validationScoreDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input score path.""".stripMargin)

    opt[String]("validationScorePerCoordinateDir").action((x, p) => p.copy(validationScorePerCoordinateDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Path to the per-coordinate validation score of the previous iteration.""".stripMargin)

    opt[String]("outputValidationDataDir").action((x, p) => p.copy(outputValidationDataDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Output path for validation data.""".stripMargin)

    opt[String]("predictionScoreColumnName").action((x, p) => p.copy(predictionScoreColumnName = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of prediction score.""".stripMargin)

    opt[String]("predictionScorePerCoordinateColumnName").action((x, p) => p.copy(predictionScorePerCoordinateColumnName = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of prediction score per-coordinate.""".stripMargin)

    opt[String]("dataFormat").action((x, p) => p.copy(dataFormat = x.trim))
      .optional
      .text(
        """Optional.
          |Data format, either avro or tfrecord.""".stripMargin)

    opt[String]("offsetColumnName").action((x, p) => p.copy(offsetColumnName = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of offset.""".stripMargin)

    opt[String]("uidColumnName").action((x, p) => p.copy(uidColumnName = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of unique id.""".stripMargin)

    opt[Int]("numPartitions").action((x, p) => p.copy(numPartitions = x))
      .optional
      .validate(
        x => if (x >= 0) success else failure("Option --numPartitions must be >= 0"))
      .text(
        """Optional.
          |Number of partitions.""".stripMargin)

  }

  def parse(args: Seq[String]): OffsetUpdaterParams = {
    val emptyOffsetUpdaterParams = OffsetUpdaterParams(
      trainingDataDir = "",
      trainingScoreDir = "",
      trainingScorePerCoordinateDir = "",
      outputTrainingDataDir = ""
    )
    offsetUpdaterParser.parse(args, emptyOffsetUpdaterParams) match {
      case Some(params) => params
      case None => throw new IllegalArgumentException(
        s"Parsing the command line arguments failed.\n" +
          s"(${args.mkString(", ")}),\n${offsetUpdaterParser.usage}")
    }
  }
}
