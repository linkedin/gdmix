package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._

/**
 * Parameters for offset update job.
 */
case class OffsetUpdaterParams(
  trainInputDataPath: String,
  trainInputScorePath: String,
  trainPerCoordinateScorePath: String,
  trainOutputDataPath: String,
  validationInputDataPath: Option[String] = None,
  validationInputScorePath: Option[String] = None,
  validationPerCoordinateScorePath: Option[String] = None,
  validationOutputDataPath: Option[String] = None,
  predictionScore: String = PREDICTION_SCORE,
  predictionScorePerCoordinate: String = PREDICTION_SCORE_PER_COORDINATE,
  dataFormat: String = AVRO,
  offset: String = OFFSET,
  uid: String = UID
)

/**
 * Parser for offset update job.
 */
object OffsetUpdaterParser {
  private val offsetUpdaterParser = new scopt.OptionParser[OffsetUpdaterParams](
    "Parsing command line for offset updater job.") {

    opt[String]("trainInputDataPath").action((x, p) => p.copy(trainInputDataPath = x.trim))
      .required
      .text(
        """Required.
          |Training input dataset path.""".stripMargin)

    opt[String]("trainInputScorePath").action((x, p) => p.copy(trainInputScorePath = x.trim))
      .required
      .text(
        """Required.
          |Training input score path.""".stripMargin)

    opt[String]("trainPerCoordinateScorePath").action((x, p) => p.copy(trainPerCoordinateScorePath = x.trim))
      .required
      .text(
        """Required.
          |Path to the per-coordinate training score of the previous iteration.""".stripMargin)

    opt[String]("trainOutputDataPath").action((x, p) => p.copy(trainOutputDataPath = x.trim))
      .required
      .text(
        """Required.
          |Output path for training data.""".stripMargin)

    opt[String]("validationInputDataPath").action((x, p) => p.copy(validationInputDataPath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input dataset path.""".stripMargin)


    opt[String]("validationInputScorePath").action((x, p) => p.copy(validationInputScorePath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input score path.""".stripMargin)

    opt[String]("validationPerCoordinateScorePath").action((x, p) => p.copy(validationPerCoordinateScorePath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Path to the per-coordinate validation score of the previous iteration.""".stripMargin)

    opt[String]("validationOutputDataPath").action((x, p) => p.copy(validationOutputDataPath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Output path for validation data.""".stripMargin)

    opt[String]("predictionScore").action((x, p) => p.copy(predictionScore = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of prediction score.""".stripMargin)

    opt[String]("predictionScorePerCoordinate").action((x, p) => p.copy(predictionScorePerCoordinate = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of prediction score per-coordinate.""".stripMargin)

    opt[String]("dataFormat").action((x, p) => p.copy(dataFormat = x.trim))
      .optional
      .text(
        """Optional.
          |Data format, either avro or tfrecord.""".stripMargin)

    opt[String]("offset").action((x, p) => p.copy(offset = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of offset.""".stripMargin)

    opt[String]("uid").action((x, p) => p.copy(uid = x.trim))
      .optional
      .text(
        """Optional.
          |Column name of unique id.""".stripMargin)

  }

  def parse(args: Seq[String]): OffsetUpdaterParams = {
    val emptyOffsetUpdaterParams = OffsetUpdaterParams(
      trainInputDataPath = "",
      trainInputScorePath = "",
      trainPerCoordinateScorePath = "",
      trainOutputDataPath = ""
    )
    offsetUpdaterParser.parse(args, emptyOffsetUpdaterParams) match {
      case Some(params) => params
      case None => throw new IllegalArgumentException(
        s"Parsing the command line arguments failed.\n" +
          s"(${args.mkString(", ")}),\n${offsetUpdaterParser.usage}")
    }
  }
}
