package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.IoUtils

/**
 * Parameters for data partition job.
 */
case class DataPartitionerParams(
  partitionEntity: String,
  inputMetadataFile: String,
  outputMetadataFile: String,
  trainInputDataPath: Option[String] = None,
  validationInputDataPath: Option[String] = None,
  trainInputScorePath: Option[String] = None,
  validationInputScorePath: Option[String] = None,
  trainOutputPartitionDataPath: Option[String] = None,
  validationOutputPartitionDataPath: Option[String] = None,
  trainPerCoordinateScorePath: Option[String] = None,
  validationPerCoordinateScorePath: Option[String] = None,
  outputPartitionListFile: Option[String] = None,
  numPartitions: Int = 10,
  dataFormat: String = TFRECORD,
  predictionScore: String = PREDICTION_SCORE,
  predictionScorePerCoordinate: String = PREDICTION_SCORE_PER_COORDINATE,
  offset: String = OFFSET,
  uid: String = UID,
  maxNumOfSamplesPerModel: Option[Int] = None,
  minNumOfSamplesPerModel: Option[Int] = None
)

/**
 * Parser for data partition job.
 */
object DataPartitionerParser {
  private val dataPartitionerParser = new scopt.OptionParser[DataPartitionerParams](
    "Parsing command line for data partitioner job.") {

    opt[String]("partitionEntity").action((x, p) => p.copy(partitionEntity = x.trim))
      .required
      .text(
        """Required.
          |The entity name used to partition.""".stripMargin)

    opt[String]("inputMetadataFile").action((x, p) => p.copy(inputMetadataFile = x.trim))
      .required
      .text(
        """Required.
          |Input metadata used for random effect data processing.""".stripMargin)

    opt[String]("outputMetadataFile").action((x, p) => p.copy(outputMetadataFile = x.trim))
      .required
      .text(
        """Required.
          |Output metadata file matches processed dataset.""".stripMargin)

    opt[String]("trainInputDataPath").action((x, p) => p.copy(trainInputDataPath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Training input dataset path.""".stripMargin)

    opt[String]("validationInputDataPath").action((x, p) => p.copy(validationInputDataPath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input dataset path.""".stripMargin)

    opt[String]("trainInputScorePath").action((x, p) => p.copy(trainInputScorePath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Training input score path.""".stripMargin)

    opt[String]("validationInputScorePath").action((x, p) => p.copy(validationInputScorePath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input score path.""".stripMargin)

    opt[String]("trainOutputPartitionDataPath").action((x, p) => p.copy(trainOutputPartitionDataPath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Partition result path for training data.""".stripMargin)

    opt[String]("validationOutputPartitionDataPath").action((x, p) => p.copy(validationOutputPartitionDataPath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Partition result path for validation data.""".stripMargin)

    opt[String]("trainPerCoordinateScorePath").action((x, p) => p.copy(trainPerCoordinateScorePath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Path to the per-coordinate training score of the previous iteration.""".stripMargin)

    opt[String]("validationPerCoordinateScorePath").action((x, p) => p.copy(validationPerCoordinateScorePath = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Path to the per-coordinate validation score of the previous iteration.""".stripMargin)

    opt[String]("outputPartitionListFile").action((x, p) => p.copy(outputPartitionListFile = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Output partition list file.""".stripMargin)

    opt[Int]("numPartitions").action((x, p) => p.copy(numPartitions = x))
      .optional
      .validate(
        x => if (x > 0) success else failure("Option --numPartitions must be > 0"))
      .text(
        """Optional.
          |Number of partitions.""".stripMargin)

    opt[String]("dataFormat").action((x, p) => p.copy(dataFormat = x.trim))
      .optional
      .text(
        """Optional.
          |Data format, either avro or tfrecord.""".stripMargin)

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

    opt[Int]("maxNumOfSamplesPerModel").action((x, p) => p.copy(maxNumOfSamplesPerModel = Some(x)))
      .optional
      .validate(
        x => if (x > 0) success else failure("Option --maxNumOfSamplesPerModel must be > 0"))
      .text(
        """Optional.
          |Maximal number of samples a model can take.""".stripMargin)

    opt[Int]("minNumOfSamplesPerModel").action((x, p) => p.copy(minNumOfSamplesPerModel = Some(x)))
      .optional
      .validate(
        x => if (x > 0) success else failure("Option --minNumOfSamplesPerModel must be > 0"))
      .text(
        """Optional.
          |Minimal number of samples needed to train a model.""".stripMargin)

    checkConfig(p =>
      if (!p.maxNumOfSamplesPerModel.isEmpty && !p.minNumOfSamplesPerModel.isEmpty &&
        (p.maxNumOfSamplesPerModel.get < p.minNumOfSamplesPerModel.get)) {
        failure("Invalid max/min number of samples per model, require max >= min")
      }
      else success)

    checkConfig(p =>
      if (IoUtils.isEmptyStr(p.trainInputDataPath) && IoUtils.isEmptyStr(p.validationInputDataPath)) {
        failure("Neither training nor validation data path is provided.")
      }
      else success)

    checkConfig(p =>
      if (!IoUtils.isEmptyStr(p.trainInputDataPath)) {
        if (IoUtils.isEmptyStr(p.trainOutputPartitionDataPath)) {
          failure("Option --trainOutputPartitionDataPath is required when --trainInputDataPath is not empty.")
        }
        else if (IoUtils.isEmptyStr(p.outputPartitionListFile)) {
          failure("Option --outputPartitionListFile is required when --trainInputDataPath is not empty.")
        }
        else success
      }
      else success)

    checkConfig(p =>
      if (!IoUtils.isEmptyStr(p.validationInputDataPath)) {
        if (IoUtils.isEmptyStr(p.validationOutputPartitionDataPath)) {
          failure("Option --validationOutputPartitionDataPath is required when --validationInputDataPath is not empty.")
        }
        else success
      }
      else success)
  }

  def parse(args: Seq[String]): DataPartitionerParams = {
    val emptyDataPartitionerParams = DataPartitionerParams(
      partitionEntity = "",
      inputMetadataFile = "",
      outputMetadataFile = "",
      trainInputDataPath = Some(""),
      trainOutputPartitionDataPath = Some(""),
      outputPartitionListFile = Some("")
    )
    dataPartitionerParser.parse(args, emptyDataPartitionerParams) match {
      case Some(params) => params
      case None => throw new IllegalArgumentException(
        s"Parsing the command line arguments failed.\n" +
          s"(${args.mkString(", ")}),\n${dataPartitionerParser.usage}")
    }
  }
}
