package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.IoUtils

/**
 * Parameters for data partition job.
 */
case class DataPartitionerParams(
  partitionId: String,
  metadataFile: String,
  outputMetadataFile: String,
  trainingDataDir: Option[String] = None,
  validationDataDir: Option[String] = None,
  trainingScoreDir: Option[String] = None,
  validationScoreDir: Option[String] = None,
  partitionedTrainingDataDir: Option[String] = None,
  partitionedValidationDataDir: Option[String] = None,
  trainingScorePerCoordinateDir: Option[String] = None,
  validationScorePerCoordinateDir: Option[String] = None,
  outputPartitionListFile: Option[String] = None,
  numPartitions: Int = 10,
  dataFormat: String = TFRECORD,
  predictionScoreColumnName: String = PREDICTION_SCORE,
  predictionScorePerCoordinateColumnName: String = PREDICTION_SCORE_PER_COORDINATE,
  offsetColumnName: String = OFFSET,
  uidColumnName: String = UID,
  savePassiveData: Boolean = true,
  maxNumOfSamplesPerModel: Option[Int] = None,
  minNumOfSamplesPerModel: Option[Int] = None
)

/**
 * Parser for data partition job.
 */
object DataPartitionerParser {
  private val dataPartitionerParser = new scopt.OptionParser[DataPartitionerParams](
    "Parsing command line for data partitioner job.") {

    opt[String]("partitionId").action((x, p) => p.copy(partitionId = x.trim))
      .required
      .text(
        """Required.
          |The column name used to partition.""".stripMargin)

    opt[String]("metadataFile").action((x, p) => p.copy(metadataFile = x.trim))
      .required
      .text(
        """Required.
          |Input metadata used for random effect data processing.""".stripMargin)

    opt[String]("outputMetadataFile").action((x, p) => p.copy(outputMetadataFile = x.trim))
      .required
      .text(
        """Required.
          |Output metadata file matches processed dataset.""".stripMargin)

    opt[String]("trainingDataDir").action((x, p) => p.copy(trainingDataDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Training input dataset path.""".stripMargin)

    opt[String]("validationDataDir").action((x, p) => p.copy(validationDataDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input dataset path.""".stripMargin)

    opt[String]("trainingScoreDir").action((x, p) => p.copy(trainingScoreDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Training input score path.""".stripMargin)

    opt[String]("validationScoreDir").action((x, p) => p.copy(validationScoreDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Validation input score path.""".stripMargin)

    opt[String]("partitionedTrainingDataDir").action((x, p) => p.copy(partitionedTrainingDataDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Partition result path for training data.""".stripMargin)

    opt[String]("partitionedValidationDataDir").action((x, p) => p.copy(partitionedValidationDataDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Partition result path for validation data.""".stripMargin)

    opt[String]("trainingScorePerCoordinateDir").action((x, p) => p.copy(trainingScorePerCoordinateDir = if (x.trim.isEmpty) None else Some(x.trim)))
      .optional
      .text(
        """Optional.
          |Path to the per-coordinate training score of the previous iteration.""".stripMargin)

    opt[String]("validationScorePerCoordinateDir").action((x, p) => p.copy(validationScorePerCoordinateDir = if (x.trim.isEmpty) None else Some(x.trim)))
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

    opt[String]("savePassiveData").action((x, p) => p.copy(savePassiveData = x.toLowerCase == "true"))
      .optional
      .text(
        """Optional.
          |Boolean whether to save passive data.""".stripMargin)

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
      if (IoUtils.isEmptyStr(p.trainingDataDir) && IoUtils.isEmptyStr(p.validationDataDir)) {
        failure("Neither training nor validation data path is provided.")
      }
      else success)

    checkConfig(p =>
      if (!IoUtils.isEmptyStr(p.trainingDataDir)) {
        if (IoUtils.isEmptyStr(p.partitionedTrainingDataDir)) {
          failure("Option --trainOutputPartitionDataPath is required when --trainInputDataPath is not empty.")
        }
        else if (IoUtils.isEmptyStr(p.outputPartitionListFile)) {
          failure("Option --outputPartitionListFile is required when --trainInputDataPath is not empty.")
        }
        else success
      }
      else success)

    checkConfig(p =>
      if (!IoUtils.isEmptyStr(p.validationDataDir)) {
        if (IoUtils.isEmptyStr(p.partitionedValidationDataDir)) {
          failure("Option --validationOutputPartitionDataPath is required when --validationInputDataPath is not empty.")
        }
        else success
      }
      else success)
  }

  def parse(args: Seq[String]): DataPartitionerParams = {
    val emptyDataPartitionerParams = DataPartitionerParams(
      partitionId = "",
      metadataFile = "",
      outputMetadataFile = "",
      trainingDataDir = Some(""),
      partitionedTrainingDataDir = Some(""),
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
