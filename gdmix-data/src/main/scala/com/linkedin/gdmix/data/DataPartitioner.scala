package com.linkedin.gdmix.data

import com.databricks.spark.avro._
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.json4s.DefaultFormats

import com.linkedin.gdmix.parsers.DataPartitionerParser
import com.linkedin.gdmix.parsers.DataPartitionerParams
import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.{IoUtils, PartitionUtils}

/**
 * Partition training and validation data into multiple partitions.
 */
object DataPartitioner {

  def main(args: Array[String]): Unit = {

    val params = DataPartitionerParser.parse(args)

    // Create a Spark session.
    val spark = SparkSession.builder().appName(getClass.getName).getOrCreate()
    try {
      run(spark, params)
    } finally {
      spark.stop()
    }
  }

  def run(spark: SparkSession, params: DataPartitionerParams): Unit = {

    // Parse the commandline option.
    val trainInputDataPath = params.trainingDataDir
    val trainInputScorePath = params.trainingScoreDir
    val trainPerCoordinateScorePath = params.trainingScorePerCoordinateDir
    val trainOutputPartitionDataPath = params.partitionedTrainingDataDir
    val validationInputDataPath = params.validationDataDir
    val validationInputScorePath = params.validationScoreDir
    val validationPerCoordinateScorePath = params.validationScorePerCoordinateDir
    val validationOutputPartitionDataPath = params.partitionedValidationDataDir
    val inputMetadataFile = params.metadataFile
    val outputMetadataFile = params.outputMetadataFile
    val outputPartitionListFile = params.outputPartitionListFile
    val partitionEntity = params.partitionId
    val numPartitions = params.numPartitions
    val dataFormat = params.dataFormat
    val predictionScore = params.predictionScoreColumnName
    val predictionScorePerCoordinate = params.predictionScorePerCoordinateColumnName
    val offset = params.offsetColumnName
    val uid = params.uidColumnName
    val maxNumOfSamplesPerModel = params.maxNumOfSamplesPerModel
    val minNumOfSamplesPerModel = params.minNumOfSamplesPerModel

    implicit val jsonFormat: DefaultFormats.type = DefaultFormats

    // Create a Spark session.
    val spark = SparkSession.builder().appName(getClass.getName).getOrCreate()

    import spark.implicits._

    // Set up Hadoop file system.
    val hadoopJobConf = new JobConf()
    val fs = FileSystem.get(hadoopJobConf)

    val schemaOpt = if (dataFormat == TFRECORD) Some(MetadataGenerator
      .createSchemaForTfrecord(inputMetadataFile)) else None

    // Partition training dataset if the training input data is provided.
    val trainOutputOpt = if (!IoUtils.isEmptyStr(trainInputDataPath)) {
      val trainInputData = IoUtils.readDataFrame(spark, trainInputDataPath.get, dataFormat, schemaOpt)
      val trainInputScoreOpt = if (!IoUtils.isEmptyStr(trainInputScorePath)) {
        Some(spark.read.avro(trainInputScorePath.get))
      } else {
        None
      }
      val trainPerCoordinateScoreOpt = if (!IoUtils.isEmptyStr(trainPerCoordinateScorePath)) {
        Some(spark.read.avro(trainPerCoordinateScorePath.get))
      } else {
        None
      }
      // Group and partition per-entity data. Split the training data into active and passive dataset.
      val outputDf = groupPartitionAndSaveDataset(
        trainInputData,
        trainInputScoreOpt,
        trainPerCoordinateScoreOpt,
        trainOutputPartitionDataPath.get,
        partitionEntity,
        numPartitions,
        dataFormat,
        predictionScore,
        predictionScorePerCoordinate,
        offset,
        uid,
        minNumOfSamplesPerModel,
        maxNumOfSamplesPerModel,
        inputMetadataFile,
        ifSplitData = true)

      // Entity list contains entity id and partition id.
      val entityList = outputDf.select(col(partitionEntity), col(PARTITION_ID))
      // Get the partition ids with non-empty entities and save the ids as the partition list.
      val partitionIds = entityList
        .select(col(PARTITION_ID))
        .distinct()
        .map(row => row.getAs[Int](PARTITION_ID))
        .collect()
      IoUtils.writeFile(fs, new Path(outputPartitionListFile.get), partitionIds.sorted.mkString(","))
      Some(outputDf)
    } else {
      None
    }

    // Partition validation dataset if the validation input data is provided.
    val validationOutputOpt = if (!IoUtils.isEmptyStr(validationInputDataPath)) {
      val validationInputData = IoUtils.readDataFrame(spark, validationInputDataPath.get, dataFormat, schemaOpt)
      val validationInputScoreOpt = if (!IoUtils.isEmptyStr(validationInputScorePath)) {
        Some(spark.read.avro(validationInputScorePath.get))
      } else {
        None
      }
      val validPerCoordinateScoreOpt = if (!IoUtils.isEmptyStr(validationPerCoordinateScorePath)) {
        Some(spark.read.avro(validationPerCoordinateScorePath.get))
      } else {
        None
      }
      // Group and partition
      val outputDf = groupPartitionAndSaveDataset(
        validationInputData,
        validationInputScoreOpt,
        validPerCoordinateScoreOpt,
        validationOutputPartitionDataPath.get,
        partitionEntity,
        numPartitions,
        dataFormat,
        predictionScore,
        predictionScorePerCoordinate,
        offset,
        uid,
        minNumOfSamplesPerModel,
        maxNumOfSamplesPerModel,
        inputMetadataFile,
        ifSplitData = false)

      Some(outputDf)
    } else {
      None
    }

    // Drop partitionId in the schema since it is not used in model training.
    val dfSchema = trainOutputOpt match {
      case Some(trainOutput) => trainOutput.drop(PARTITION_ID).schema
      case None => validationOutputOpt match {
        case Some(validationOutput) => validationOutput.drop(PARTITION_ID).schema
        case None => throw new IllegalArgumentException("No partition output is generated.")
      }
    }

    // Save the metadata of partitioned dataset.
    MetadataGenerator.saveMetaDataForPartitions(
      dfSchema,
      inputMetadataFile,
      outputMetadataFile,
      dataFormat)
  }

  /**
   * Optionally join the input score, partition the input data, group the samples per entity.
   *
   * @param inputData Input data frame
   * @param inputScoreOpt Input accumulative scores from previous stages
   * @param perCoordinateScoreOpt The score for current entity only, used in multiple iterations
   * @param outputPartitionDataPath The root path where the output dataset will be saved
   * @param partitionEntity The entity by which the output dataset will be partitioned
   * @param numPartitions The number of partitions
   * @param dataFormat Avro or TFRecord
   * @param predictionScore The column name for predicted score
   * @param predictionScorePerCoordinate Column name for the prediction score of the previous iteration for the same
   *                                     coordinate
   * @param offset Column name for the updated offset
   * @param uid Column name for the unique id
   * @param lowerBound The minimal samples per entity
   * @param upperBound The maximal samples per entity
   * @param inputMetadataFile The input metadata file path
   * @param ifSplitData Whether to split the data into active and passive folders for the output
   * @return The grouped data frame with partition id.
   */
  private[data] def groupPartitionAndSaveDataset(
    inputData: DataFrame,
    inputScoreOpt: Option[DataFrame],
    perCoordinateScoreOpt: Option[DataFrame],
    outputPartitionDataPath: String,
    partitionEntity: String,
    numPartitions: Int,
    dataFormat: String,
    predictionScore: String,
    predictionScorePerCoordinate: String,
    offset: String,
    uid: String,
    lowerBound: Option[Int],
    upperBound: Option[Int],
    inputMetadataFile: String,
    ifSplitData: Boolean): DataFrame = {

    // Join the offsets.
    val joinedDf = joinAndUpdateScores(
      inputData,
      inputScoreOpt,
      perCoordinateScoreOpt,
      predictionScore,
      predictionScorePerCoordinate,
      offset,
      uid)

    // Group and bound the dataset by a lower bound and an upper bound.
    val groupedDf = boundAndGroupData(joinedDf, lowerBound, upperBound, partitionEntity)
      .persist(StorageLevel.MEMORY_AND_DISK)

    // Add partition Id.
    val dFWithPartitionId = groupedDf
      .withColumn(PARTITION_ID, PartitionUtils.getPartitionIdUDF(numPartitions)(col(partitionEntity)))

    val recordType = if (dataFormat.equals(TFRECORD)) TF_SEQUENCE_EXAMPLE else null

    if (ifSplitData) {
      // Save the active data.
      val activeData = dFWithPartitionId.filter(col(GROUP_ID) === 0).drop(GROUP_ID)
      if (!activeData.rdd.isEmpty()) {
        IoUtils.saveDataFrame(
          activeData,
          outputPartitionDataPath + "/" + ACTIVE,
          dataFormat,
          numPartitions,
          PARTITION_ID,
          recordType)
      }

      // Save the passive data. Passive data is generated when there is a lower bound or an upper bound.
      if (!(lowerBound.isEmpty && upperBound.isEmpty)) {
        val passiveData = dFWithPartitionId.filter(col(GROUP_ID) =!= 0).drop(GROUP_ID)
        if (!passiveData.rdd.isEmpty()) {
          IoUtils.saveDataFrame(
            passiveData,
            outputPartitionDataPath + "/" + PASSIVE,
            dataFormat,
            numPartitions,
            PARTITION_ID,
            recordType)
        }
      }
    } else {
      // Validation and inference data don't need to be filtered as active and passive.
      IoUtils.saveDataFrame(
        dFWithPartitionId.drop(GROUP_ID),
        outputPartitionDataPath,
        dataFormat,
        numPartitions,
        PARTITION_ID,
        recordType)
    }

    // Return the data frame with partition id.
    dFWithPartitionId.drop(GROUP_ID)
  }

  /**
   * Bound the data frame by lower bound and upper bound, and group the bounded data frame by
   * (partitionEntity, groupId). After bounding, the dataset is split into active data and passive data. Active data
   * is for training the random effect and both the active data and passive data will be scored after random effect
   * models are trained.
   *
   * @param dataFrame Input data frame
   * @param lowerBound Lower bound on the number of records per entity
   * @param upperBound Upper bound on the number of records per entity
   * @param partitionEntity The entity by which the output dataset will be partitioned
   * @return The bounded and grouped data frame.
   */
  private[data] def boundAndGroupData(
    dataFrame: DataFrame,
    lowerBound: Option[Int],
    upperBound: Option[Int],
    partitionEntity: String): DataFrame = {

    // Get the group id for each entity.
    val dfWithGroupId = getGroupId(dataFrame, lowerBound, upperBound, partitionEntity)

    // The columns of partitionEntity and groupId are not needed to be grouped since they are the same per-group.
    val groupedColumnNames = dfWithGroupId.columns
      .filter(!_.equals(partitionEntity))
      .filter(!_.equals(GROUP_ID))

    // Group by partitionEntity, merge the featurebag_indices if it is sparse
    val aggExpr = groupedColumnNames.map(x => collect_list(col(x)).alias(x))
    dfWithGroupId
      .groupBy(col(partitionEntity), col(GROUP_ID))
      .agg(aggExpr.head, aggExpr.tail: _*)
  }

  /**
   * Get the group id of each entity. Group id is used to distinguish active data and passive data. Group id 0
   * indicates active data; Group id -1 indicates the passive data filtered by lower bound; Any group id greater
   * than 0 indicates the passive data filtered by upper bound.
   *
   * @param dataFrame Input data frame
   * @param lowerBound Lower bound on the number of records per entity
   * @param upperBound Upper bound on the number of records per entity
   * @param partitionEntity The entity by which the output dataset will be partitioned
   * @return The bounded and grouped data frame.
   */
  private[data] def getGroupId(
    dataFrame: DataFrame,
    lowerBound: Option[Int],
    upperBound: Option[Int],
    partitionEntity: String): DataFrame = {

    // No lower bound and upper bound. All the samples are active data.
    if (lowerBound.isEmpty && upperBound.isEmpty) {
      dataFrame.withColumn(GROUP_ID, lit(0))
    } else {
      // If there's a bound, either lower bound or upper bound, we need to count the samples per-entity.
      val perEntityCounts = dataFrame
        .select(partitionEntity)
        .groupBy(partitionEntity)
        .count()
        .select(col(partitionEntity), col(COUNT).alias(PER_ENTITY_TOTAL_SAMPLE_COUNT))
      val dfWithEntityCount = dataFrame.join(perEntityCounts, partitionEntity)

      // If there's an upper bound, calculate the number of groups needed to bound the data.
      val dfWithGroupCounts = if (!upperBound.isEmpty) {
        dfWithEntityCount
          .withColumn(PER_ENTITY_GROUP_COUNT, (col(PER_ENTITY_TOTAL_SAMPLE_COUNT) / upperBound.get + 1).cast(IntegerType))
      } else {
        dfWithEntityCount.withColumn(PER_ENTITY_GROUP_COUNT, lit(1))
      }

      // Assign the group id and drop redundant columns. The group id is uniformly random.
      // TODO: Explore different sampling strategy.
      val dfWithGroupId = lowerBound match {
        case Some(lb) =>
          dfWithGroupCounts
          .withColumn(GROUP_ID,
          when(col(PER_ENTITY_TOTAL_SAMPLE_COUNT) < lb, -1)
            .otherwise((col(PER_ENTITY_GROUP_COUNT) * rand()).cast(IntegerType)))
        case _ =>
          dfWithGroupCounts
            .withColumn(GROUP_ID, (col(PER_ENTITY_GROUP_COUNT) * rand()).cast(IntegerType))
      }
      dfWithGroupId.drop(PER_ENTITY_TOTAL_SAMPLE_COUNT, PER_ENTITY_GROUP_COUNT)
    }
  }

  /**
   * Join data with score optionally, update the input score optionally. It is easy to
   * explain with an example. Let us assume we are training a (global, per-member, per-job),
   * currently we are at the 2nd iteration and at training per-job.
   * inputScoreOpt = global score + per-member score from the 2nd iteration.
   * dFPerCoordinateScoreOpt = per-job score from the 1st iteration.
   * After the update: new offset
   * = inputScoreOpt - dFPerCoordinateScoreOpt
   * = global score (2nd iter) + per-member score (2nd iter) - per-job score (1st iter).
   * The score of the same coordinate from last iteration is subtracted.
   *
   * @param inputData Input data frame
   * @param inputScoreOpt Input accumulative scores until last coordinate.
   * @param dFPerCoordinateScoreOpt Input individual score of last coordinate
   * @param predictionScore Column name for the accumulative score util last coordinate
   * @param predictionScorePerCoordinate Column name for the prediction score of the previous iteration for the same
   *                                     coordinate
   * @param offset Column name for the updated offset
   * @param uid Column name for the unique id
   * @return Data frame with the updated offset.
   */
  private def joinAndUpdateScores(
    inputData: DataFrame,
    inputScoreOpt: Option[DataFrame],
    dFPerCoordinateScoreOpt: Option[DataFrame],
    predictionScore: String,
    predictionScorePerCoordinate: String,
    offset: String,
    uid: String): DataFrame = {
    // Optionally join and update the offset
    inputScoreOpt match {
      case Some(inputScore) => OffsetUpdater.updateOffset(
        inputData,
        inputScore,
        dFPerCoordinateScoreOpt,
        predictionScore,
        predictionScorePerCoordinate,
        offset,
        uid)
      case None => inputData
    }
  }
}
