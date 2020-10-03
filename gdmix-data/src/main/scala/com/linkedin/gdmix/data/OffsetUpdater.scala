package com.linkedin.gdmix.data

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

import com.linkedin.gdmix.parsers.OffsetUpdaterParser
import com.linkedin.gdmix.parsers.OffsetUpdaterParams
import com.linkedin.gdmix.utils.Constants.{AVRO, FLOAT, LONG}
import com.linkedin.gdmix.utils.IoUtils

/**
 * Update offset for fixed-effect data once random-effects are done. Only used in multiple coordinate descent iterations.
 */
object OffsetUpdater {

  def main(args: Array[String]): Unit = {

    val params = OffsetUpdaterParser.parse(args)
    // Create a Spark session.
    val spark = SparkSession.builder().appName(getClass.getName).getOrCreate()
    try {
      run(spark, params)
    } finally {
      spark.stop()
    }
  }

  def run(spark: SparkSession, params: OffsetUpdaterParams): Unit = {

    // Parse the commandline option.
    val trainInputDataPath = params.trainInputDataPath
    val trainInputScorePath = params.trainInputScorePath
    val trainPerCoordinateScorePath = params.trainPerCoordinateScorePath
    val trainOutputDataPath = params.trainOutputDataPath
    val validationInputDataPath = params.validationInputDataPath
    val validationInputScorePath = params.validationInputScorePath
    val validationPerCoordinateScorePath = params.validationPerCoordinateScorePath
    val validationOutputDataPath = params.validationOutputDataPath
    val predictionScore = params.predictionScore
    val predictionScorePerCoordinate = params.predictionScorePerCoordinate
    val offset = params.offset
    val uid = params.uid
    val dataFormat = params.dataFormat

    // Create a Spark session.
    val spark = SparkSession.builder().appName(getClass.getName).getOrCreate()

    // Update offset in training data.
    val trainInputData = IoUtils.readDataFrame(spark, trainInputDataPath, dataFormat)
    val trainInputScore = IoUtils.readDataFrame(spark, trainInputScorePath, AVRO)
    val trainPerCoordinateScore = IoUtils.readDataFrame(spark, trainPerCoordinateScorePath, AVRO)
    val trainOutputData = updateOffset(
      trainInputData,
      trainInputScore,
      Some(trainPerCoordinateScore),
      predictionScore,
      predictionScorePerCoordinate,
      offset,
      uid)
    IoUtils.saveDataFrame(trainOutputData, trainOutputDataPath, dataFormat)

    // Update offset in validation data.
    if (!IoUtils.isEmptyStr(validationInputDataPath)
      && !IoUtils.isEmptyStr(validationInputScorePath)
      && !IoUtils.isEmptyStr(validationPerCoordinateScorePath)
      && !IoUtils.isEmptyStr(validationOutputDataPath)) {
      val validationInputData = IoUtils.readDataFrame(spark, validationInputDataPath.get, dataFormat)
      val validationInputScore = IoUtils.readDataFrame(spark, validationInputScorePath.get, AVRO)
      val validationPerCoordinateScore = IoUtils.readDataFrame(spark, validationPerCoordinateScorePath.get, AVRO)
      val validationOutputData = updateOffset(
        validationInputData,
        validationInputScore,
        Some(validationPerCoordinateScore),
        predictionScore,
        predictionScorePerCoordinate,
        offset,
        uid)
      IoUtils.saveDataFrame(validationOutputData, validationOutputDataPath.get, dataFormat)
    }
  }

  /**
   * Update "offset" = "offset of last coordinate" - "offset of previous iteration for the same coordinate".
   *
   * @param data Data frame to train or validate
   * @param dFLastCoordinateOffset Data frame contains the offset of last coordinate
   * @param dFPerCoordinateScoreOpt Optional data frame contains per-coordinate score of last iteration
   * @param predictionScore Column name of prediction score
   * @param predictionScorePerCoordinate Column name of prediction score per coordinate
   * @param offset Column name of offset
   * @param uid Column name of uid
   * @return The data frame with offset updated.
   */
  def updateOffset(
    data: DataFrame,
    dFLastCoordinateOffset: DataFrame,
    dFPerCoordinateScoreOpt: Option[DataFrame],
    predictionScore: String,
    predictionScorePerCoordinate: String,
    offset: String,
    uid: String): DataFrame = {

    val lastCoordinateOffset = dFLastCoordinateOffset
      .select(col(uid).cast(LONG), col(predictionScore).cast(FLOAT) as offset)

    val offsetUpdated = dFPerCoordinateScoreOpt match {
      case Some(dFPerCoordinateScore) =>
        val perCoordinateScore = dFPerCoordinateScore.select(col(uid), col(predictionScorePerCoordinate))
        lastCoordinateOffset
          .join(perCoordinateScore, uid)
          .withColumn(offset, col(offset) - col(predictionScorePerCoordinate))
          .drop(predictionScorePerCoordinate)

      case None =>
        lastCoordinateOffset
    }
    data.drop(offset).join(offsetUpdated, uid)
  }
}