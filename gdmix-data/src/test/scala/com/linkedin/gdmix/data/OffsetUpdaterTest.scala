package com.linkedin.gdmix.data

import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.gdmix.utils.SharedSparkSession

/**
 * Unit tests for [[OffsetUpdater]].
 */
class OffsetUpdaterTest extends SharedSparkSession {

  import spark.implicits._
  import OffsetUpdaterTest._

  /**
   * Unit test for [[OffsetUpdater.updateOffset()]].
   */
  @Test
  def testUpdateOffset(): Unit = {
    val data = Seq((1L, 0.0F), (2L, 0.0F)).toDF(UID, OFFSET)
    val lastOffset = Seq((1L, 1.0F), (2L, 2.0F)).toDF(UID, PREDICTION_SCORE)
    val perCoordinateScore = Seq((1L, 0.1F), (2L, 0.2F)).toDF(UID, PREDICTION_SCORE_PER_COORDINATE)

    val updatedData1 = OffsetUpdater.updateOffset(
      data,
      lastOffset,
      None,
      PREDICTION_SCORE,
      PREDICTION_SCORE_PER_COORDINATE,
      OFFSET,
      UID)
    val res1 = updatedData1.map(row => (row.getAs[Long](UID), row.getAs[Float](OFFSET))).collect().toMap
    assertEquals(res1(1L), 1.0F)
    assertEquals(res1(2L), 2.0F)

    val updatedData2 = OffsetUpdater.updateOffset(
      data,
      lastOffset,
      Some(perCoordinateScore),
      PREDICTION_SCORE,
      PREDICTION_SCORE_PER_COORDINATE,
      OFFSET,
      UID)
    val res2 = updatedData2.map(row => (row.getAs[Long](UID), row.getAs[Float](OFFSET))).collect().toMap
    assertEquals(res2(1L), 0.9F)
    assertEquals(res2(2L), 1.8F)
  }
}

object OffsetUpdaterTest {
  val OFFSET = "offset"
  val UID = "uid"
  val PREDICTION_SCORE = "predictionScore"
  val PREDICTION_SCORE_PER_COORDINATE = "predictionScorePerCoordinate"
}
