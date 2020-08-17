package com.linkedin.gdmix.evaluation

import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.gdmix.utils.SharedSparkSession

/**
 * Unit tests for [[AreaUnderROCCurveEvaluator]].
 */
class NdcgEvaluatorTest extends SharedSparkSession {

  import spark.implicits._

  @DataProvider(name = "ScoreAndLabels")
  def scoreAndLabels(): Array[Array[Any]] = {
    Array(
      Array(
        Array(Array(1.0, 3.0, 2.0, 4.0), Array(3.0, 2.0, 1.0)), // Prediction scores
        Array(Array(0.0, 0.0, 1.0, 2.0), Array(1.0, 0.0, 1.0)), // Labels
        0.93498, // Traditional NDCG@3 = (0.95023 + 0.91972) / 2
        0.94183), // Non-traditional NDCG@3 = (0.96394 + 0.91972) / 2
      Array(
        Array(Array(3.0, 4.0, 2.0), Array(3.0, 2.0)), // Prediction scores
        Array(Array(0.0, 2.0, 1.0), Array(1.0, 0.0)), // Labels
        0.97512, // Traditional NDCG@3 = (0.95023 + 1) / 2
        0.98197) // Non-traditional NDCG@3 = (0.96394 + 1) / 2
    )
  }

  /**
   * Unit test for [[NdcgEvaluator.calculateNdcgAt]].
   */
  @Test(dataProvider = "ScoreAndLabels")
  def testCalculateNdcgAt(
    scores: Array[Array[Double]],
    labels: Array[Array[Double]],
    traditionalNdcg: Double,
    nonTraditionalNdcg: Double): Unit = {
    val labelName = "label"
    val scoreName = "score"
    val scoresAndLabels = (scores zip labels)
      .toList
      .toDF(scoreName, labelName)
      .rdd
      .map(row => (row.getAs[Seq[Double]](scoreName), row.getAs[Seq[Double]](labelName)))

    val epsilon = 1.0e-4
    val ndcg1 = NdcgEvaluator.calculateNdcgAt(scoresAndLabels, 3, true)
    assertEquals(ndcg1, traditionalNdcg, epsilon)
    val ndcg2 = NdcgEvaluator.calculateNdcgAt(scoresAndLabels, 3, false)
    assertEquals(ndcg2, nonTraditionalNdcg, epsilon)
  }
}
