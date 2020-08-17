package com.linkedin.gdmix.evaluation

import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.gdmix.utils.SharedSparkSession

/**
 * Unit tests for [[AreaUnderROCCurveEvaluator]].
 */
class AreaUnderROCCurveEvaluatorTest extends SharedSparkSession {

  import spark.implicits._

  @DataProvider(name = "ScoreAndLabels")
  def scoreAndLabels(): Array[Array[Any]] = {
    Array(
      Array(Array(0.1, 0.4, 0.35, 0.8),
        Array(0, 0, 1.0, 1.0),
        0.75),
      Array(Array(0.5, 0.7, 0.3, 0.4, 0.45, 0.8),
        Array(0, 0, 1.0, 1.0, 0, 1.0),
        0.3333333),
      Array(Array(0.5, 0.75, 0.8, 0.2, 0.3, 0.4, 0.45, 0.5),
        Array(0, 0, 0, 0, 1.0, 1.0, 0, 1.0),
        0.3)
    )
  }

  /**
   * Unit test for [[AreaUnderROCCurveEvaluator.calculateAreaUnderROCCurve]].
   */
  @Test(dataProvider = "ScoreAndLabels")
  def testCalculateAreaUnderROCCurve(score: Array[Double], label: Array[Double], auc: Double): Unit = {
    val labelName = "label"
    val scoreName = "score"
    val delta = 1.0e-5
    val df = (score zip label).toList.toDF(scoreName, labelName)
    val calculatedAUC = AreaUnderROCCurveEvaluator.calculateAreaUnderROCCurve(df, labelName, scoreName)
    assertEquals(calculatedAUC, auc, delta)
  }
}
