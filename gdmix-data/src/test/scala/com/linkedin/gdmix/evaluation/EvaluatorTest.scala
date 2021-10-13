package com.linkedin.gdmix.evaluation

import org.testng.Assert.assertEquals
import org.testng.annotations.{DataProvider, Test}

import com.linkedin.gdmix.utils.SharedSparkSession

/**
 * Unit tests for [[Evaluator]].
 */
class EvaluatorTest extends SharedSparkSession {

  import spark.implicits._

  @DataProvider(name = "AUCScoreAndLabels")
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

  @DataProvider(name = "MSEScoreAndLabels")
  def predictionAndObservations(): Array[Array[Any]] = {
    Array(
      Array(Array(0.1, 0.4, 0.35, 0.8),
        Array(0, 0, 1.0, 2.0),
        0.5081250),
      Array(Array(0.5, 0.7, 1.3, 3.4, 5.45, 0.8),
        Array(0, 0, 1.0, 2.0, 3.0, 1.0),
        1.4720833),
      Array(Array(0.5, 0.75, -0.8, 0.2, -0.3, 0.4, 0.45, 0.5),
        Array(0, 0, 0.2, 0, 0.4, -1.1, 0, -1.0),
        0.880625)
    )
  }

  /**
   * Unit test for [[Evaluator.calculateMetric]] on caculating AUC.
   */
  @Test(dataProvider = "AUCScoreAndLabels")
  def testCalculateAreaUnderROCCurve(score: Array[Double], label: Array[Double], auc: Double): Unit = {
    val metricName = "auc"
    val labelName = "label"
    val scoreName = "score"
    val delta = 1.0e-5
    val df = (score zip label).toList.toDF(scoreName, labelName)
    val calculatedAUC = Evaluator.calculateMetric(df, labelName, scoreName, metricName)
    assertEquals(calculatedAUC, auc, delta)
  }


  /**
   * Unit test for [[Evaluator.calculateMetric]] on caculating MSE.
   */
  @Test(dataProvider = "MSEScoreAndLabels")
  def testCalculateMeanSquaredError(score: Array[Double], label: Array[Double], mse: Double): Unit = {
    val metricName = "mse"
    val labelName = "label"
    val scoreName = "score"
    val delta = 1.0e-5
    val df = (score zip label).toList.toDF(scoreName, labelName)
    val calculatedMSE = Evaluator.calculateMetric(df, labelName, scoreName, metricName)
    assertEquals(calculatedMSE, mse, delta)
  }
}
