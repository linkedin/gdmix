package com.linkedin.gdmix.utils

import org.apache.log4j.Logger
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.testng.annotations.{AfterSuite, BeforeSuite}

/**
 * We need a common utility to create sparkSession. This is because
 * of the way Spark session works. We cannot have separate sparkSession
 * in each function/compilation unit level.
 */
trait SharedSparkSession {
  private var _spark: SparkSession = _
  lazy val spark: SparkSession = _spark
  val logger: Logger = Logger.getLogger(getClass)

  @BeforeSuite
  def setupSpark(): Unit = {
    val sparkConf = new SparkConf().setMaster("local[*]").setAppName("SharedSparkSession")
    _spark = SparkSession.builder().config(sparkConf).getOrCreate()
  }

  @AfterSuite
  def stopSpark(): Unit = {
    _spark.stop()
    _spark = null
  }
}
