package com.linkedin.gdmix.utils

import com.linkedin.gdmix.utils.Constants.{CROSS, MEANS, MODEL_ID, NAME, TERM, VALUE}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.Row
import org.apache.spark.sql.types._
import org.testng.Assert.assertTrue
import org.testng.annotations.Test

/**
 * Test functions in ConversionUtils
 */
class ConversionUtilsTest extends SharedSparkSession{

  import spark.implicits._

  /**
   * Unit test for [[splitModelIdUdf]].
   */
  @Test
  def tesSplitModelIdUdf(): Unit = {
    val schema = StructType(List(StructField(MEANS, StructType(List(StructField(NAME,StringType, true),
      StructField(TERM, StringType, true), StructField(VALUE, DoubleType, true))), true)))
    val inputData = Seq(
      Row(Row(s"m1${CROSS}f1", "t1", 0.3)),
      Row(Row(s"m2${CROSS}f2", "", 0.5)))
    val inputDf = spark.createDataFrame(spark.sparkContext.parallelize(inputData), schema)
    val splitDf = inputDf.withColumn(MEANS,
      ConversionUtils.splitModelIdUdf(col(MEANS)))
    val expectedData = Seq(
      Row(Row("m1", Row("f1", "t1", 0.3))),
      Row(Row("m2", Row("f2", "", 0.5))))
    val expectedSchema = StructType(List(StructField(MEANS, StructType(List(StructField("_1", StringType, true),
      StructField("_2", StructType(List(StructField(NAME,StringType, true),
      StructField(TERM, StringType, true), StructField(VALUE, DoubleType, true))), true))))))
    val expectedDf = spark.createDataFrame(spark.sparkContext.parallelize(expectedData), expectedSchema)
    assertTrue(TestUtils.equalSmallDataFrame(splitDf, expectedDf, MEANS))
  }
}
