package com.linkedin.gdmix.data

import scala.collection.mutable.WrappedArray

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types._
import org.testng.Assert.assertEquals
import org.testng.annotations.Test

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.SharedSparkSession

/**
 * Unit tests for [[DataPartitioner]].
 */
class DataPartitionerTest extends SharedSparkSession {

  import DataPartitionerTest._

  // Raw data.
  val uid = Seq(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)
  val entityId = Seq(0L, 0L, 0L, 1L, 1L, 1L, 1L, 1L, 1L, 2L)
  val label = Seq(0, 0, 1, 1, 1, 0, 0, 1, 1, 1)
  val indices = Seq(Seq(0L, 1L), Seq(0L, 1L, 2L), Seq(3L, 4L), Seq(5L, 6L), Seq(7L, 8L),
    Seq(9L, 10L), Seq(3L, 4L), Seq(5L, 9L), Seq(0L), Seq(0L, 2L))
  val values = Seq(Seq(0F, 1F), Seq(0F, 1.0F, 2.2F), Seq(3F, 4.1F), Seq(5.5F, 6.6F), Seq(7.7F, 8.8F),
    Seq(9.3F, 10.12F), Seq(0.3F, 0.8F), Seq(0.8F, 1.8F), Seq(0.0F, 2.1F), Seq(1.0F, -2.2F))

  // Expected values. Note the second column is randomly split with lowerBound = 2 and upperBound = 4. Therefore the second
  // column won't be tested.
  val expectedUid = Seq(Seq(0L, 1L, 2L), Seq(3L, 4L, 5L, 6L, 7L, 8L), Seq(9L))
  val expectedEid = Seq(0L, 1L, 2L)
  val expectedLabel = Seq(Seq(0, 0, 1), Seq(1, 1, 0, 0, 1, 1), Seq(1))
  val expectedIndices = Seq(
    Seq(Seq(0L, 1L), Seq(0L, 1L, 2L), Seq(3L, 4L)),
    Seq(Seq(5L, 6L), Seq(7L, 8L), Seq(9L, 10L), Seq(3L, 4L), Seq(5L, 9L), Seq(0L)),
    Seq(Seq(0L, 2L)))
  val expectedValues = Seq(
    Seq(Seq(0F, 1F), Seq(0F, 1.0F, 2.2F), Seq(3F, 4.1F)),
    Seq(Seq(5.5F, 6.6F), Seq(7.7F, 8.8F), Seq(9.3F, 10.12F), Seq(0.3F, 0.8F), Seq(0.8F, 1.8F), Seq(0.0F, 2.1F)),
    Seq(Seq(1.0F, -2.2F)))

  val avroSchema = StructType(
    Seq(
      StructField(UID, LongType, true),
      StructField(ENTITY_ID, LongType, true),
      StructField(LABEL, IntegerType, true),
      StructField(GLOBAL,
        StructType(
          Seq(
            StructField(INDICES, ArrayType(LongType, true), true),
            StructField(VALUES, ArrayType(FloatType, true), true))
        ), true)
    )
  )
  val tfRecordSchema = StructType(
    Seq(
      StructField(UID, LongType, true),
      StructField(ENTITY_ID, LongType, true),
      StructField(LABEL, IntegerType, true),
      StructField(INDICES, ArrayType(LongType, true), true),
      StructField(VALUES, ArrayType(FloatType, true), true)
    )
  )

  def createAvroDataFrame(
    uid: Seq[Long],
    entityId: Seq[Long],
    label: Seq[Int],
    indices: Seq[Seq[Long]],
    values: Seq[Seq[Float]]): DataFrame = {
    // Initialize columns.
    val inputData =  (uid zip entityId zip label zip indices zip values).map {
      case ((((u, e), l), i), v) => Row(u, e, l, Row(i, v))
    }.toList
    spark.createDataFrame(spark.sparkContext.parallelize(inputData), avroSchema)
  }

  def createTfRecordDataFrame(
    uid: Seq[Long],
    entityId: Seq[Long],
    label: Seq[Int],
    indices: Seq[Seq[Long]],
    values: Seq[Seq[Float]]): DataFrame = {
    // Initialize columns.
    val inputData =  (uid zip entityId zip label zip indices zip values).map {
      case ((((u, e), l), i), v) => Row(u, e, l, i, v)
    }.toList
    spark.createDataFrame(spark.sparkContext.parallelize(inputData), tfRecordSchema)
  }

  /**
   * Unit test for [[DataPartitioner.getGroupId]].
   */
  @Test()
  def testGetGroupIdAvro(): Unit = {
    val dfAvro = createAvroDataFrame(uid, entityId, label, indices, values)
    val lowerBound = 2
    val upperBound = 4

    val dfWithGroupId = DataPartitioner.getGroupId(dfAvro, lowerBound, upperBound, ENTITY_ID)

    // Entity 0 has 3 samples. The samples should all be active data with group id 0.
    val groupIdOfEntity0 = dfWithGroupId.filter(col(ENTITY_ID) === 0L).select(GROUP_ID).head.getInt(0)
    assertEquals(groupIdOfEntity0, 0)

    // Entity 1 has 6 samples. The number of groups should be 1 to 2. Note we uniformly randomize group id, the number
    // of samples per-group could possibly go beyond the upper bound, specially when the upper bound is small.
    val groupCountOfEntity1 = dfWithGroupId.filter(col(ENTITY_ID) === 1L).select(GROUP_ID).distinct().count()
    assert(groupCountOfEntity1 >= 1 && groupCountOfEntity1 <= 2)

    // Entity 2 has 1 samples. It should be assigned as passive data with group id -1.
    val groupIdOfEntity2 = dfWithGroupId.filter(col(ENTITY_ID) === 2L).select(GROUP_ID).head.getInt(0)
    assertEquals(groupIdOfEntity2, -1)
  }

  /**
   * Unit test for [[DataPartitioner.getGroupId]].
   */
  @Test()
  def testGetGroupIdTfRecord(): Unit = {
    val dfTfRecord = createTfRecordDataFrame(uid, entityId, label, indices, values)
    val lowerBound = 2
    val upperBound = 4

    val dfWithGroupId = DataPartitioner.getGroupId(dfTfRecord, lowerBound, upperBound, ENTITY_ID)

    // Entity 0 has 3 samples. The samples should all be active data with group id 0.
    val groupIdOfEntity0 = dfWithGroupId.filter(col(ENTITY_ID) === 0L).select(GROUP_ID).head.getInt(0)
    assertEquals(groupIdOfEntity0, 0)

    // Entity 1 has 6 samples. The number of groups should be 1 to 2. Note we uniformly randomize group id, the number
    // of samples per-group could possibly go beyond the upper bound, specially when the upper bound is small.
    val groupCountOfEntity1 = dfWithGroupId.filter(col(ENTITY_ID) === 1L).select(GROUP_ID).distinct().count()
    assert(groupCountOfEntity1 >= 1 && groupCountOfEntity1 <= 2)

    // Entity 2 has 1 samples. It should be assigned as passive data with group id -1.
    val groupIdOfEntity2 = dfWithGroupId.filter(col(ENTITY_ID) === 2L).select(GROUP_ID).head.getInt(0)
    assertEquals(groupIdOfEntity2, -1)
  }

  /**
   * Unit test for [[DataPartitioner.boundAndGroupData]].
   */
  @Test()
  def testBoundAndGroupDataAvro(): Unit = {
    val dfAvro = createAvroDataFrame(uid, entityId, label, indices, values)
    val lowerBound = 2
    val upperBound = 4
    val dfGrouped = DataPartitioner
      .boundAndGroupData(dfAvro, lowerBound, upperBound, ENTITY_ID)

    // Entity 0 has 3 samples. The samples should all be active data with group id 0.
    val entity0 = dfGrouped.filter(col(ENTITY_ID) === 0L).drop(GROUP_ID).collect()
    assertEquals(entity0.length, 1)
    assertEquals(entity0(0)(0), expectedEid(0))
    assertEquals(entity0(0)(1), expectedUid(0))
    assertEquals(entity0(0)(2), expectedLabel(0))
    val featuresOfEntity0 = entity0(0).getAs[WrappedArray[Row]](GLOBAL)
    for (i <- (0 to 2)) {
      val indexList = featuresOfEntity0(i).getAs[WrappedArray[Long]](INDICES).toList
      assertEquals(indexList, expectedIndices(0)(i))
    }
    for (i <- (0 to 2)) {
      val valueList = featuresOfEntity0(i).getAs[WrappedArray[Float]](VALUES).toList
      assertEquals(valueList, expectedValues(0)(i))
    }

    // Entity 1 has 6 samples. The number of groups should be 1 to 2. Note we uniformly randomize group id, the number
    // of samples per-group could possibly go beyond the upper bound, specially when the upper bound is small.
    val entity1 = dfGrouped.filter(col(ENTITY_ID) === 1L).drop(GROUP_ID).collect()
    assert(entity1.length >= 1 && entity1.length <= 2)

    // Entity 2 has 1 samples. It should be assigned as passive data with group id -1.
    val entity2 = dfGrouped.filter(col(ENTITY_ID) === 2L).drop(GROUP_ID).collect()
    assertEquals(entity2.length, 1)
    assertEquals(entity2(0)(0), expectedEid(2))
    assertEquals(entity2(0)(1), expectedUid(2))
    assertEquals(entity2(0)(2), expectedLabel(2))
    val featuresOfEntity2 = entity2(0).getAs[WrappedArray[Row]](GLOBAL)
    val indexList = featuresOfEntity2(0).getAs[WrappedArray[Long]](INDICES).toList
    assertEquals(indexList, expectedIndices(2)(0))
    val valueList = featuresOfEntity2(0).getAs[WrappedArray[Float]](VALUES).toList
    assertEquals(valueList, expectedValues(2)(0))
  }

  /**
   * Unit test for [[DataPartitioner.boundAndGroupData]].
   */
  @Test()
  def testBoundAndGroupDataTfRecord(): Unit = {
    val dfTfRecord = createTfRecordDataFrame(uid, entityId, label, indices, values)
    val lowerBound = 2
    val upperBound = 4
    val dfGrouped = DataPartitioner
      .boundAndGroupData(dfTfRecord, lowerBound, upperBound, ENTITY_ID)

    // Entity 0 has 3 samples. The samples should all be active data with group id 0.
    val entity0 = dfGrouped.filter(col(ENTITY_ID) === 0L).drop(GROUP_ID).collect()
    assertEquals(entity0.length, 1)
    assertEquals(entity0(0)(0), expectedEid(0))
    assertEquals(entity0(0)(1), expectedUid(0))
    assertEquals(entity0(0)(2), expectedLabel(0))
    assertEquals(entity0(0)(3), expectedIndices(0))
    assertEquals(entity0(0)(4), expectedValues(0))

    // Entity 1 has 6 samples. The number of groups should be 1 to 2. Note we uniformly randomize group id, the number
    // of samples per-group could possibly go beyond the upper bound, specially when the upper bound is small.
    val entity1 = dfGrouped.filter(col(ENTITY_ID) === 1L).drop(GROUP_ID).collect()
    assert(entity1.length >= 1 && entity1.length <= 2)

    // Entity 2 has 1 samples. It should be assigned as passive data with group id -1.
    val entity2 = dfGrouped.filter(col(ENTITY_ID) === 2L).drop(GROUP_ID).collect()
    assertEquals(entity2.length, 1)
    assertEquals(entity2(0)(0), expectedEid(2))
    assertEquals(entity2(0)(1), expectedUid(2))
    assertEquals(entity2(0)(2), expectedLabel(2))
    assertEquals(entity2(0)(3), expectedIndices(2))
    assertEquals(entity2(0)(4), expectedValues(2))
  }
}

object DataPartitionerTest {
  val UID = "uid"
  val LABEL = "label"
  val WEIGHT = "weight"
  val ENTITY_ID = "entityId"
}
