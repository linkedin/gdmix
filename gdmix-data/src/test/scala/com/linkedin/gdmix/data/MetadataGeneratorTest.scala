package com.linkedin.gdmix.data

import scala.collection.mutable

import org.apache.spark.sql.types.{ArrayType, FloatType, IntegerType, LongType, StructField, StructType}
import org.testng.Assert.{assertEquals, assertFalse, assertTrue}
import org.testng.annotations.Test

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.configs.{DataType, TensorMetadata}
import com.linkedin.gdmix.utils.IoUtils.readFile

/**
 * Unit tests for [[MetadataGenerator]].
 */

class MetadataGeneratorTest {

  val weight = TensorMetadata("weight", DataType.float, Seq(), false)
  val count = TensorMetadata("count", DataType.int, Seq(), false)
  val label = TensorMetadata("label", DataType.int, Seq(), false)
  val global = TensorMetadata("global", DataType.float, Seq(), true)
  val inputFeatureColumns = Seq(weight, count, global)
  val inputLabelColumns = Seq(label)
  val inputMap = Map(
    "weight" -> weight,
    "count" -> count,
    "global" -> global,
    "label" -> label)
  val inputMapNoLabel = Map(
    "weight" -> weight,
    "count" -> count,
    "global" -> global)
  val schema =  StructType(
    Seq(
      StructField("label", IntegerType, true),
      StructField("count", IntegerType, true),
      StructField("global",
        StructType(
          Seq(
            StructField("indices", ArrayType(LongType, true), true),
            StructField("values", ArrayType(FloatType, true), true))
        ), true),
      StructField("weight", FloatType, true)
    )
  )

  /**
   * Unit test for [[MetadataGenerator.buildColumnMap]].
   */
  @Test
  def testBuildColumnMap(): Unit = {
    val outputMap = MetadataGenerator.buildColumnMap(inputFeatureColumns, Some(inputLabelColumns))
    assertEquals(inputMap, outputMap)
    val outputMapNoLabel = MetadataGenerator.buildColumnMap(inputFeatureColumns)
    assertEquals(inputMapNoLabel, outputMapNoLabel)
  }

  /**
   * Unit test for [[MetadataGenerator.getFeatureAndLabelColumns]].
   */
  @Test
  def testGetFeatureAndLabelColumns(): Unit = {
    val metadataJsonFile = "metadata/ExpectedGlobalMetadata.json"
    val metadataJson = readFile(null, metadataJsonFile, true)
    val (featureCols, labelColsOpt) = MetadataGenerator.getFeatureAndLabelColumns(metadataJson)

    // Check the number of TensorMetadata objects
    assertEquals(featureCols.size, 2)
    val labelCols = labelColsOpt.get
    assertEquals(labelCols.size, 1)

    // Check the fields of TensorMetadata objects
    val tensorMetadataGlobal = featureCols(0)
    val tensorMetadataLabel = labelCols(0)
    assertEquals(tensorMetadataGlobal.name, "global")
    assertEquals(tensorMetadataGlobal.dtype, DataType.float)
    assertEquals(tensorMetadataGlobal.shape, Seq(3))
    assertEquals(tensorMetadataGlobal.isSparse, true)
    assertEquals(tensorMetadataLabel.name, "response")
    assertEquals(tensorMetadataLabel.dtype, DataType.float)
    assertEquals(tensorMetadataLabel.shape, Seq())
    assertEquals(tensorMetadataLabel.isSparse, false)
  }

  /**
   * Unit test for [[MetadataGenerator.isSimpleColumn]].
   */
  @Test
  def testIsSimpleColumn(): Unit = {
    assertTrue(MetadataGenerator.isSimpleColumn(schema("label")))
    assertTrue(MetadataGenerator.isSimpleColumn(schema("count")))
    assertTrue(MetadataGenerator.isSimpleColumn(schema("weight")))
    assertFalse(MetadataGenerator.isSimpleColumn(schema("global")))
  }

  /**
   * Unit test for [[MetadataGenerator.isSimpleArrayTypeColumn]].
   */
  @Test
  def testIsSimpleArrayTypeColumn(): Unit = {
    assertFalse(MetadataGenerator.isSimpleArrayTypeColumn(schema("label")))
    assertFalse(MetadataGenerator.isSimpleArrayTypeColumn(schema("count")))
    assertFalse(MetadataGenerator.isSimpleArrayTypeColumn(schema("weight")))
    assertFalse(MetadataGenerator.isSimpleArrayTypeColumn(schema("global")))
    val indices = schema("global").dataType.asInstanceOf[StructType]("indices")
    val values = schema("global").dataType.asInstanceOf[StructType]("values")
    assertTrue(MetadataGenerator.isSimpleArrayTypeColumn(indices))
    assertTrue(MetadataGenerator.isSimpleArrayTypeColumn(values))
  }

  /**
   * Unit test for [[MetadataGenerator.appendNewColumns]].
   * Add a simple column
   */
  @Test
  def testAppendSimpleColumns(): Unit = {
    val schema_freq =  StructType(
      Seq(
        StructField("label", IntegerType, true),
        StructField("count", IntegerType, true),
        StructField("global",
          StructType(
            Seq(
              StructField("indices", ArrayType(LongType, true), true),
              StructField("values", ArrayType(FloatType, true), true))
          ), true),
        StructField("weight", FloatType, true),
        StructField("freq", FloatType, true),
        StructField("global_indices", IntegerType, true),
        StructField("global_values", FloatType, true),
        StructField("random_values", FloatType, true),
        StructField("float_array", ArrayType(FloatType, true), true)
      )
    )
    val inputArray = mutable.ArrayBuffer(inputFeatureColumns: _*)
    MetadataGenerator.appendNewColumns(schema_freq, inputMap, inputArray, TFRECORD)
    val freqColumn = TensorMetadata(
      "freq",
      DataType.float,
      Seq(),
      false)
    val randomColumn = TensorMetadata(
      "random_values",
      DataType.float,
      Seq(),
      false)
    val floatColumn = TensorMetadata(
      "float_array",
      DataType.float,
      Seq(),
      false)
    val outputArray = inputFeatureColumns ++ Seq(freqColumn, randomColumn, floatColumn)
    assertEquals(inputArray, outputArray)
  }

  /**
   * Unit test for [[MetadataGenerator.appendNewColumns]].
   * Add a complex column
   */
  @Test(expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testAppendComplexColumns(): Unit = {
    val schema_freq =  StructType(
      Seq(
      StructField("label", IntegerType, true),
      StructField("count", IntegerType, true),
      StructField("global", StructType(
        Seq(StructField("indices", ArrayType(LongType, true), true),
        StructField("values", ArrayType(FloatType, true), true))), true
      ),
      StructField("weight", FloatType, true),
      StructField("freq", StructType(
        Seq(
          StructField("indices", ArrayType(LongType, true), true),
        StructField("values", ArrayType(FloatType, true), true))), true)
      )
    )
    val inputArray = mutable.ArrayBuffer(inputFeatureColumns: _*)
    MetadataGenerator.appendNewColumns(schema_freq, inputMap, inputArray, AVRO)
  }

  /**
   * Unit test for [[MetadataGenerator.isSparseColumnComponent]].
   */
  @Test
  def testIsSparseColumnComponent(): Unit = {
    assertTrue(MetadataGenerator
      .isSparseColumnComponent(inputMap, "global_indices"))
    assertTrue(MetadataGenerator
      .isSparseColumnComponent(inputMap, "global_values"))
    assertFalse(MetadataGenerator
      .isSparseColumnComponent(inputMap, "xyz_indices"))
    assertFalse(MetadataGenerator
      .isSparseColumnComponent(inputMap, "xyz_values"))
    assertFalse(MetadataGenerator
      .isSparseColumnComponent(inputMap, "xyz"))
    assertFalse(MetadataGenerator
      .isSparseColumnComponent(inputMap, ""))
  }
}
