package com.linkedin.gdmix.utils

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

import com.linkedin.gdmix.configs.DataType
import com.linkedin.gdmix.utils.Constants.{NAME, TERM}

/**
 * Helper class to convert NTVs to sparse vectors
 */
object ConversionUtils {

  /**
   * Case class to represent the NameTermValue (NTV)
   *
   * @param name Name of a feature
   * @param term Term of a feature
   * @param value Value of a feature
   */
  case class NameTermValue(name: String, term: String, value: Float)

  /**
   * Case class for SparseVector type
   * @param indices The indices of a sparse vector
   * @param values The values of a sparse vector
   */
  case class SparseVector(indices: Seq[Long], values: Seq[Float])

  /**
   * UDF to get name and term given a row of NTV
   * @return A string of "$name,$term"
   */
  def getNameTermUdf: UserDefinedFunction = udf { r: Row => (r.getAs[String](NAME), r.getAs[String](TERM)) }

  /**
   * UDF to convert a Seq of indices and a Seq of values to a sparse vector
   * @return A sparse with indices and values
   */
  def collectIdValueUdf(sortIndex: Boolean): UserDefinedFunction = udf {
    (indices: Seq[Int], values: Seq[Float]) =>
      val (sortedIndices, sortedValues) = if (sortIndex)
        (indices zip values).sortBy(_._1).unzip
        else
        (indices, values)
    SparseVector(sortedIndices.map(_.toLong), sortedValues)
  }

  /**
   * UDF to convert a Seq of indices and a Seq of values to a dense vector
   * @param length dense vector length
   * @return A dense vector
   */
  def convertToDenseVector(length: Long): UserDefinedFunction = udf {
    (indices: Seq[Int], values: Seq[Float]) =>
      val vector = collection.mutable.Seq.fill(length.toInt)(0.0f)
      (indices zip values).foreach { x =>
        vector(x._1) = x._2
      }
      Seq(vector: _*)
  }

  /**
   * Sort indices, values by indices in ascending order
   */
  def sortIndexUdf: UserDefinedFunction = udf {
    (indices: Seq[Int], values: Seq[Float]) => SparseVector(indices.map(_.toLong), values)
  }

  /**
   * Convert input Config DataType to Spark sql DataType
   */
  final val ConfigDataTypeMap = Map[DataType.DataType, org.apache.spark.sql.types.DataType](
    DataType.byte -> ByteType,
    DataType.double -> DoubleType,
    DataType.float -> FloatType,
    DataType.int -> IntegerType,
    DataType.long -> LongType,
    DataType.string -> StringType
  )

  /**
   * Map Spark sql DataType -> Config DataType
   */
  def mapSparkToConfigDataType(
    sparkType: org.apache.spark.sql.types.DataType
  ): DataType.DataType = sparkType match {
    case ByteType => DataType.byte
    case DoubleType => DataType.double
    case FloatType => DataType.float
    case IntegerType => DataType.int
    case LongType => DataType.long
    case StringType => DataType.string
  }
}
