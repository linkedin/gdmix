package com.linkedin.gdmix.utils

import org.apache.spark.sql.Row
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._

import com.linkedin.gdmix.configs.DataType
import com.linkedin.gdmix.utils.Constants.{NAME, TERM, VALUE, CROSS}

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
   * Case class to represent the NameTermValue (NTV) where the value is float (nullable)
   *
   * @param name Name of a feature
   * @param term Term of a feature
   * @param value Value of a feature (Option[Float])
   */
  case class NameTermValueOptionFloat(name: String, term: String, value: Option[Float])

  /**
   * Case class to represent the NameTermValue (NTV) where the value is double (nullable)
   * Photon-ML generates model values are in double format and nullable.
   *
   * @param name Name of a feature
   * @param term Term of a feature
   * @param value Value of a feature (Option[Double])
   */
  case class NameTermValueOptionDouble(name: String, term: String, value: Option[Double])

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
   *  Split the full name into (model_id, feature_name) tuple
   */
  def splitModelIdUdf: UserDefinedFunction = udf { r: Row =>
    val Array(modelId, name) = r.getAs[String](NAME).split(CROSS)
    val term = r.getAs[String](TERM)
    val value = r.getAs[Double](VALUE)
    (modelId, NameTermValueOptionDouble(name, term, Some(value)))
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
