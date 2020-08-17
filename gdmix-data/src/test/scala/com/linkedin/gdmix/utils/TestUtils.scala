package com.linkedin.gdmix.utils

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.ConversionUtils.NameTermValue

/**
 * Helper functions for unit test in data module.
 */
object TestUtils {

  /**
   * UDF to mock NTVs for unit tests
   *
   * @param k The integer to mod uid to generate name, term and value
   * @return An array of NTVs
   */
  def mockNtvUdf(k: Int): UserDefinedFunction = udf {
    uid: Long => {
      val name1 = (uid % k).toString
      val term1 = (uid % k).toString
      val value1 = (uid % k).toFloat
      val name2 = (uid % k + 1).toString
      val term2 = (uid % k + 1).toString
      val value2 = (uid % k + 1).toFloat
      Seq(NameTermValue(name1, term1, value1), NameTermValue(name2, term2, value2))
    }
  }

  /**
   * UDF to get the indices given the feature bag
   *
   * @return An array of indices
   */
  def getIndicesUdf: UserDefinedFunction = udf {
    r: Row => r.getAs[Seq[Long]](INDICES)
  }

  /**
   * Check if two small dataframes have equal content
   * @param df1 The first dataframe
   * @param df2 The second dataframe
   * @param sortedBy The column name by which the dataframes to be sorted
   * @return true of false
   */
  def equalSmallDataFrame(df1: DataFrame, df2: DataFrame, sortedBy: String): Boolean = {
    val sdf1 = df1.sort(sortedBy)
    val sdf2 = df2.sort(sortedBy)
    (sdf1.collect().sameElements(sdf2.collect())
      && sdf1.schema.equals(sdf2.schema))
  }

  /**
   * Remove the whitespace from a string.
   *
   * @param s - the string
   * @return the string with whitespaces removed
   */
  def removeWhiteSpace(s: String): String = s.replaceAll("\\s", "")
}
