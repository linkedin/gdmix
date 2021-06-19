package com.linkedin.gdmix.utils

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.ConversionUtils.NameTermValue

/**
 * Helper functions for unit test in data module.
 */
object TestUtils {

  /**
   * Sort the columns of a DataFrame
   * if the original columns are ["c", "b", "a"],
   * the sorted columns are ["a", "b", "c"]
   * @param df: input DataFrame
   * @return: the sorted DataFrame
   */
  def sortColumns(df: DataFrame): DataFrame = {
    val sortedColumns = df.columns.sorted.map(str => col(str))
    df.select(sortedColumns:_*)
  }

  /**
   * Check if two small dataframes have equal content
   * @param df1 The first dataframe
   * @param df2 The second dataframe
   * @param sortedBy The column name by which the dataframes to be sorted
   * @return true of false
   */
  def equalSmallDataFrame(df1: DataFrame, df2: DataFrame, sortedBy: String): Boolean = {
    // sort the rows
    val sdf1 = df1.sort(sortedBy)
    val sdf2 = df2.sort(sortedBy)

    // sort the columns
    val ssdf1 = sortColumns(sdf1)
    val ssdf2 = sortColumns(sdf2)

    (ssdf1.schema.equals(ssdf2.schema)
      && ssdf1.collect().sameElements(ssdf2.collect()))
  }

  /**
   * Remove the whitespace from a string.
   *
   * @param s - the string
   * @return the string with whitespaces removed
   */
  def removeWhiteSpace(s: String): String = s.replaceAll("\\s", "")
}
