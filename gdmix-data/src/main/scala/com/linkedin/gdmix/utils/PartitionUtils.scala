package com.linkedin.gdmix.utils

import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.{col, udf}

import com.linkedin.gdmix.utils.Constants._

/**
 * Helper class to partition data in Spark data frame
 */
object PartitionUtils {

  /**
   * UDF to add offset to each value in a sequence
   */
  def addOffsetUDF: UserDefinedFunction = {
    udf{
      (indices: Seq[Long], offset: Int) => {
        indices.map(index => index + offset)
      }
    }
  }

  /**
   * UDF to get partition id by (hash(item id) % number of partitions)
   *
   * @param numPartitions Number of partitions
   * @return A UDF to get the partition id.
   */
  def getPartitionIdUDF(numPartitions: Int): UserDefinedFunction = {
    udf {
      itemId: String => {
        Math.abs(itemId.hashCode) % numPartitions
      }
    }
  }
}
