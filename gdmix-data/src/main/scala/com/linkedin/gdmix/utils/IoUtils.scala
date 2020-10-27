package com.linkedin.gdmix.utils

import java.io.{
  BufferedWriter, File,
  FileInputStream, OutputStreamWriter, PrintWriter
}

import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.StructType

import com.databricks.spark.avro._

import com.linkedin.gdmix.utils.Constants.{
  AVRO, AVRO_FORMAT,
  TFRECORD, TF_EXAMPLE, TF_SEQUENCE_EXAMPLE
}

/**
 * Helper routines for reading and writing files on HDFS.
 */
object IoUtils {

  /**
   * Helper method to get resource file path.
   *
   * @param filePath The relative path of the resource file
   * @return The absolution path of the input
   */
  def getResourceFilePath(filePath: String): String = {
    new File(
      getClass.getClassLoader.getResource(filePath).getFile
    ).getAbsolutePath
  }

  /**
   * Helper method to read a file from HDFS as text.
   *
   * @param fs The Hadoop file system or null if local resource directly is used.
   * @param inputPath The HDFS input path from which to read the file
   * @return The contents of the file as text
   */
  def readFile(fs: FileSystem, inputPath: String, fromResource: Boolean = false): String = {

    val stream = if (fromResource)
      getClass.getClassLoader.getResourceAsStream(inputPath)
    else if (fs == null)
      new FileInputStream(new File(inputPath))
    else
      fs.open(new Path(inputPath))
    try {
      scala.io.Source.fromInputStream(stream).mkString
    } finally {
      stream.close()
    }
  }

  /**
   * Helper method to write the text to a file in HDFS.
   *
   * @param fs The Hadoop file system
   * @param outputPath The output file path
   * @param text The text to be written to HDFS
   */
  def writeFile(fs: FileSystem, outputPath: Path, text: String) {

    if (fs.exists(outputPath)) {
      fs.delete(outputPath, false)
    }

    val outputStream = fs.create(outputPath)
    val writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(outputStream)))

    try {
      writer.write(text)
    } finally {
      writer.close()
    }
  }

  /**
   * Get the number of features in the feature list. Plus one to the feature list for unknown features.
   *
   * @param fs The Hadoop file system
   * @param featureListPath The HDFS input path to the feature list
   * @return The number of features
   */
  def readNumFeatures(fs: FileSystem, featureListPath: String): Int = {
    val featureListString = readFile(fs, featureListPath)
    val numFeatures = featureListString.split("\n").length

    numFeatures
  }

  /**
   * Copy directory from src to dst on HDFS.
   *
   * @param fs The Hadoop file system
   * @param conf The Hadoop job configuration
   * @param src Source path
   * @param dst Destination path
   */
  def copyDirectory(fs: FileSystem, conf: JobConf, src: String, dst: String): Unit = {
    val srcPath = new Path(src)
    val dstPath = new Path(dst)
    fs.delete(dstPath, true)
    FileUtil.copy(fs, srcPath, fs, dstPath, false, conf)
  }

  /**
   * Save data frame to HDFS, partition if requested.
   *
   * @param dataFrame The output dataframe
   * @param outputDir The output directory
   * @param outputFormat The saved file format
   * @param numPartitions Number of partitions if requested
   * @param partitionColumn The column name to be partitioned by
   * @param recordType The TFRecord type, Example or SequenceExample,
   *                   used in TFRecord only.
   */
  def saveDataFrame(
    dataFrame: DataFrame,
    outputDir: String,
    outputFormat: String,
    numPartitions: Int = 0,
    partitionColumn: String = null,
    recordType: String = TF_EXAMPLE
  ): Unit = {
    val dataFrameWriter = if ((numPartitions > 0)
      && (partitionColumn != null)) {
      dataFrame
        .repartition(numPartitions, col(partitionColumn))
        .write
        .mode(SaveMode.Overwrite)
        .partitionBy(partitionColumn)
    } else {
      dataFrame.write.mode(SaveMode.Overwrite)
    }
    outputFormat match {
      case AVRO =>
        dataFrameWriter
          .format(AVRO_FORMAT)
          .save(outputDir)
      case TFRECORD =>
        dataFrameWriter
          .format(TFRECORD)
          .option("recordType", recordType)
          .save(outputDir)
      case _ =>
        throw new IllegalArgumentException(s"Unknown format $outputFormat, " +
          s"use avro or tfrecord only")
    }
  }

  /**
   * Read dataframe by Spark, support AVRO and TFRECORD formats.
   *
   * @param spark Spark session
   * @param inputPath Path for the input files
   * @param inputFormat Input format, either AVRO or TFRECORD
   * @param schemaOpt Schema for TFRecord, optional.
   * @param recordType The TFRecord type, Example or SequenceExample,
   *                   used in TFRecord only.
   * @return the read dataframe
   */
  def readDataFrame(
    spark: SparkSession,
    inputPath: String,
    inputFormat: String,
    schemaOpt: Option[StructType] = None,
    recordType: String = TF_EXAMPLE
  ): DataFrame = {
    inputFormat match {
      case AVRO => spark.read.avro(inputPath)
      case TFRECORD =>
        val reader = spark.read.format(TFRECORD).option("recordType", recordType)
        schemaOpt match {
          case Some(schema) => reader.schema(schema).load(inputPath)
          case _ => reader.load(inputPath)
        }
      case _ =>
        throw new IllegalArgumentException(s"Unknown format $inputFormat, " +
          s"use avro or tfrecord only")
    }
  }

  /**
   * Check if a Option[string] is a non-empty string.
   *
   * @param str Option string
   * @return Boolean, true if it's a non-empty string
   */
  def isEmptyStr(str: Option[String]): Boolean =
    str match {
      case Some(s) => s.trim.isEmpty
      case _ => true
    }
}
