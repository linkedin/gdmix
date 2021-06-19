package com.linkedin.gdmix.model

import org.apache.avro.Schema
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import com.databricks.spark.avro._

import com.linkedin.gdmix.parsers.LrModelSplitterParser
import com.linkedin.gdmix.parsers.LrModelSplitterParams
import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.ConversionUtils.{NameTermValueOptionDouble, splitModelIdUdf}

import java.io.File
import scala.util.{Success, Try}

/**
 * Split crossed global model to multiple random effect models.
 * The input model file contains a single model with crossed feature names, e.g.
 * model_1_gdmixcross_feature_1, model_1_gdmixcross_feature_2,
 * model_2_gdmixcross_feature_3, model_2_gdmixcross_feature_4,
 * model_3_gdmixcross_feature_5, model_3_gdmixcross_feature_6
 *
 * The result mode files contain the following models:
 * model_1:
 *     feature_1
 *     feature_2
 * model_2:
 *     feature_3
 *     feature_4
 * model_3:
 *     feature_5
 *     feature_6
 */
object LrModelSplitter {

  val LR_MODEL_SCHEMA_FILE = "model/lr_model.avsc"

  def main(args: Array[String]): Unit = {

    val params = LrModelSplitterParser.parse(args)

    // Create a Spark session.
    val spark = SparkSession.builder().appName(getClass.getName).getOrCreate()
    try {
      run(spark, params)
    } finally {
      spark.stop()
    }
  }

  def run(spark: SparkSession, params: LrModelSplitterParams): Unit = {

    // Parse the commandline option.
    val modelInputDir = params.modelInputDir
    val modelOutputDir = params.modelOutputDir
    val numOutputFiles = params.numOutputFiles

    val df = spark.read.avro(modelInputDir)
    val means = splitModelId(MEANS, df)
    val hasVariances = Try(df.first().getAs[Seq[NameTermValueOptionDouble]](VARIANCES)) match {
      case Success(value) if value != null => true
      case _ => false
    }

    // append variances column
    val meansAndVariances = if (hasVariances) {
      val variances = splitModelId(VARIANCES, df)
      means.join(variances, MODEL_ID)
    } else {
      means.withColumn(VARIANCES, typedLit[Option[NameTermValueOptionDouble]](None))
    }

    // append other columns
    val outDf = meansAndVariances
      .withColumn("modelClass", typedLit[String](LR_MODEL_CLASS))
      .withColumn("lossFunction", typedLit[String](""))

    val schema = new Schema.Parser().parse(
      getClass.getClassLoader.getResourceAsStream(LR_MODEL_SCHEMA_FILE))

    outDf.repartition(numOutputFiles).write.option("avroSchema", schema.toString)
      .mode(SaveMode.Overwrite).format(AVRO_FORMAT).save(modelOutputDir)
  }

  /**
   * Separate the model Id from feature names.
   * Break a single model into multiple smaller models identified by their model Ids.
   * @param colName: the name of the column that has all coefficients of the global model
   * @param df: the input dataframe.
   * @return: a dataframe where each row is the coefficients of a separated model.
   */
  private[model] def splitModelId(colName: String, df: DataFrame): DataFrame = {
    df.select(explode(col(colName)).alias("explodeCol"))
      .withColumn("splitCol", splitModelIdUdf(col("explodeCol")))
      .select("splitCol.*")
      .withColumnRenamed("_1", MODEL_ID)
      .withColumnRenamed("_2", colName)
      .groupBy(MODEL_ID)
      .agg(collect_list(col(colName)).alias(colName))
  }
}
