package com.linkedin.gdmix.evaluation

import com.databricks.spark.avro._
import org.apache.commons.cli.{BasicParser, CommandLine, CommandLineParser, Options}
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.mapred.JobConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.{IoUtils, JsonUtils}

/**
 * Evaluator for NDCG.
 */
object NdcgEvaluator {

  // Create a Spark session.
  val spark: SparkSession = SparkSession
    .builder()
    .appName(getClass.getName)
    .getOrCreate()

  // Set up Hadoop file system.
  val hadoopJobConf = new JobConf()
  val fs: FileSystem = FileSystem.get(hadoopJobConf)

  /**
   * Compute the average NDCG value of all the queries, truncated at ranking position k. This function implements two
   * formulas: (1) traditional: sum_i=1..n rel(i)/log_2(i+1) and
   * (2) non-traditional: sum_i=1..n (pow(2,rel(i)) - 1)/log_2(i+1), determined by a Boolean argument.
   *
   * @param scoresAndLabels Prediction scores and labels
   * @param k Position at which to truncate this ranking
   * @param isTraditional Whether to use the traditional formula or not
   * @return The average NDCG at the first k ranking positions.
   */
  def calculateNdcgAt(
    scoresAndLabels: RDD[(Seq[Double], Seq[Double])],
    k: Int,
    isTraditional: Boolean): Double = {

    require(k > 0, "In NDCG calculation, k must be positive")

    scoresAndLabels
      .map { case (unsortedScores, unsortedLabels) =>
        if (unsortedScores.isEmpty) {
          throw new Exception("Cannot compute NDCG for empty data")
        }
        if (unsortedScores.length > unsortedLabels.length) {
          throw new Exception("Length of scores must be less than or equal to the length of the labels")
        }

        // Sort the scores and labels by scores in descending order.
        val (scores, labels) = unsortedScores.zip(unsortedLabels).sortBy(_._1)(Ordering[Double].reverse).unzip

        // Normalized DCG is the actual DCG normalized by the DCG if the predicted labels were sorted
        // in the same order as the actual labels
        var dcg = 0.0 // DCG is computed using the sort order from the predicted scores
        var idealDcg = 0.0
        val sortedLabels = labels.sortBy(-_) // For idealized DCG re-sort using the actual labels

        // Truncate to to top k results. We process ideal DCG up to the number of labels, but for DCG
        // we additionally truncate DCG up to the number of predictions.
        val n = math.min(labels.length, k)

        // Right now we'll use b = 2 as the base of the log in the computation
        val b = 2.0
        val logb = math.log(b)

        for (i <- 0 until n) {
          val discount = math.log((i + 1.0) + b - 1.0) / logb
          val gain = if (i >= scores.length) {
            0.0
          } else {
            if (isTraditional) labels(i) / discount else (math.pow(2.0, labels(i)) - 1.0) / discount
          }
          val idealGain = if (isTraditional) {
            sortedLabels(i) / discount
          } else {
            (math.pow(2.0, sortedLabels(i)) - 1.0) / discount
          }
          dcg += gain
          idealDcg += idealGain
        }

        dcg / idealDcg
      }
      .mean()
  }

  def main(args: Array[String]): Unit = {

    // Define options.
    val options = new Options()
    options.addOption("inputPath", true, "input score path")
    options.addOption("outputPath", true, "output path")
    options.addOption("labelName", true, "ground truth label name")
    options.addOption("scoreName", true, "predicted score name")
    options.addOption("positionK", true, "the position to compute the truncated ndcg")
    options.addOption("isTraditional", true, "whether to use the traditional ndcg formula")

    // Get the parser.
    val parser: CommandLineParser = new BasicParser()
    val cmd: CommandLine = parser.parse(options, args)

    // Parse the commandline option.
    val inputPath = cmd.getOptionValue("inputPath")
    val outputPath = cmd.getOptionValue("outputPath")
    val labelName = cmd.getOptionValue("labelName")
    val scoreName = cmd.getOptionValue("scoreName")
    val positionK = cmd.getOptionValue("positionK")
    val isTraditional = cmd.getOptionValue("isTraditional", "true").toBoolean

    // Sanity check.
    require(inputPath != null
      && outputPath != null
      && labelName != null
      && scoreName != null
      && positionK != null,
      "Incorrect number of input parameters")

    // Read file and get scores and labels.
    val dataFrame = spark.read.avro(inputPath)
    val scoresAndLabels = dataFrame.rdd.map(
      row => (row.getAs[Seq[Float]](scoreName).map(_.toDouble), row.getAs[Seq[Float]](labelName).map(_.toDouble)))

    // Compute NDCG.
    val ndcg = calculateNdcgAt(scoresAndLabels, positionK.toInt, isTraditional)

    // Convert to json and save to HDFS.
    val jsonResult = JsonUtils.toJsonString(Map(s"ndcgAt$positionK" -> ndcg))
    IoUtils.writeFile(fs, new Path(outputPath, EVAL_SUMMARY_JSON), jsonResult)
  }
}
