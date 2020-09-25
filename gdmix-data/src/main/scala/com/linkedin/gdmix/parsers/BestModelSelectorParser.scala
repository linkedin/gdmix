package com.linkedin.gdmix.parsers

import com.linkedin.gdmix.utils.Constants._
import com.linkedin.gdmix.utils.IoUtils

/**
 * Parameters for best model selector job.
 */
case class BestModelSelectorParams(
    inputMetricsPaths: Seq[String],
    outputBestModelPath: String,
    evalMetric: String,
    hyperparameters: String,
    inputModelPaths: Option[String] = None,
    outputBestMetricsPath: Option[String] = None,
    copyBestOutput: Boolean = false
)

/**
 * Parser for best model selector job.
 */
object BestModelSelectorParser {
    private val bestModelSelectorParser = new scopt.OptionParser[BestModelSelectorParams](
        "Parsing command line for best model selector job.") {

        opt[String]("inputMetricsPaths").action((x, p) => p.copy(
            inputMetricsPaths = x.split(CONFIG_SPLITTER).map(_.trim)))
        .required
        .text(
            """Required.
                |Input model metric paths, separated by semicolon.""".stripMargin)


        opt[String]("outputBestModelPath").action((x, p) => p.copy(outputBestModelPath = x.trim))
        .required
        .text(
            """Required.
                |Output best model path.""".stripMargin)

        opt[String]("evalMetric").action((x, p) => p.copy(evalMetric = x.trim))
        .required
        .text(
            """Required.
                |Evaluation metric.""".stripMargin)

        opt[String]("hyperparameters").action((x, p) => p.copy(hyperparameters = x.trim))
        .required
        .text(
            """Required.
                |Hyper-parameters of each model encoded in base64.""".stripMargin)

        opt[String]("inputModelPaths").action((x, p) => p.copy(inputModelPaths = if (x.trim.isEmpty) None else Some(x.trim)))
        .optional
        .text(
            """Optional.
                |Input model paths, separated by semicolons..""".stripMargin)

        opt[String]("outputBestMetricsPath").action((x, p) => p.copy(outputBestMetricsPath = if (x.trim.isEmpty) None else Some(x.trim)))
        .optional
        .text(
            """Optional.
                |Path to best model metric.""".stripMargin)

        opt[String]("copyBestOutput").action((x, p) => p.copy(copyBestOutput = x.toLowerCase == "true"))
        .optional
        .text(
            """Optional.
                |Boolean whether to copy the best model.""".stripMargin)

        checkConfig(p =>
            if (p.copyBestOutput) {
                if (IoUtils.isEmptyStr(p.inputModelPaths)){
                    failure("Option --inputModelPaths is required when --copyBestOutput is true.")
                }

                else if (IoUtils.isEmptyStr(p.outputBestMetricsPath)){
                    failure("Option --outputBestMetricsPath is required when --copyBestOutput is true.")
                }
                else success
            }
            else success)
    }

    def parse(args: Seq[String]): BestModelSelectorParams = {
        val emptyBestModelSelectorParams = BestModelSelectorParams(
            inputMetricsPaths = Seq(""),
            outputBestModelPath = "",
            evalMetric = "",
            hyperparameters = ""
        )
        bestModelSelectorParser.parse(args, emptyBestModelSelectorParams) match {
        case Some(params) => params
        case None => throw new IllegalArgumentException(
            s"Parsing the command line arguments failed.\n" +
            s"(${args.mkString(", ")}),\n${bestModelSelectorParser.usage}")
        }
    }
}
