package com.linkedin.gdmix.parsers

import org.testng.annotations.{DataProvider, Test}
import org.testng.Assert.assertEquals

/**
 * Unit tests for DataPartitionerParser.
 */
class DataPartitionerParserTest {

  @DataProvider
  def dataCompleteArgs(): Array[Array[Any]] = {
    Array(
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--trainOutputPartitionDataPath", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // missing partitionEntity
      Array(
        Seq(
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--trainOutputPartitionDataPath", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // missing fetureBag
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--trainOutputPartitionDataPath", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // missing inputMetadataFile
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--trainOutputPartitionDataPath", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // missing outputMetadataFile
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--trainOutputPartitionDataPath", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // missing both trainInputDataPath and validationInputDataPath
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json")),
      // provide trainInputDataPath but no trainOutputPartitionDataPath
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // provide trainInputDataPath but no outputPartitionListFile
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--trainOutputPartitionDataPath", "per-member/partition/trainingData")),
      // provide validationInputDataPath but no validationOutputPartitionDataPath
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--validationInputDataPath", "per-member/validationData")),
      // maxNumOfSamplesPerModel < minNumOfSamplesPerModel
      Array(
        Seq(
          "--partitionEntity", "memberId",
          "--featureBag", "per_member",
          "--inputMetadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainInputDataPath", "per-member/trainingData",
          "--trainInputScorePath", "global/trainingScore",
          "--trainOutputPartitionDataPath", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt",
          "--minNumOfSamplesPerModel", "10",
          "--maxNumOfSamplesPerModel", "1"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = DataPartitionerParser.parse(completeArgs)
    val expectedParams = DataPartitionerParams(
      partitionEntity = "memberId",
      featureBag = "per_member",
      inputMetadataFile = "per-member/metadata/tensor_metadata.json",
      outputMetadataFile = "per-member/partition/metadata/tensor_metadata.json",
      trainInputDataPath = Option("per-member/trainingData"),
      trainInputScorePath = Option("global/trainingScore"),
      trainOutputPartitionDataPath = Option("per-member/partition/trainingData"),
      outputPartitionListFile = Option("per-member/partition/partitionList.txt")
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
    DataPartitionerParser.parse(inCompleteArgs)
  }
}
