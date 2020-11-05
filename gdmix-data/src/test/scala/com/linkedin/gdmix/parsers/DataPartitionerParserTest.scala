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
          "--partitionId", "memberId",
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainingDataDir", "per-member/trainingData",
          "--trainingScoreDir", "global/trainingScore",
          "--partitionedTrainingDataDir", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")))
  }

  @DataProvider
  def dataIncompleteArgs(): Array[Array[Any]] = {

    Array(
      // missing partitionId
      Array(
        Seq(
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainingDataDir", "per-member/trainingData",
          "--trainingScoreDir", "global/trainingScore",
          "--partitionedTrainingDataDir", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // missing metadataFile
      Array(
        Seq(
          "--partitionId", "memberId",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainingDataDir", "per-member/trainingData",
          "--trainingScoreDir", "global/trainingScore",
          "--partitionedTrainingDataDir", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // missing outputMetadataFile
      Array(
        Seq(
          "--partitionId", "memberId",
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--trainingDataDir", "per-member/trainingData",
          "--trainingScoreDir", "global/trainingScore",
          "--partitionedTrainingDataDir", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // missing both trainingDataDir and validationDataDir
      Array(
        Seq(
          "--partitionId", "memberId",
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json")),
      // provide trainingDataDir but no partitionedTrainingDataDir
      Array(
        Seq(
          "--partitionId", "memberId",
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainingDataDir", "per-member/trainingData",
          "--trainingScoreDir", "global/trainingScore",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt")),
      // provide trainingDataDir but no outputPartitionListFile
      Array(
        Seq(
          "--partitionId", "memberId",
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainingDataDir", "per-member/trainingData",
          "--trainingScoreDir", "global/trainingScore",
          "--partitionedTrainingDataDir", "per-member/partition/trainingData")),
      // provide validationDataDir but no partitionedValidationDataDir
      Array(
        Seq(
          "--partitionId", "memberId",
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--validationDataDir", "per-member/validationData")),
      // maxNumOfSamplesPerModel < minNumOfSamplesPerModel
      Array(
        Seq(
          "--partitionId", "memberId",
          "--metadataFile", "per-member/metadata/tensor_metadata.json",
          "--outputMetadataFile", "per-member/partition/metadata/tensor_metadata.json",
          "--trainingDataDir", "per-member/trainingData",
          "--trainingScoreDir", "global/trainingScore",
          "--partitionedTrainingDataDir", "per-member/partition/trainingData",
          "--outputPartitionListFile", "per-member/partition/partitionList.txt",
          "--minNumOfSamplesPerModel", "10",
          "--maxNumOfSamplesPerModel", "1"))
    )
  }

  @Test(dataProvider = "dataCompleteArgs")
  def testParseCompleteArguments(completeArgs: Seq[String]): Unit = {

    val params = DataPartitionerParser.parse(completeArgs)
    val expectedParams = DataPartitionerParams(
      partitionId = "memberId",
      metadataFile = "per-member/metadata/tensor_metadata.json",
      outputMetadataFile = "per-member/partition/metadata/tensor_metadata.json",
      trainingDataDir = Option("per-member/trainingData"),
      trainingScoreDir = Option("global/trainingScore"),
      partitionedTrainingDataDir = Option("per-member/partition/trainingData"),
      outputPartitionListFile = Option("per-member/partition/partitionList.txt")
    )
    assertEquals(params, expectedParams)
  }

  @Test(dataProvider = "dataIncompleteArgs", expectedExceptions = Array(classOf[IllegalArgumentException]))
  def testThrowIllegalArgumentException(inCompleteArgs: Seq[String]): Unit = {
    DataPartitionerParser.parse(inCompleteArgs)
  }
}
