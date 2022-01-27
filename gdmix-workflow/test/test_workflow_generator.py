import os
import unittest
from copy import deepcopy
from dataclasses import replace
from os.path import join as path_join

from detext.run_detext import DetextArg
from gdmix.models.custom.fixed_effect_lr_lbfgs_model import FixedLRParams
from gdmix.models.custom.random_effect_lr_lbfgs_model import REParams
from gdmix.params import Params
from gdmix.util.constants import ACTION_INFERENCE

from gdmixworkflow.common.utils import yaml_config_file_to_obj
from gdmixworkflow.fixed_effect_workflow_generator import FixedEffectWorkflowGenerator
from gdmixworkflow.random_effect_workflow_generator import RandomEffectWorkflowGenerator
from gdmixworkflow.single_node.local_ops import get_param_list


class TestGDMixWorkflowGenerator(unittest.TestCase):
    """
    Test gdmix workflow generator
    """

    def setUp(self):
        lr_config_file = path_join(
            os.getcwd(),
            "test/resources/lr-movieLens.yaml")
        self.lr_config_obj = yaml_config_file_to_obj(lr_config_file)

        detext_config_file = path_join(
            os.getcwd(),
            "test/resources/detext-movieLens.yaml")
        self.detext_config_obj = yaml_config_file_to_obj(detext_config_file)

        # Set self.maxDiff to None to see diff for long text
        self.maxDiff = None

    def test_lr_model_fixed_effect_workflow_generator(self):
        fe_workflow = FixedEffectWorkflowGenerator(self.lr_config_obj)
        # check sequence
        seq = fe_workflow.get_job_sequence()
        self.assertEqual(len(seq), 2)
        # check job properties
        actual_train_job = seq[0]
        actual_compute_metric_job = seq[1]
        expected_train_job = (
            'gdmix_tfjob',
            'global-tf-train',
            '',
            ({'uid_column_name': 'uid',
              'weight_column_name': 'weight',
              'label_column_name': 'response',
              'prediction_score_column_name': 'predictionScore',
              'prediction_score_per_coordinate_column_name': 'predictionScorePerCoordinate',
              'action': 'train',
              'stage': 'fixed_effect',
              'model_type': 'logistic_regression',
              'training_score_dir': 'lr-training/global/training_scores',
              'validation_score_dir': 'lr-training/global/validation_scores',
              'partition_list_file': None,
              '__frozen__': True},
             {'training_data_dir': 'movieLens/global/trainingData',
              'validation_data_dir': 'movieLens/global/validationData',
              'feature_file': 'movieLens/global/featureList/global',
              'feature_bag': 'global',
              'metadata_file': 'movieLens/global/metadata/tensor_metadata.json',
              'l2_reg_weight': 1.0,
              'regularize_bias': False,
              'lbfgs_tolerance': 1e-12,
              'num_of_lbfgs_iterations': 100,
              'num_of_lbfgs_curvature_pairs': 10,
              'copy_to_local': False,
              'output_model_dir': 'lr-training/global/models'}))
        self.assertEqual(expected_train_job, actual_train_job)

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.Evaluator',
            {'\\--metricsInputDir': 'lr-training/global/validation_scores',
             '--outputMetricFile': 'lr-training/global/metric',
             '--labelColumnName': 'response',
             '--metricName': 'auc',
             '--predictionColumnName': 'predictionScore'})
        self.assertEqual(actual_compute_metric_job, expected_compute_metric_job)

    def test_detext_model_fixed_effect_workflow_generator(self):
        # return  # skip test for now
        fe_workflow = FixedEffectWorkflowGenerator(self.detext_config_obj)
        # check sequence
        seq = fe_workflow.get_job_sequence()
        self.assertEqual(len(seq), 3)
        # check job properties
        actual_train_job = seq[0]
        actual_inference_job = seq[1]
        actual_compute_metric_job = seq[2]

        expected_train_job_param = (
            ({'uid_column_name': 'uid',
              'weight_column_name': 'weight',
              'label_column_name': 'response',
              'prediction_score_column_name': 'predictionScore',
              'prediction_score_per_coordinate_column_name': 'predictionScorePerCoordinate',
              'action': 'train',
              'stage': 'fixed_effect',
              'model_type': 'detext',
              'training_score_dir': 'detext-training/global/training_scores',
              'validation_score_dir': 'detext-training/global/validation_scores',
              'partition_list_file': None,
              '__frozen__': True},
             {'ftr_ext': 'cnn',
              'ltr_loss_fn': 'pointwise',
              'learning_rate': 0.002,
              'num_classes': 1,
              'max_len': 16,
              'min_len': 3,
              'num_filters': 50,
              'num_train_steps': 1000,
              'num_units': 64,
              'optimizer': 'adamw',
              'pmetric': 'auc',
              'steps_per_stats': 10,
              'steps_per_eval': 100,
              'train_batch_size': 64,
              'test_batch_size': 64,
              'vocab_file': 'movieLens/detext/vocab.txt',
              'resume_training': False,
              'train_file': 'movieLens/detext/trainingData',
              'dev_file': 'movieLens/detext/validationData',
              'test_file': 'movieLens/detext/validationData',
              'keep_checkpoint_max': 1,
              'distribution_strategy': 'one_device',
              'task_type': 'binary_classification',
              'sparse_ftrs_column_names': 'wide_ftrs_sp',
              'doc_text_column_names': 'doc_query',
              'nums_sparse_ftrs': 100,
              'num_gpu': 0,
              'out_dir': 'detext-training/global/models'}))

        expected_train_job = ('gdmix_tfjob', 'global-tf-train', '', expected_train_job_param)
        self.assertEqual(expected_train_job, actual_train_job)
        expected_inference_job_param = deepcopy(expected_train_job_param)
        expected_inference_job_param[0]["action"] = ACTION_INFERENCE
        expected_inference_job = 'gdmix_tfjob', 'global-tf-inference', '', expected_inference_job_param
        self.assertEqual(expected_inference_job, actual_inference_job)

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.Evaluator',
            {'\\--metricsInputDir': 'detext-training/global/validation_scores',
             '--outputMetricFile': 'detext-training/global/metric',
             '--labelColumnName': 'response',
             '--metricName': 'auc',
             '--predictionColumnName': 'predictionScore'})
        self.assertEqual(actual_compute_metric_job, expected_compute_metric_job)

    def test_lr_model_random_effect_workflow_generator(self):
        re_workflow = RandomEffectWorkflowGenerator(self.lr_config_obj, prev_model_name='global')
        # check sequence
        seq = re_workflow.get_job_sequence()
        self.assertEqual(len(seq), 6)
        # check job properties
        actual_partition_job = seq[0]
        actual_train_job = seq[1]
        actual_compute_metric_job = seq[2]

        expected_partition_job = (
            'gdmix_sparkjob',
            'per-user-partition',
            'com.linkedin.gdmix.data.DataPartitioner',
            {'\\--trainingDataDir': 'movieLens/per_user/trainingData',
             '--validationDataDir': 'movieLens/per_user/validationData',
             '--metadataFile': 'movieLens/per_user/metadata/tensor_metadata.json',
             '--partitionId': 'user_id',
             '--numPartitions': 1,
             '--dataFormat': 'tfrecord',
             '--partitionedTrainingDataDir': 'lr-training/per-user/partition/trainingData',
             '--partitionedValidationDataDir': 'lr-training/per-user/partition/validationData',
             '--outputMetadataFile': 'lr-training/per-user/partition/metadata/tensor_metadata.json',
             '--outputPartitionListFile': 'lr-training/per-user/partition/partitionList.txt',
             '--predictionScoreColumnName': 'predictionScore',
             '--trainingScoreDir': 'lr-training/global/training_scores',
             '--validationScoreDir': 'lr-training/global/validation_scores'})

        expected_train_job = (
            'gdmix_tfjob',
            'per-user-tf-train',
            '',
            ({'uid_column_name': 'uid',
              'weight_column_name': 'weight',
              'label_column_name': 'response',
              'prediction_score_column_name': 'predictionScore',
              'prediction_score_per_coordinate_column_name': 'predictionScorePerCoordinate',
              'action': 'train',
              'stage': 'random_effect',
              'model_type': 'logistic_regression',
              'training_score_dir': 'lr-training/per-user/training_scores',
              'validation_score_dir': 'lr-training/per-user/validation_scores',
              'partition_list_file': 'lr-training/per-user/partition/partitionList.txt',
              '__frozen__': True},
             {'metadata_file': 'lr-training/per-user/partition/metadata/tensor_metadata.json',
              'output_model_dir': 'lr-training/per-user/models',
              'training_data_dir': 'lr-training/per-user/partition/trainingData',
              'validation_data_dir': 'lr-training/per-user/partition/validationData',
              'feature_bag': 'per_user',
              'feature_file': 'movieLens/per_user/featureList/per_user',
              'regularize_bias': False,
              'l2_reg_weight': 1.0,
              'lbfgs_tolerance': 1e-12,
              'num_of_lbfgs_curvature_pairs': 10,
              'num_of_lbfgs_iterations': 100,
              'has_intercept': True,
              'offset_column_name': 'offset',
              'batch_size': 16,
              'data_format': 'tfrecord',
              'partition_entity': 'user_id',
              'enable_local_indexing': False,
              'max_training_queue_size': 10,
              'training_queue_timeout_in_seconds': 300,
              'num_of_consumers': 1,
              'random_effect_variance_mode': None,
              'disable_random_effect_scoring_after_training': False,
              '__frozen__': True}))

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'per-user-compute-metric',
            'com.linkedin.gdmix.evaluation.Evaluator',
            {'\\--metricsInputDir': 'lr-training/per-user/validation_scores',
             '--outputMetricFile': 'lr-training/per-user/metric',
             '--labelColumnName': 'response',
             '--metricName': 'auc',
             '--predictionColumnName': 'predictionScore'})

        self.assertEqual(expected_partition_job, actual_partition_job)
        self.assertEqual(expected_train_job, actual_train_job)
        self.assertEqual(expected_compute_metric_job, actual_compute_metric_job)


if __name__ == '__main__':
    unittest.main()
