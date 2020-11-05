from copy import deepcopy
from gdmixworkflow.common.utils import json_config_file_to_obj
from gdmixworkflow.fixed_effect_workflow_generator \
    import FixedEffectWorkflowGenerator
from gdmixworkflow.random_effect_workflow_generator \
    import RandomEffectWorkflowGenerator
import json
import os
from os.path import join as path_join
import unittest

class TestGDMixWorkflowGenerator(unittest.TestCase):
    """
    Test gdmix workflow generator
    """

    def setUp(self):
        """
        Sets configurations of the config file.

        Args:
            self: (todo): write your description
        """
        lr_config_file = path_join(
            os.getcwd(),
            "test/resources/lr-single-node-movieLens.config")
        self.lr_config_obj = json_config_file_to_obj(lr_config_file)

        detext_config_file = path_join(
            os.getcwd(),
            "test/resources/detext-single-node-movieLens.config")
        self.detext_config_obj = json_config_file_to_obj(detext_config_file)

        # Set self.maxDiff to None to see diff for long text
        self.maxDiff = None

    def test_lr_model_fixed_effect_workflow_generator(self):
        """
        Test if the cross - only workflow

        Args:
            self: (todo): write your description
        """
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
            {'--stage': 'fixed_effect',
             '--action': 'train',
             '--model_type': 'logistic_regression',
             '--train_data_path': 'movieLens/global/trainingData',
             '--validation_data_path': 'movieLens/global/validationData',
             '--copy_to_local': False,
             '--feature_file': 'movieLens/global/featureList/global',
             '--label': 'response',
             '--sample_id': 'uid',
             '--sample_weight': 'weight',
             '--feature_bag': 'global',
             '--metadata_file': 'movieLens/global/metadata/tensor_metadata.json',
             '--l2_reg_weight': 1.0,
             '--regularize_bias': False,
             '--optimizer_name': 'LBFGS',
             '--lbfgs_tolerance': 1e-12,
             '--num_of_lbfgs_iterations': 100,
             '--num_of_lbfgs_curvature_pairs': 10,
             '--prediction_score': 'predictionScore',
             '--model_output_dir': 'lr-training/global/models',
             '--training_output_dir': 'lr-training/global/train_scores',
             '--validation_output_dir': 'lr-training/global/validation_scores'})

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator',
            {'\\--inputPath': 'lr-training/global/validation_scores',
             '--outputPath': 'lr-training/global/metric',
             '--labelName': 'response',
             '--scoreName': 'predictionScore'})
        self.assertEqual(actual_train_job, expected_train_job)
        self.assertEqual(actual_compute_metric_job, expected_compute_metric_job)

    def test_detext_model_fixed_effect_workflow_generator(self):
        """
        Test if a fixed - size fixed - size

        Args:
            self: (todo): write your description
        """
        fe_workflow = FixedEffectWorkflowGenerator(self.detext_config_obj)
        # check sequence
        seq = fe_workflow.get_job_sequence()
        self.assertEqual(len(seq), 4)
        # check job properties
        actual_train_job = seq[0]
        actual_inference_training_data_job = seq[1]
        actual_inference_validation_data_job = seq[2]
        actual_compute_metric_job = seq[3]

        expected_train_job_param = {
            '--stage': 'fixed_effect',
            '--model_type': 'detext',
            '--ftr_ext': 'cnn',
            '--elem_rescale': True,
            '--ltr_loss_fn': 'pointwise',
            '--init_weight': '0.1',
            '--learning_rate': '0.002',
            '--num_classes': '1',
            '--max_len': '16',
            '--min_len': '3',
            '--num_filters': '50',
            '--num_train_steps': '1000',
            '--num_units': '64',
            '--optimizer': 'bert_adam',
            '--pmetric': 'auc',
            '--all_metric': 'auc',
            '--steps_per_stats': '10',
            '--steps_per_eval': '100',
            '--train_batch_size': '64',
            '--test_batch_size': '64',
            '--use_deep': True,
            '--vocab_file': 'movieLens/detext/vocab.txt',
            '--resume_training': False,
            '--feature_names': 'label,doc_query,uid,wide_ftrs_sp_idx,wide_ftrs_sp_val',
            '--num_wide_sp': '45',
            '--train_file': 'movieLens/detext/trainingData/train_data.tfrecord',
            '--dev_file': 'movieLens/detext/validationData/test_data.tfrecord',
            '--test_file': 'movieLens/detext/validationData/test_data.tfrecord',
            '--metadata_file': 'movieLens/per-user/metadata/tensor_metadata.json',
            '--label': 'response',
            '--sample_id': 'uid',
            '--sample_weight': 'weight',
            '--feature_bag': 'global',
            '--keep_checkpoint_max': '1',
            '--prediction_score': 'predictionScore',
            '--out_dir': 'detext-training/global/models'}
        expected_train_job = ('gdmix_tfjob', 'global-tf-train', '', expected_train_job_param)
        self.assertEqual(expected_train_job, actual_train_job)

        expected_inference_training_data_job_param = deepcopy(expected_train_job_param)
        expected_inference_training_data_job_param["--dev_file"] = 'movieLens/detext/trainingData/train_data.tfrecord'
        expected_inference_training_data_job_param["--validation_output_dir"] = "detext-training/global/train_scores"
        expected_inference_training_data_job_param["--action"] = "inference"
        expected_inference_training_data_job = (
            'gdmix_tfjob',
            'global-tf-inference-train-data',
            '',
            expected_inference_training_data_job_param)
        self.assertEqual(expected_inference_training_data_job, actual_inference_training_data_job)

        expected_inference_validation_data_job_param = deepcopy(expected_train_job_param)
        expected_inference_validation_data_job_param["--dev_file"] = 'movieLens/detext/validationData/test_data.tfrecord'
        expected_inference_validation_data_job_param["--validation_output_dir"] = "detext-training/global/validation_scores"
        expected_inference_validation_data_job_param["--action"] = "inference"
        expected_inference_validation_data_job = (
            'gdmix_tfjob',
            'global-tf-inference-validation-data',
            '',
            expected_inference_validation_data_job_param)
        self.assertEqual(expected_inference_validation_data_job, actual_inference_validation_data_job)

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator',
            {'\\--inputPath': 'detext-training/global/validation_scores',
             '--outputPath': 'detext-training/global/metric',
             '--labelName': 'response',
             '--scoreName': 'predictionScore'})
        self.assertEqual(actual_compute_metric_job, expected_compute_metric_job)

    def test_lr_model_random_effect_workflow_generator(self):
        """
        Randomly generate learning rate rate sequence

        Args:
            self: (todo): write your description
        """
        re_workflow = RandomEffectWorkflowGenerator(self.lr_config_obj)
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
            {'\\--trainInputDataPath': 'movieLens/per_user/trainingData',
             '--validationInputDataPath': 'movieLens/per_user/validationData',
             '--inputMetadataFile': 'movieLens/per_user/metadata/tensor_metadata.json',
             '--partitionEntity': 'user_id',
             '--numPartitions': 1,
             '--dataFormat': 'tfrecord',
             '--trainOutputPartitionDataPath': 'lr-training/per-user/partition/trainingData',
             '--validationOutputPartitionDataPath': 'lr-training/per-user/partition/validationData',
             '--outputMetadataFile': 'lr-training/per-user/partition/metadata/tensor_metadata.json',
             '--outputPartitionListFile': 'lr-training/per-user/partition/partitionList.txt',
             '--predictionScore': 'predictionScore',
             '--trainInputScorePath': 'lr-training/global/train_scores',
             '--validationInputScorePath': 'lr-training/global/validation_scores'})

        expected_train_job = (
            'gdmix_tfjob',
            'per-user-tf-train',
            '',
            {'--stage': 'random_effect',
             '--action': 'train',
             '--model_type': 'logistic_regression',
             '--partition_entity': 'user_id',
             '--train_data_path': 'lr-training/per-user/partition/trainingData',
             '--validation_data_path': 'lr-training/per-user/partition/validationData',
             '--feature_file': 'movieLens/per_user/featureList/per_user',
             '--label': 'response',
             '--sample_id': 'uid',
             '--sample_weight': 'weight',
             '--feature_bag': 'per_user',
             '--metadata_file': 'lr-training/per-user/partition/metadata/tensor_metadata.json',
             '--l2_reg_weight': 1.0,
             '--regularize_bias': False,
             '--num_partitions': 1,
             '--optimizer_name': 'LBFGS',
             '--lbfgs_tolerance': 1e-12,
             '--num_of_lbfgs_iterations': 100,
             '--num_of_lbfgs_curvature_pairs': 10,
             '--max_training_queue_size': 10,
             '--num_of_consumers': 1,
             '--enable_local_indexing': False,
             '--prediction_score': 'predictionScore',
             '--partition_list_file': 'lr-training/per-user/partition/partitionList.txt',
             '--model_output_dir': 'lr-training/per-user/models',
             '--training_output_dir': 'lr-training/per-user/train_scores',
             '--validation_output_dir': 'lr-training/per-user/validation_scores'})

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'per-user-compute-metric',
            'com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator',
            {'\\--inputPath': 'lr-training/per-user/validation_scores',
             '--outputPath': 'lr-training/per-user/metric',
             '--labelName': 'response',
             '--scoreName': 'predictionScore'})

        self.assertEqual(expected_partition_job, actual_partition_job)
        self.assertEqual(expected_train_job, actual_train_job)
        self.assertEqual(expected_compute_metric_job, actual_compute_metric_job)


if __name__ == '__main__':
    unittest.main()
