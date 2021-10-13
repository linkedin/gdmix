import os
import unittest
from dataclasses import replace
from os.path import join as path_join

from detext.run_detext import DetextArg
from gdmix.models.custom.fixed_effect_lr_lbfgs_model import FixedLRParams
from gdmix.models.custom.random_effect_lr_lbfgs_model import REParams
from gdmix.params import Params
from gdmix.util.constants import ACTION_INFERENCE

from gdmixworkflow.common.utils import json_config_file_to_obj
from gdmixworkflow.fixed_effect_workflow_generator import FixedEffectWorkflowGenerator
from gdmixworkflow.random_effect_workflow_generator import RandomEffectWorkflowGenerator
from gdmixworkflow.single_node.local_ops import get_param_list



class TestGDMixWorkflowGenerator(unittest.TestCase):
    """
    Test gdmix workflow generator
    """

    def setUp(self):
        logistic_regression_config_file = path_join(
            os.getcwd(),
            "test/resources/logistic-regression-single-node-movieLens.config")
        self.logistic_regression_config_obj = json_config_file_to_obj(logistic_regression_config_file)

        linear_regression_config_file = path_join(
            os.getcwd(),
            "test/resources/logistic-regression-single-node-movieLens.config")
        self.linear_regression_config_obj = json_config_file_to_obj(linear_regression_config_file)

        detext_config_file = path_join(
            os.getcwd(),
            "test/resources/detext-movieLens.yaml")
        self.detext_config_obj = json_config_file_to_obj(detext_config_file)

        # Set self.maxDiff to None to see diff for long text
        self.maxDiff = None

    def test_logistic_regression_model_fixed_effect_workflow_generator(self):
        fe_workflow = FixedEffectWorkflowGenerator(self.logistic_regression_config_obj)
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
             '--model_output_dir': 'logistic-regression-training/global/models',
             '--training_output_dir': 'logistic-regression-training/global/train_scores',
             '--validation_output_dir': 'logistic-regression-training/global/validation_scores'})

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.Evaluator',
            {'\\-inputPath': 'logistic-regression-training/global/validation_scores',
             '-outputPath': 'logistic-regression-training/global/metric',
             '-labelName': 'response',
             '-scoreName': 'predictionScore',
             '-metricName': 'auc'})
        self.assertEqual(actual_train_job, expected_train_job)
        self.assertEqual(actual_compute_metric_job, expected_compute_metric_job)

    def test_linear_regression_model_fixed_effect_workflow_generator(self):
        fe_workflow = FixedEffectWorkflowGenerator(self.linear_regression_config_obj)
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
             '--model_type': 'linear_regression',
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
             '--model_output_dir': 'linear-regression-training/global/models',
             '--training_output_dir': 'linear-regression-training/global/train_scores',
             '--validation_output_dir': 'linear-regression-training/global/validation_scores'})

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.Evaluator',
            {'\\-inputPath': 'linear-regression-training/global/validation_scores',
             '-outputPath': 'linear-regression-training/global/metric',
             '-labelName': 'response',
             '-scoreName': 'predictionScore',
             '-metricName': 'mse'})
        self.assertEqual(actual_train_job, expected_train_job)
        self.assertEqual(actual_compute_metric_job, expected_compute_metric_job)

    def test_detext_model_fixed_effect_workflow_generator(self):
        # return  # skip test for now
        fe_workflow = FixedEffectWorkflowGenerator(self.detext_config_obj)
        # check sequence
        seq = fe_workflow.get_job_sequence()
        self.assertEqual(len(seq), 3)
        # check job properties
        actual_train_job = seq[0]
        actual_inference_training_data_job = seq[1]
        actual_inference_validation_data_job = seq[2]
        actual_compute_metric_job = seq[3]

        expected_train_job_param = (
            Params(uid_column_name='uid', weight_column_name='weight', label_column_name='response', prediction_score_column_name='predictionScore',
                   prediction_score_per_coordinate_column_name='predictionScorePerCoordinate', action='train', stage='fixed_effect', model_type='detext',
                   training_score_dir='detext-training/global/training_scores', validation_score_dir='detext-training/global/validation_scores'),
            DetextArg(queyr='query', wide_ftrs='wide_ftrs', doc_text='doc_title', usr_text='user_headline',
                      wide_ftrs_sp_idx='wide_ftrs_sp_idx', wide_ftrs_sp_val='wide_ftrs_sp_val', ftr_ext='cnn', num_units=64,
                      sp_emb_size=1, num_hidden=[0], num_wide=0, num_wide_sp=45, use_deep=True, elem_rescale=True, use_doc_projection=False,
                      use_usr_projection=False, ltr_loss_fn='pointwise', emb_sim_func=['inner'], num_classes=1, filter_window_sizes=[3], num_filters=50,
                      explicit_empty=False, use_bert_dropout=False, unit_type='lstm', num_layers=1,
                      num_residual_layers=0, forget_bias=1.0, rnn_dropout=0.0, bidirectional=False, normalized_lm=False, optimizer='bert_adam',
                      max_gradient_norm=1.0, learning_rate=0.002, num_train_steps=1000, num_warmup_steps=0, train_batch_size=64,
                      test_batch_size=64, l1=None, l2=None, train_file='movieLens/detext/trainingData/train_data.tfrecord',
                      dev_file='movieLens/detext/validationData/test_data.tfrecord', test_file='movieLens/detext/validationData/test_data.tfrecord',
                      out_dir='detext-training/global/models', max_len=16, min_len=3, vocab_file='movieLens/detext/vocab.txt', we_file=None,
                      we_trainable=True, PAD='[PAD]', SEP='[SEP]', CLS='[CLS]', UNK='[UNK]', MASK='[MASK]', we_file_for_id_ftr=None,
                      we_trainable_for_id_ftr=True, PAD_FOR_ID_FTR='[PAD]', UNK_FOR_ID_FTR='[UNK]', random_seed=1234, steps_per_stats=10, num_eval_rounds=None,
                      steps_per_eval=100, keep_checkpoint_max=1, init_weight=0.1, pmetric='auc', all_metrics=['auc'], score_rescale=None,
                      add_first_dim_for_query_placeholder=False, add_first_dim_for_usr_placeholder=False, tokenization='punct', resume_training=False,
                      use_tfr_loss=False, tfr_loss_fn='softmax_loss'))
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
        expected_inference_validation_data_job_param[
            "--dev_file"] = 'movieLens/detext/validationData/test_data.tfrecord'
        expected_inference_validation_data_job_param[
            "--validation_output_dir"] = "detext-training/global/validation_scores"
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
            'com.linkedin.gdmix.evaluation.Evaluator',
            {'\\-inputPath': 'detext-training/global/validation_scores',
             '-outputPath': 'detext-training/global/metric',
             '-labelName': 'response',
             '-scoreName': 'predictionScore',
             '-metricName': 'auc'})
        self.assertEqual(actual_compute_metric_job, expected_compute_metric_job)

    def test_logistic_regression_model_random_effect_workflow_generator(self):
        re_workflow = RandomEffectWorkflowGenerator(self.logistic_regression_config_obj)
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
            {'\\-trainInputDataPath': 'movieLens/per_user/trainingData',
             '--validationInputDataPath': 'movieLens/per_user/validationData',
             '-inputMetadataFile': 'movieLens/per_user/metadata/tensor_metadata.json',
             '-partitionEntity': 'user_id',
             '-numPartitions': 1,
             '-dataFormat': 'tfrecord',
             '-featureBag': 'per_user',
             '-maxNumOfSamplesPerModel': -1,
             '-minNumOfSamplesPerModel': -1,
             '-trainOutputPartitionDataPath': 'logistic-regression-training/per-user/partition/trainingData',
             '-validationOutputPartitionDataPath': 'logistic-regression-training/per-user/partition/validationData',
             '-outputMetadataFile': 'logistic-regression-training/per-user/partition/metadata/tensor_metadata.json',
             '-outputPartitionListFile': 'logistic-regression-training/per-user/partition/partitionList.txt',
             '-predictionScore': 'predictionScore',
             '-trainInputScorePath': 'logistic-regression-training/global/train_scores',
             '-validationInputScorePath': 'logistic-regression-training/global/validation_scores'})

        expected_train_job = (
            'gdmix_tfjob',
            'per-user-tf-train',
            '',
            {'--stage': 'random_effect',
             '--action': 'train',
             '--model_type': 'logistic_regression',
             '--partition_entity': 'user_id',
             '--train_data_path': 'logistic-regression-training/per-user/partition/trainingData',
             '--validation_data_path': 'logistic-regression-training/per-user/partition/validationData',
             '--feature_file': 'movieLens/per_user/featureList/per_user',
             '--label': 'response',
             '--sample_id': 'uid',
             '--sample_weight': 'weight',
             '--feature_bag': 'per_user',
             '--metadata_file': 'logistic-regression-training/per-user/partition/metadata/tensor_metadata.json',
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
             '--partition_list_file': 'logistic-regression-training/per-user/partition/partitionList.txt',
             '--model_output_dir': 'logistic-regression-training/per-user/models',
             '--training_output_dir': 'logistic-regression-training/per-user/train_scores',
             '--validation_output_dir': 'logistic-regression-training/per-user/validation_scores'})
        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'per-user-compute-metric',
            'com.linkedin.gdmix.evaluation.Evaluator',
            {'\\-inputPath': 'logistic-regression-training/per-user/validation_scores',
             '-outputPath': 'logistic-regression-training/per-user/metric',
             '-labelName': 'response',
             '-scoreName': 'predictionScore',
             '-metricName': 'auc'})

        self.assertEqual(expected_partition_job, actual_partition_job)
        self.assertEqual(expected_train_job, actual_train_job)
        self.assertEqual(expected_compute_metric_job, actual_compute_metric_job)


if __name__ == '__main__':
    unittest.main()
