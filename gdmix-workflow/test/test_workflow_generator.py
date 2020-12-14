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
        lr_config_file = path_join(
            os.getcwd(),
            "test/resources/lr-movieLens.config")
        self.lr_config_obj = json_config_file_to_obj(lr_config_file)

        detext_config_file = path_join(
            os.getcwd(),
            "test/resources/detext-movieLens.config")
        self.detext_config_obj = json_config_file_to_obj(detext_config_file)

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
            (Params(uid_column_name='uid', weight_column_name='weight', label_column_name='response', prediction_score_column_name='predictionScore',
                    prediction_score_per_coordinate_column_name='predictionScorePerCoordinate', action='train', stage='fixed_effect',
                    model_type='logistic_regression', training_score_dir='lr-training/global/training_scores',
                    validation_score_dir='lr-training/global/validation_scores', partition_list_file=None),
             FixedLRParams(metadata_file='movieLens/global/metadata/tensor_metadata.json', output_model_dir='lr-training/global/models',
                           training_data_dir='movieLens/global/trainingData', validation_data_dir='movieLens/global/validationData', feature_bag='global',
                           feature_file='movieLens/global/featureList/global', regularize_bias=False, l2_reg_weight=1.0, lbfgs_tolerance=1e-12,
                           num_of_lbfgs_curvature_pairs=10, num_of_lbfgs_iterations=100, offset='offset', batch_size=16, data_format='tfrecord',
                           copy_to_local=False, num_server_creation_retries=50, retry_interval=2, delayed_exit_in_seconds=60)))

        self.assertEqual(expected_train_job, actual_train_job)

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator',
            {'\\--metricsInputDir': 'lr-training/global/validation_scores',
             '--outputMetricFile': 'lr-training/global/metric',
             '--labelColumnName': 'response',
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
            Params(uid_column_name='uid', weight_column_name='weight', label_column_name='response', prediction_score_column_name='predictionScore',
                   prediction_score_per_coordinate_column_name='predictionScorePerCoordinate', action='train', stage='fixed_effect', model_type='detext',
                   training_score_dir='detext-training/global/training_scores', validation_score_dir='detext-training/global/validation_scores'),
            DetextArg(feature_names=['label', 'doc_query', 'uid', 'wide_ftrs_sp_idx', 'wide_ftrs_sp_val'], ftr_ext='cnn', num_units=64,
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

        expected_inference_job_param = (replace(expected_train_job_param[0],
                                                action=ACTION_INFERENCE,
                                                training_score_dir="detext-training/global/training_scores",
                                                validation_score_dir="detext-training/global/validation_scores"), expected_train_job_param[1])
        expected_inference_job = 'gdmix_tfjob', 'global-tf-inference', '', expected_inference_job_param
        self.assertEqual(expected_inference_job, actual_inference_job)

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'global-compute-metric',
            'com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator',
            {'\\--metricsInputDir': 'detext-training/global/validation_scores',
             '--outputMetricFile': 'detext-training/global/metric',
             '--labelColumnName': 'response',
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
            (Params(uid_column_name='uid', weight_column_name='weight', label_column_name='response', prediction_score_column_name='predictionScore',
                    prediction_score_per_coordinate_column_name='predictionScorePerCoordinate', action='train', stage='random_effect',
                    model_type='logistic_regression', training_score_dir='lr-training/per-user/training_scores',
                    validation_score_dir='lr-training/per-user/validation_scores', partition_list_file='lr-training/per-user/partition/partitionList.txt'),
             REParams(metadata_file='lr-training/per-user/partition/metadata/tensor_metadata.json', output_model_dir='lr-training/per-user/models',
                      training_data_dir='lr-training/per-user/partition/trainingData', validation_data_dir='lr-training/per-user/partition/validationData',
                      feature_bag='per_user', feature_file='movieLens/per_user/featureList/per_user', regularize_bias=False, l2_reg_weight=1.0,
                      data_format='tfrecord', partition_entity='user_id', enable_local_indexing=False, max_training_queue_size=10,
                      training_queue_timeout_in_seconds=300, num_of_consumers=1)))

        expected_compute_metric_job = (
            'gdmix_sparkjob',
            'per-user-compute-metric',
            'com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator',
            {'\\--metricsInputDir': 'lr-training/per-user/validation_scores',
             '--outputMetricFile': 'lr-training/per-user/metric',
             '--labelColumnName': 'response',
             '--predictionColumnName': 'predictionScore'})

        self.assertEqual(expected_partition_job, actual_partition_job)
        self.assertEqual(expected_train_job, actual_train_job)
        self.assertEqual(expected_compute_metric_job, actual_compute_metric_job)


if __name__ == '__main__':
    unittest.main()
