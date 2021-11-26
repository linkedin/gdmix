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
            "test/resources/lr-movieLens.yaml")
        self.lr_config_obj = json_config_file_to_obj(lr_config_file)

        detext_config_file = path_join(
            os.getcwd(),
            "test/resources/detext-movieLens.yaml")
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
                           num_of_lbfgs_curvature_pairs=10, num_of_lbfgs_iterations=100, batch_size=16, data_format='tfrecord',
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
                training_score_dir='detext-training/global/training_scores', validation_score_dir='detext-training/global/validation_scores',
                partition_list_file=None),
            DetextArg(use_lr_schedule=True, num_warmup_steps=0, optimizer='bert_adam', use_bias_correction_for_adamw=False,
                max_gradient_norm=1.0, learning_rate=0.002, task_type='ranking', num_classes=1, l1=0.0, l2=0.0, pmetric='auc', all_metrics=['auc'],
                ltr_loss_fn='pointwise', use_tfr_loss=False, tfr_loss_fn='softmax_loss', tfr_lambda_weights=None, explicit_allreduce=True,
                lambda_metric=None, random_seed=1234, ftr_ext='cnn', num_units=64, num_units_for_id_ftr=128, sparse_embedding_size=1,
                sparse_embedding_cross_ftr_combiner='sum', sparse_embedding_same_ftr_combiner='sum', num_hidden=[], rescale_dense_ftrs=True,
                add_doc_projection=False, add_user_projection=False, emb_sim_func=['inner'], filter_window_sizes=[3], num_filters=50, lr_bert=0.0,
                bert_hub_url=None, num_layers=1, forget_bias=1.0, rnn_dropout=0.0, bidirectional=False, max_filter_window_size=3, query_column_name='',
                label_column_name='', weight_column_name='', uid_column_name='', task_id_column_name='', task_ids=None, task_weights=None,
                dense_ftrs_column_names=[], nums_dense_ftrs=[], sparse_ftrs_column_names=[], nums_sparse_ftrs=[], user_text_column_names=[],
                doc_text_column_names=[], user_id_column_names=[], doc_id_column_names=[], std_file=None, vocab_file='movieLens/detext/vocab.txt',
                vocab_hub_url='', we_file='', embedding_hub_url='', we_trainable=True, PAD='[PAD]', SEP='[SEP]', CLS='[CLS]', UNK='[UNK]', MASK='[MASK]',
                vocab_file_for_id_ftr='', vocab_hub_url_for_id_ftr='', we_file_for_id_ftr='', embedding_hub_url_for_id_ftr='', we_trainable_for_id_ftr=True,
                PAD_FOR_ID_FTR='[PAD]', UNK_FOR_ID_FTR='[UNK]', max_len=16, min_len=3, feature_type2name={}, has_query=False, use_dense_ftrs=False,
                total_num_dense_ftrs=0, use_sparse_ftrs=False, total_num_sparse_ftrs=0, num_doc_fields=0, num_user_fields=0, num_doc_id_fields=0,
                num_user_id_fields=0, num_id_fields=0, num_text_fields=0, ftr_mean=None, ftr_std=None, distribution_strategy='one_device',
                all_reduce_alg=None, num_gpu=0, run_eagerly=False, train_file='movieLens/detext/trainingData/train_data.tfrecord',
                dev_file='movieLens/detext/validationData/test_data.tfrecord', test_file='movieLens/detext/validationData/test_data.tfrecord',
                out_dir='detext-training/global/models', num_train_steps=1000, num_eval_steps=0, num_epochs=0, steps_per_stats=10,
                num_eval_rounds=0, steps_per_eval=100, resume_training=False, keep_checkpoint_max=1, train_batch_size=64, test_batch_size=64))

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
