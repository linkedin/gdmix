"""Tests for deep-match-training."""
import os
import shutil
import tensorflow as tf
import pytest
from gdmix.gdmix import run

root_dir = os.path.abspath(os.path.dirname(__file__) + "/./../../resources")

tfrecord_out_dir = os.path.join(root_dir, "tfrecord-output")
score_out_dir = os.path.join(tfrecord_out_dir, "score-output")


class TestDetextModel(tf.test.TestCase):

    def _cleanUp(self, tf_out_dir):
        """
        Cleans up the temporary directory.

        Args:
            self: (todo): write your description
            tf_out_dir: (str): write your description
        """
        if os.path.exists(tf_out_dir):
            shutil.rmtree(tf_out_dir, ignore_errors=True)

    @pytest.mark.skip(reason="no way of currently testing this")
    def test_run_dm_tfrecord(self):
        """
        This method test the entire model in run_dm using LocalExecutor.
        """
        config = '{"cluster":{"chief":["localhost:2222"], "worker":["localhost:2224"]}, "job_name":"worker", ' \
                 '"task": {"index":0, "type": "worker"}, "is_chief":"True", "num_shards":1, "shard_index":0} '
        os.environ["TF_CONFIG"] = config
        args = ["--action", 'train',
                "--stage", "fixed_effect",
                "--model_type", "detext",
                "--label", "label",
                "--batch_size", "2",
                "--data_format", "tfrecord",
                "--sample_id", "uid",
                "--sample_weight", "weight",
                "--label", "label",
                "--prediction_score", "prediction_score",
                "--l2", "2",
                "--all_metrics", "precision@1,ndcg@10",
                "--use_tfr_loss", "True",
                "--tfr_loss_fn", "softmax_loss",
                "--emb_sim_func", "inner",
                "--elem_rescale", "True",
                "--explicit_empty", "False",
                "--filter_window_size", "3",
                "--ftr_ext", "cnn",
                "--feature_names", "label,query,doc_completedQuery,usr_headline,usrId_currTitles,docId_completedQuery,wide_ftrs,weight",
                "--init_weight", "0.1",
                "--lambda_metric", "None",
                "--learning_rate", "0.002",
                "--ltr_loss_fn", "softmax",
                "--max_gradient_norm", "1.0",
                "--max_len", "16",
                "--min_len", "3",
                "--num_filters", "4",
                "--num_hidden", "10",
                "--num_train_steps", "4",  # set to >=4 to test robustness of serving_input_fn
                "--num_units", "4",
                "--num_wide", "3",
                "--optimizer", "bert_adam",
                "--out_dir", tfrecord_out_dir,
                "--pmetric", "ndcg@10",
                "--random_seed", "11",
                "--steps_per_stats", "1",
                "--steps_per_eval", "2",
                "--train_batch_size", "2",
                "--test_batch_size", "2",
                "--test_file", os.path.join(root_dir, "train", "dataset", "tfrecord"),
                "--dev_file", os.path.join(root_dir, "train", "dataset", "tfrecord"),
                "--train_file", os.path.join(root_dir, "train", "dataset", "tfrecord"),
                "--use_wide", "True",
                "--use_deep", "True",
                "--vocab_file", os.path.join(root_dir, "vocab.txt"),
                "--vocab_file_for_id_ftr", os.path.join(root_dir, "vocab.txt"),
                "--metadata_path", os.path.join(root_dir, "train", "dataset", "tensor_metadata.json"),
                "--resume_training", "False"]
        run(args)
        config = '{"cluster":{"chief":["localhost:2222"], "worker":["localhost:2224"]}, "job_name":"worker", ' \
                 '"task": {"index":0, "type": "worker"}, "is_chief":"True", "num_shards":1, "shard_index":0} '
        os.environ["TF_CONFIG"] = config
        args = ["--action", 'validate',
                "--stage", "fixed_effect",
                "--model_type", "detext",
                "--label", "label",
                "--batch_size", "2",
                "--data_format", "tfrecord",
                "--sample_id", "uid",
                "--sample_weight", "weight",
                "--label", "label",
                "--prediction_score", "prediction_score",
                "--l2", "2",
                "--validation_output_dir", score_out_dir,
                "--all_metrics", "precision@1,ndcg@10",
                "--use_tfr_loss", "True",
                "--tfr_loss_fn", "softmax_loss",
                "--emb_sim_func", "inner",
                "--elem_rescale", "True",
                "--explicit_empty", "False",
                "--filter_window_size", "3",
                "--ftr_ext", "cnn",
                "--feature_names", "label,query,doc_completedQuery,usr_headline,usrId_currTitles,docId_completedQuery,wide_ftrs,weight",
                "--init_weight", "0.1",
                "--lambda_metric", "None",
                "--learning_rate", "0.002",
                "--ltr_loss_fn", "softmax",
                "--max_gradient_norm", "1.0",
                "--max_len", "16",
                "--min_len", "3",
                "--num_filters", "4",
                "--num_hidden", "10",
                "--num_train_steps", "4",  # set to >=4 to test robustness of serving_input_fn
                "--num_units", "4",
                "--num_wide", "3",
                "--optimizer", "bert_adam",
                "--out_dir", tfrecord_out_dir,
                "--pmetric", "ndcg@10",
                "--random_seed", "11",
                "--steps_per_stats", "1",
                "--steps_per_eval", "2",
                "--train_batch_size", "2",
                "--test_batch_size", "2",
                "--test_file", os.path.join(root_dir, "train", "dataset", "tfrecord"),
                "--dev_file", os.path.join(root_dir, "train", "dataset", "tfrecord"),
                "--train_file", os.path.join(root_dir, "train", "dataset", "tfrecord"),
                "--use_wide", "True",
                "--use_deep", "True",
                "--vocab_file", os.path.join(root_dir, "vocab.txt"),
                "--vocab_file_for_id_ftr", os.path.join(root_dir, "vocab.txt"),
                "--metadata_path", os.path.join(root_dir, "train", "dataset", "tensor_metadata.json"),
                "--resume_training", "False"]
        run(args)
        self._cleanUp(tfrecord_out_dir)
