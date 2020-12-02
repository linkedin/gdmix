"""Tests for detext."""
import os
import shutil
import tempfile
import tensorflow as tf
import pytest
from gdmix.gdmix import run

root_dir = os.path.abspath(os.path.dirname(__file__) + "/./../../resources")
detext_base_dir = tempfile.mkdtemp()
detext_model_dir = os.path.join(detext_base_dir, "detext-model")
os.mkdir(detext_model_dir)
inference_score_dir = os.path.join(detext_base_dir, "inference-score")
os.mkdir(inference_score_dir)


class TestDetextModel(tf.test.TestCase):

    def _cleanUp(self, tf_out_dir):
        if os.path.exists(tf_out_dir):
            shutil.rmtree(tf_out_dir, ignore_errors=True)

    @pytest.mark.skip(reason="DeText outer training loop requires TF eager mode while GDMix linear models disable "
                             "eager mode. Once eager mode is set globally and once turned off, it can't be turned "
                             "on again. Therefore the unit tests of detext and linear models can not be combined. "
                             "To run a standalone local test of detext: uncomment this annotation and type: "
                             "ligradle python --file gdmix-trainer/test/models/detext/test_detext.py")
    def test_run_detext(self):
        """
        This method tests running the detext model using LocalExecutor.
        """
        config = '{"cluster":{"chief":["localhost:2222"], "worker":["localhost:2224"]}, "job_name":"worker", ' \
                 '"task": {"index":0, "type": "worker"}, "is_chief":"True", "num_shards":1, "shard_index":0} '
        os.environ["TF_CONFIG"] = config
        args = ["--action", "train",
                "--stage", "fixed_effect",
                "--model_type", "detext",
                "--batch_size", "2",
                "--data_format", "tfrecord",
                "--uid_column_name", "uid",
                "--prediction_score_column_name", "prediction_score",
                "--l2", "2",
                "--all_metrics", "precision@1", "ndcg@10",
                "--use_tfr_loss", "True",
                "--tfr_loss_fn", "softmax_loss",
                "--emb_sim_func", "inner",
                "--elem_rescale", "True",
                "--explicit_empty", "False",
                "--filter_window_size", "3",
                "--ftr_ext", "cnn",
                "--label", "label",
                "--query", "query",
                "--doc_text", "doc_completedQuery",
                "--usr_text", "usr_headline", "usr_skills", "usr_currTitles",
                "--usr_id", "usrId_currTitles",
                "--doc_id", "docId_completedQuery",
                "--wide_ftrs", "wide_ftrs", "one_more_wide_ftrs",
                "--weight", "weight",
                "--num_wide", "3", "3",
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
                "--optimizer", "bert_adam",
                "--out_dir", detext_model_dir,
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
                "--resume_training", "False",
                "--num_gpu", "0",
                "--distribution_strategy", "one_device",
                "--run_eagerly", "True"]
        run(args)
        config = '{"cluster":{"chief":["localhost:2222"], "worker":["localhost:2224"]}, "job_name":"worker", ' \
                 '"task": {"index":0, "type": "worker"}, "is_chief":"True", "num_shards":1, "shard_index":0} '
        os.environ["TF_CONFIG"] = config
        args = ["--action", "inference",
                "--validation_score_dir", inference_score_dir,
                "--stage", "fixed_effect",
                "--model_type", "detext",
                "--batch_size", "2",
                "--data_format", "tfrecord",
                "--uid_column_name", "uid",
                "--prediction_score_column_name", "prediction_score",
                "--l2", "2",
                "--all_metrics", "precision@1", "ndcg@10",
                "--use_tfr_loss", "True",
                "--tfr_loss_fn", "softmax_loss",
                "--emb_sim_func", "inner",
                "--elem_rescale", "True",
                "--explicit_empty", "False",
                "--filter_window_size", "3",
                "--ftr_ext", "cnn",
                "--label", "label",
                "--query", "query",
                "--doc_text", "doc_completedQuery",
                "--usr_text", "usr_headline", "usr_skills", "usr_currTitles",
                "--usr_id", "usrId_currTitles",
                "--doc_id", "docId_completedQuery",
                "--wide_ftrs", "wide_ftrs", "one_more_wide_ftrs",
                "--weight", "weight",
                "--num_wide", "3", "3",
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
                "--optimizer", "bert_adam",
                "--out_dir", detext_model_dir,
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
                "--resume_training", "False",
                "--num_gpu", "0",
                "--distribution_strategy", "one_device",
                "--run_eagerly", "True"]
        run(args)
        self._cleanUp(detext_base_dir)
