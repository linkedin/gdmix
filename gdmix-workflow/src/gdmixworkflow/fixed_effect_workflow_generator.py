from copy import deepcopy
from os.path import join as path_join
import os
from gdmixworkflow.workflow_generator import WorkflowGenerator
from gdmixworkflow.common.utils import *
from gdmixworkflow.common.constants import *


class FixedEffectWorkflowGenerator(WorkflowGenerator):
    """ Generate gdmix fixed effect workflow consisting of
    - tfjob: train and inference training and validation data
    - sparkjob: compute-metric
    """

    def __init__(self, gdmix_config_obj, jar_path="", namespace="",
                 secret_name="", image="", service_account="", job_suffix=""):
        """ Init to generate gdmix fixed effect workflow. """
        super().__init__(gdmix_config_obj, jar_path, namespace, secret_name, image, service_account, job_suffix)
        self.fixed_effect_config_obj = self.gdmix_config_obj.fixed_effect_config
        self.fixed_effect_name = self.fixed_effect_config_obj.name
        self.output_dir = path_join(gdmix_config_obj.output_dir,
                                   self.fixed_effect_config_obj.name)
        self.output_model_dir = path_join(self.output_dir, MODELS)
        self.training_score_dir = path_join(self.output_dir, TRAINING_SCORES)
        self.validation_score_dir = path_join(self.output_dir, VALIDATION_SCORES)
        self.metric_file = path_join(self.output_dir, METRIC)
        self.model_type = self.fixed_effect_config_obj.model_type

    def get_train_job(self):
        """ Get tfjob training job.
        Return: (job_type, job_name, "", job_params)
        """
        params = {STAGE: FIXED_EFFECT}
        # get params from config
        flatten_config_obj(params, self.fixed_effect_config_obj)
        # adjust param keys
        params.pop("name")
        params[PREDICTION_SCORE_COLUMN_NAME] = params.pop(PREDICTION_SCORE_COLUMN_NAME)
        # add output params
        if self.model_type == LOGISTIC_REGRESSION:
            params[OUTPUT_MODEL_DIR] = self.output_model_dir
            params[TRAINING_SCORE_DIR] = self.training_score_dir
            params[VALIDATION_SCORE_DIR] = self.validation_score_dir
        elif self.model_type == DETEXT:
            params[DETEXT_MODEL_OUTPUT_DIR] = self.output_model_dir
        else:
            raise ValueError('unsupported model_type: {}'.format(model_type))

        params = prefix_dash_dash(params)
        return (GDMIX_TFJOB, "{}-tf-train".format(self.fixed_effect_name), "", params)

    def get_detext_inference_job(self):
        """ Get detext inference job. For LR model the inference job is included in train
        job, this job is for DeText model inference.
        Return: an inferece job inferencing training and validation data
        (job_type, job_name, "", job_params)
        """
        params = {STAGE: FIXED_EFFECT, ACTION: ACTION_INFERENCE}
        # get params from config
        flatten_config_obj(params, self.fixed_effect_config_obj)
        # adjust param keys
        params.pop("name")
        params[PREDICTION_SCORE_COLUMN_NAME] = params.pop(PREDICTION_SCORE_COLUMN_NAME)
        params[DETEXT_MODEL_OUTPUT_DIR] = self.output_model_dir
        # "--dev_file" and "--validation_output_dir" are used as input and output for the detext inference job
        inference_params = deepcopy(params)
        inference_params[TRAINING_DATA_DIR] = self.fixed_effect_config_obj.train_file
        inference_params[TRAINING_SCORE_DIR] = self.training_score_dir
        inference_params[VALIDATION_DATA_DIR] = self.fixed_effect_config_obj.dev_file
        inference_params[VALIDATION_SCORE_DIR] = self.validation_score_dir
        inference_job = (GDMIX_TFJOB, "{}-tf-inference".format(
            self.fixed_effect_name), "", prefix_dash_dash(inference_params))

        return inference_job

    def get_compute_metric_job(self):
        """ Get sparkjob compute metric job.
        Return: (job_type, job_name, class_name, job_params)
        """
        params = {
            r"\--metricsInputDir": self.validation_score_dir,
            "--outputMetricFile": self.metric_file,
            "--labelColumnName": self.fixed_effect_config_obj.input_column_names.label_column_name,
            "--predictionColumnName": self.fixed_effect_config_obj.prediction_score_column_name
        }
        return (GDMIX_SPARKJOB,
                "{}-compute-metric".format(self.fixed_effect_name),
                "com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator",
                params)

    def get_job_sequence(self):
        """ Get job sequence of fixed effect workflow. """
        if self.model_type == LOGISTIC_REGRESSION:
            jobs = [self.get_train_job(), self.get_compute_metric_job()]
        elif self.model_type == DETEXT:
            jobs = [self.get_train_job(),
                    self.get_detext_inference_job(),
                    self.get_compute_metric_job()]
        else:
            raise ValueError('unsupported model_type: {}'.format(model_type))
        return jobs
