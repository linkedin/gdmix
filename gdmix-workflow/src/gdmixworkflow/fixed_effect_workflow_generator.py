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
        self.model_path = path_join(self.output_dir, MODELS)
        self.train_score_path = path_join(self.output_dir, TRAIN_SCORES)
        self.validation_score_path = path_join(self.output_dir, VALIDATION_SCORES)
        self.metric_path = path_join(self.output_dir, METRIC)
        self.model_type = self.fixed_effect_config_obj.model_type

    def get_train_job(self):
        """ Get tfjob training job.
        Return: [job_type, job_name, "", job_params]
        """
        params = {STAGE: FIXED_EFFECT}
        # get params from config
        flatten_config_obj(params, self.fixed_effect_config_obj)
        # adjust param keys
        params.pop("name")
        params[PREDICTION_SCORE] = params.pop(OUTPUT_COLUMN_NAME)
        # add output params
        if self.model_type == LOGISTIC_REGRESSION:
            params[MODEL_OUTPUT_DIR] = self.model_path
            params[TRAINING_OUTPUT_DIR] = self.train_score_path
            params[VALIDATION_OUTPUT_DIR] = self.validation_score_path
        elif self.model_type == DETEXT:
            params[DETEXT_MODEL_OUTPUT_DIR] = self.model_path
        else:
            raise ValueError('unsupported model_type: {}'.format(model_type))

        params = prefix_dash_dash(params)
        return (GDMIX_TFJOB, "{}-tf-train".format(self.fixed_effect_name), "", params)

    def get_detext_inference_jobs(self):
        """ Get detext inference job. For LR model the inference job is included in train
        job, this job is for DeText model inference.
        Return: two inferece jobs for training and  validation data
        ([job_type, job_name, "", job_params], [job_type, job_name, "", job_params])
        """
        params = {STAGE: FIXED_EFFECT, ACTION: ACTION_INFERENCE}
        # get params from config
        flatten_config_obj(params, self.fixed_effect_config_obj)
        # adjust param keys
        params.pop("name")
        params[PREDICTION_SCORE] = params.pop(OUTPUT_COLUMN_NAME)
        params[DETEXT_MODEL_OUTPUT_DIR] = self.model_path
        # "--dev_file" and "--validation_output_dir" are used as input and output for the detext inference job
        inference_train_data_params = deepcopy(params)
        inference_train_data_params[DETEXT_DEV_FILE] = self.fixed_effect_config_obj.train_file
        inference_train_data_params[VALIDATION_OUTPUT_DIR] = self.train_score_path
        inference_train_data_job = (GDMIX_TFJOB, "{}-tf-inference-train-data".format(
            self.fixed_effect_name), "", prefix_dash_dash(inference_train_data_params))

        inference_validation_data_params = deepcopy(params)
        inference_validation_data_params[DETEXT_DEV_FILE] = self.fixed_effect_config_obj.dev_file
        inference_validation_data_params[VALIDATION_OUTPUT_DIR] = self.validation_score_path
        inference_validation_data_job = (GDMIX_TFJOB, "{}-tf-inference-validation-data".format(
            self.fixed_effect_name), "", prefix_dash_dash(inference_validation_data_params))
        return (inference_train_data_job, inference_validation_data_job)

    def get_compute_metric_job(self):
        """ Get sparkjob compute metric job.
        Return: [job_type, job_name, class_name, job_params]
        """
        params = {
            r"\-inputPath": self.validation_score_path,
            "-outputPath": self.metric_path,
            "-labelName": self.fixed_effect_config_obj.input_column_names.label,
            "-scoreName": self.fixed_effect_config_obj.output_column_name
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
            inference_train_data_job, inference_validation_data_job = self.get_detext_inference_jobs()
            jobs = [self.get_train_job(),
                    inference_train_data_job,
                    inference_validation_data_job,
                    self.get_compute_metric_job()]
        else:
            raise ValueError('unsupported model_type: {}'.format(model_type))
        return jobs