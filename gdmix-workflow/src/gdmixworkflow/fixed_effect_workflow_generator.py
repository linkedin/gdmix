from dataclasses import replace
from os.path import join as path_join

from detext.run_detext import DetextArg
from gdmix.models.custom.fixed_effect_lr_lbfgs_model import FixedLRParams
from gdmix.params import Params

from gdmixworkflow.common.constants import *
from gdmixworkflow.workflow_generator import WorkflowGenerator


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
        self.metric_name = MSE if self.model_type == LINEAR_REGRESSION else AUC

    def get_train_job(self):
        """ Get tfjob training job.
        :return (job_type, job_name, "", job_params) where job_params are typed param containers supported by smart-arg
        """
        params = {STAGE: FIXED_EFFECT}
        # get params from config
        flatten_config_obj(params, self.fixed_effect_config_obj)
        # adjust param keys
        params.pop("name")
        params[PREDICTION_SCORE] = params.pop(OUTPUT_COLUMN_NAME)
        # add output params
        if self.model_type in [LINEAR_REGRESSION, LOGISTIC_REGRESSION]:
            params[MODEL_OUTPUT_DIR] = self.model_path
            params[TRAINING_OUTPUT_DIR] = self.train_score_path
            params[VALIDATION_OUTPUT_DIR] = self.validation_score_path
        elif self.model_type == DETEXT:
            params = DetextArg(**self.fixed_effect_config, out_dir=self.output_model_dir)
        else:
            raise ValueError(f'unsupported model_type: {self.model_type}')
        return GDMIX_TFJOB, f"{self.fixed_effect_name}-tf-train", "", (self.gdmix_params, params)

    def get_detext_inference_job(self):
        """ Get detext inference job. For LR model the inference job is included in train
        job, this job is for DeText model inference.
        Return: an inference job inferencing training and validation data
        (job_type, job_name, "", job_params)
        """
        params = replace(self.gdmix_params, action=ACTION_INFERENCE), DetextArg(**self.fixed_effect_config, out_dir=self.output_model_dir)
        return GDMIX_TFJOB, f"{self.fixed_effect_name}-tf-inference", "", params

    def get_compute_metric_job(self):
        """ Get sparkjob compute metric job.
        Return: (job_type, job_name, class_name, job_params)
        """
        params = {
            r"\-inputPath": self.validation_score_path,
            "-outputPath": self.metric_path,
            "-labelName": self.fixed_effect_config_obj.input_column_names.label,
            "-scoreName": self.fixed_effect_config_obj.output_column_name,
            "-metricName": self.metric_name
        }
        return (GDMIX_SPARKJOB,
                "{}-compute-metric".format(self.fixed_effect_name),
                "com.linkedin.gdmix.evaluation.Evaluator",
                params)

    def get_job_sequence(self):
        """ Get job sequence of fixed effect workflow. """
        if self.model_type in [LINEAR_REGRESSION, LOGISTIC_REGRESSION]:
            jobs = [self.get_train_job(), self.get_compute_metric_job()]
        elif self.model_type == DETEXT:
            jobs = [self.get_train_job(),
                    self.get_detext_inference_job(),
                    self.get_compute_metric_job()]
        else:
            raise ValueError('unsupported model_type: {}'.format(model_type))
        return jobs
