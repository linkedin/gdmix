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

        self.fixed_effect_name, self.fixed_effect_config = tuple(self.gdmix_config_obj.fixed_effect_config.items())[0]
        self.output_dir = path_join(gdmix_config_obj.output_dir, self.fixed_effect_name)
        self.output_model_dir = path_join(self.output_dir, MODELS)
        self.validation_score_dir = path_join(self.output_dir, VALIDATION_SCORES)

        self.gdmix_params: Params = Params(**gdmix_config_obj.gdmix_config,
                                           training_score_dir=path_join(self.output_dir, TRAINING_SCORES),
                                           validation_score_dir=self.validation_score_dir)

        self.model_type = self.gdmix_params.model_type

    def get_train_job(self):
        """ Get tfjob training job.
        Return: (job_type, job_name, "", job_params)
        """
        if self.model_type == LOGISTIC_REGRESSION:
            params = self.gdmix_params, FixedLRParams(**self.fixed_effect_config, output_model_dir=self.output_model_dir)
        elif self.model_type == DETEXT:
            detext_arg = DetextArg(**self.fixed_effect_config, out_dir=self.output_model_dir)
            params = replace(self.gdmix_params, training_score_dir=None, validation_score_dir=None), \
                detext_arg._replace(feature_names=[','.join(detext_arg.feature_names)])
        else:
            raise ValueError(f'unsupported model_type: {self.model_type}')
        return GDMIX_TFJOB, f"{self.fixed_effect_name}-tf-train", "", params

    def get_detext_inference_job(self):
        """ Get detext inference job. For LR model the inference job is included in train
        job, this job is for DeText model inference.
        Return: an inferece job inferencing training and validation data
        (job_type, job_name, "", job_params)
        """
        detext_arg = DetextArg(**self.fixed_effect_config, out_dir=self.output_model_dir)
        gdmix_params = replace(self.gdmix_params, action=ACTION_INFERENCE, training_score_dir=None, validation_score_dir=None)
        return GDMIX_TFJOB, f"{self.fixed_effect_name}-tf-inference", "", (gdmix_params, detext_arg._replace(feature_names=[','.join(detext_arg.feature_names)]))

    def get_compute_metric_job(self):
        """ Get sparkjob compute metric job.
        Return: (job_type, job_name, class_name, job_params)
        """
        params = {
            r"\--metricsInputDir": self.validation_score_dir,
            "--outputMetricFile": path_join(self.output_dir, METRIC),
            "--labelColumnName": self.gdmix_params.label_column_name,
            "--predictionColumnName": self.gdmix_params.prediction_score_column_name
        }
        return (GDMIX_SPARKJOB,
                f"{self.fixed_effect_name}-compute-metric",
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
            raise ValueError(f'unsupported model_type: {self.model_type}')
        return jobs
