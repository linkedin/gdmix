from dataclasses import replace
from os.path import join as path_join
from types import SimpleNamespace

from gdmix.models.custom.random_effect_lr_lbfgs_model import REParams
from gdmix.params import Params

from gdmixworkflow.common.constants import *
from gdmixworkflow.workflow_generator import WorkflowGenerator


class RandomEffectWorkflowGenerator(WorkflowGenerator):
    """ Generate gdmix random effect workflow consisting of
    - sparkjob: partition
    - tfjob: train
    - sparkjob: compute-metric
    """
    def __init__(self, gdmix_config_obj, jar_path="", namespace="", secret_name="", image="", service_account="", job_suffix="", *, prev_model_name):
        """ Init to generate gdmix random effect workflow. """
        super().__init__(gdmix_config_obj, jar_path, namespace, secret_name, image, service_account, job_suffix)
        self.prev_model_name = prev_model_name
        self.root_output_dir = gdmix_config_obj.output_dir

    def get_prev_model_score_paths(self, prev_model_name):
        """ Get paths of the scoring results from fixed-effect model or random-effect
        model before it. """
        output_dir = path_join(self.root_output_dir, prev_model_name)
        prev_train_score_dir = path_join(output_dir, TRAINING_SCORES)
        prev_validation_score_dir = path_join(output_dir, VALIDATION_SCORES)
        return prev_train_score_dir, prev_validation_score_dir

    def get_train_output_paths(self, random_effect_name):
        """ Get output paths for the training job. """
        output_dir = path_join(self.root_output_dir, random_effect_name)
        output_model_dir = path_join(output_dir, MODELS)
        training_score_dir = path_join(output_dir, TRAINING_SCORES)
        validation_score_dir = path_join(output_dir, VALIDATION_SCORES)
        return output_model_dir, training_score_dir, validation_score_dir

    def get_train_input_paths(self, random_effect_name):
        """ Get input paths for the training job that are from the partition job. """
        output_dir = path_join(self.root_output_dir, path_join(random_effect_name, "partition"))
        training_data_dir = path_join(output_dir, "trainingData")
        validation_data_dir = path_join(output_dir, "validationData")
        metadata_file = path_join(output_dir, path_join("metadata", "tensor_metadata.json"))
        partition_list_file = path_join(output_dir, "partitionList.txt")
        return training_data_dir, validation_data_dir, metadata_file, partition_list_file

    def get_metric_output_path(self, random_effect_name):
        """ Get metric output path. """
        output_dir = path_join(self.root_output_dir, random_effect_name)
        metric_output_path = path_join(output_dir, METRIC)
        return metric_output_path

    def get_partition_job(self, random_effect_config_obj, prev_model_name, score_column_name):
        """ Get sparkjob partition command.
        Return: [job_type, job_name, "", job_params]
        """
        prev_train_score_dir, prev_validation_score_dir = \
            self.get_prev_model_score_paths(prev_model_name)

        random_effect_name = random_effect_config_obj.name
        training_data_dir, validation_data_dir, metadata_file, partition_list_file = \
            self.get_train_input_paths(random_effect_name)

        params = {
            r"\--trainingDataDir": random_effect_config_obj.training_data_dir,
            "--validationDataDir": random_effect_config_obj.validation_data_dir,
            "--metadataFile": random_effect_config_obj.metadata_file,
            "--partitionId": random_effect_config_obj.partition_entity,
            "--numPartitions": random_effect_config_obj.num_partitions,
            "--dataFormat": TFRECORD,
            "--partitionedTrainingDataDir": training_data_dir,
            "--partitionedValidationDataDir": validation_data_dir,
            "--outputMetadataFile": metadata_file,
            "--outputPartitionListFile": partition_list_file,
            "--predictionScoreColumnName": score_column_name,
            "--trainingScoreDir": prev_train_score_dir,
            "--validationScoreDir": prev_validation_score_dir}
        return GDMIX_SPARKJOB, f"{random_effect_name}-partition", "com.linkedin.gdmix.data.DataPartitioner", params

    def get_train_job(self, name, random_effect_config_obj):
        """ Get tfjob training job.
        Return: [job_type, job_name, "", job_params]
        """
        training_data_dir, validation_data_dir, metadata_file, partition_list_file = self.get_train_input_paths(name)
        output_model_dir, training_score_dir, validation_score_dir = self.get_train_output_paths(name)

        gdmix_params = Params(**self.gdmix_config_obj.gdmix_config, stage=RANDOM_EFFECT,
                              training_score_dir=training_score_dir,
                              validation_score_dir=validation_score_dir,
                              partition_list_file=partition_list_file)
        re_params = replace(REParams(**random_effect_config_obj, output_model_dir=output_model_dir),
                            training_data_dir=training_data_dir, validation_data_dir=validation_data_dir, metadata_file=metadata_file)
        return GDMIX_TFJOB, f"{name}-tf-train", "", (gdmix_params, re_params)

    def get_compute_metric_job(self, random_effect_config_obj, score_column_name, label_column_name):
        """ Get sparkjob compute metrics command.
        Return: [job_type, job_name, class_name, job_params]
        """
        random_effect_name = random_effect_config_obj.name
        _, _, validation_score_dir = self.get_train_output_paths(random_effect_name)
        params = {
            r"\--metricsInputDir": validation_score_dir,
            "--outputMetricFile": self.get_metric_output_path(random_effect_name),
            "--labelColumnName": label_column_name,
            "--predictionColumnName": score_column_name
        }
        return (GDMIX_SPARKJOB, f"{random_effect_name}-compute-metric",
                "com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator",
                params)

    def get_job_sequence(self):
        """ Get gdmix job sequence.
        """
        random_effect_config_obj_list = self.gdmix_config_obj.random_effect_config.items()
        jobs = []
        prev_model_name = self.prev_model_name

        for name, re_config_obj in random_effect_config_obj_list:
            spark_job_conf = SimpleNamespace(name=name, **re_config_obj)
            re_config_obj.pop('num_partitions')
            score_column_name = self.gdmix_config_obj.gdmix_config['prediction_score_column_name']
            label_column_name = self.gdmix_config_obj.gdmix_config['label_column_name']
            jobs.append(self.get_partition_job(spark_job_conf, prev_model_name, score_column_name))
            jobs.append(self.get_train_job(name, re_config_obj))
            jobs.append(self.get_compute_metric_job(spark_job_conf, score_column_name, label_column_name))
            prev_model_name = name
        return jobs
