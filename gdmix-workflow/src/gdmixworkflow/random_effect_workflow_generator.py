from gdmixworkflow.common.utils import *
from gdmixworkflow.common.constants import *
from gdmixworkflow.workflow_generator import WorkflowGenerator
import os
from os.path import join as path_join


class RandomEffectWorkflowGenerator(WorkflowGenerator):
    """ Generate gdmix random effect workflow consisting of
    - sparkjob: partition
    - tfjob: train
    - sparkjob: compute-metric
    """

    def __init__(self, gdmix_config_obj, jar_path="", namespace="",
                 secret_name="", image="", service_account="", job_suffix=""):
        """ Init to generate gdmix random effect workflow. """
        super().__init__(gdmix_config_obj, jar_path, namespace, secret_name, image, service_account, job_suffix)
        self.root_output_dir = gdmix_config_obj.output_dir

    def get_prev_model_score_paths(self, prev_model_name):
        """ Get paths of the scoring results from fixed-effect model or random-effect
        model before it. """
        output_dir = path_join(self.root_output_dir, prev_model_name)
        prev_train_score_path = path_join(output_dir, TRAIN_SCORES)
        prev_validation_score_path = path_join(output_dir, VALIDATION_SCORES)
        return (prev_train_score_path, prev_validation_score_path)

    def get_train_output_paths(self, random_effect_name):
        """ Get output paths for the training job. """
        output_dir = path_join(self.root_output_dir, random_effect_name)
        model_path = path_join(output_dir, MODELS)
        train_score_path = path_join(output_dir, TRAIN_SCORES)
        validation_score_path = path_join(output_dir, VALIDATION_SCORES)

        return (model_path, train_score_path, validation_score_path)

    def get_train_input_paths(self, random_effect_name):
        """ Get input paths for the training job that are from the partition job. """
        output_dir = path_join(self.root_output_dir,
                              path_join(random_effect_name, "partition"))
        train_data_path = path_join(output_dir, "trainingData")
        validation_data_path = path_join(output_dir, "validationData")
        metadata_file = path_join(output_dir,
                                 path_join("metadata","tensor_metadata.json"))
        partition_list_path = path_join(output_dir, "partitionList.txt")
        return (train_data_path, validation_data_path, metadata_file, partition_list_path)

    def get_metric_output_path(self, random_effect_name):
        """ Get metric output path. """
        output_dir = path_join(self.root_output_dir, random_effect_name)
        metric_output_path = path_join(output_dir, METRIC)
        return metric_output_path

    def get_partition_job(self, random_effect_config_obj, prev_model_name):
        """ Get sparkjob partition command.
        Return: [job_type, job_name, "", job_params]
        """
        prev_train_score_path, prev_validation_score_path = \
        self.get_prev_model_score_paths(prev_model_name)

        random_effect_name = random_effect_config_obj.name
        train_data_path, validation_data_path, metadata_file, partition_list_path = \
        self.get_train_input_paths(random_effect_name)

        params = {
            r"\-trainInputDataPath": random_effect_config_obj.train_data_path,
            "--validationInputDataPath": random_effect_config_obj.validation_data_path,
            "-inputMetadataFile": random_effect_config_obj.metadata_file,
            "-partitionEntity": random_effect_config_obj.partition_entity,
            "-numPartitions": random_effect_config_obj.num_partitions,
            "-dataFormat": TFRECORD,
            "-featureBag": random_effect_config_obj.input_column_names.feature_bag,
            "-maxNumOfSamplesPerModel": -1,
            "-minNumOfSamplesPerModel": -1,
            "-trainOutputPartitionDataPath": train_data_path,
            "-validationOutputPartitionDataPath": validation_data_path,
            "-outputMetadataFile": metadata_file,
            "-outputPartitionListFile": partition_list_path,
            "-predictionScore": random_effect_config_obj.output_column_name,
            "-trainInputScorePath": prev_train_score_path,
            "-validationInputScorePath": prev_validation_score_path
        }
        return (GDMIX_SPARKJOB, "{}-partition".format(random_effect_name),
                "com.linkedin.gdmix.data.DataPartitioner", params)

    def get_train_job(self, random_effect_config_obj):
        """ Get tfjob training job.
        Return: [job_type, job_name, "", job_params]
        """
        random_effect_name = random_effect_config_obj.name
        params = {STAGE: RANDOM_EFFECT}
        # get params from config
        flatten_config_obj(params, random_effect_config_obj)
        # adjust param keys
        params.pop("name")
        params[PREDICTION_SCORE] = params.pop(OUTPUT_COLUMN_NAME)
        # adjust training/validation data and metadata path as the output of partition job
        train_data_path, validation_data_path, metadata_file, partition_list_path = \
        self.get_train_input_paths(random_effect_name)
        params[TRAIN_DATA_PATH] = train_data_path
        params[VALIDATION_DATA_PATH] = validation_data_path
        params[METADATA_FILE] = metadata_file
        params[PARTITION_LIST_FILE] = partition_list_path
        # add output params
        model_path, train_score_path, validation_score_path = \
        self.get_train_output_paths(random_effect_name)

        params[MODEL_OUTPUT_DIR] = model_path
        params[TRAINING_OUTPUT_DIR] = train_score_path
        params[VALIDATION_OUTPUT_DIR] = validation_score_path

        params = prefix_dash_dash(params)
        return (GDMIX_TFJOB, "{}-tf-train".format(random_effect_name), "", params)

    def get_compute_metric_job(self, random_effect_config_obj):
        """ Get sparkjob compute metrics command.
        Return: [job_type, job_name, class_name, job_params]
        """
        random_effect_name = random_effect_config_obj.name
        _, _, validation_score_path = self.get_train_output_paths(random_effect_name)
        params = {
            r"\-inputPath": validation_score_path,
            "-outputPath": self.get_metric_output_path(random_effect_name),
            "-labelName": random_effect_config_obj.input_column_names.label,
            "-scoreName": random_effect_config_obj.output_column_name
        }
        return (GDMIX_SPARKJOB, "{}-compute-metric".format(random_effect_name),
                "com.linkedin.gdmix.evaluation.AreaUnderROCCurveEvaluator",
                params)

    def get_job_sequence(self):
        """ Get gdmix job sequence.
        """
        random_effect_config_obj_list = self.gdmix_config_obj.random_effect_config
        jobs = []
        prev_model_name = self.gdmix_config_obj.fixed_effect_config.name

        for re_config_obj in random_effect_config_obj_list:
            jobs.append(self.get_partition_job(re_config_obj, prev_model_name))
            jobs.append(self.get_train_job(re_config_obj))
            jobs.append(self.get_compute_metric_job(re_config_obj))
            prev_model_name = re_config_obj.name
        return jobs
